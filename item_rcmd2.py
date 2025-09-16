# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from collections import defaultdict, deque
from datetime import datetime, timedelta

# =========================================================
# 0) 더미 JSON (refer.json / user_state.json / fr.json) 예시
#    - 실제에선 파일에서 읽어와도 됩니다.
# =========================================================
refer = {
    "users": [
        {"couple_id": 101, "attachment_type": "불안형-2"},
        {"couple_id": 102, "attachment_type": "안정형-1"}
    ],
    "advice": [
        {"advice_id": 1, "advice_comment": "호흡 4-7-8로 긴장 완화 루틴을 시작해요.", "tags": ["호흡","즉시완화"]},
        {"advice_id": 2, "advice_comment": "일과 걱정을 분리하는 '우려 다이어리'를 10분만 써봐요.", "tags": ["기록","인지"]},
        {"advice_id": 3, "advice_comment": "중요도-긴급도 매트릭스로 오늘 할 일을 3개만 뽑아요.", "tags": ["우선순위","실행"]},
        {"advice_id": 4, "advice_comment": "잠들기 1시간 전 스크린 오프, 수면 신호를 일정하게.", "tags": ["수면","위생"]},
        {"advice_id": 5, "advice_comment": "감사 저널: 오늘 고마웠던 것 3가지를 적어봐요.", "tags": ["기록","정서"]}
    ],
    "advice_feature": [
        {"advice_id": 1, "embed": [0.91,0.12,0.03,0.44], "ver": "v1"},
        {"advice_id": 2, "embed": [0.62,0.77,0.11,0.09], "ver": "v1"},
        {"advice_id": 3, "embed": [0.15,0.18,0.94,0.22], "ver": "v1"},
        {"advice_id": 4, "embed": [0.71,0.09,0.21,0.83], "ver": "v1"},
        {"advice_id": 5, "embed": [0.33,0.88,0.14,0.05], "ver": "v1"}
    ],
    "advice_sim": [
        {"advice_id_a": 1, "advice_id_b": 4, "sim": 0.82, "ver":"v1"},
        {"advice_id_a": 2, "advice_id_b": 5, "sim": 0.86, "ver":"v1"},
        {"advice_id_a": 2, "advice_id_b": 3, "sim": 0.44, "ver":"v1"},
        {"advice_id_a": 1, "advice_id_b": 3, "sim": 0.31, "ver":"v1"},
        {"advice_id_a": 4, "advice_id_b": 5, "sim": 0.28, "ver":"v1"}
    ]
}

user_state = {
    "user_state_daily": [
        {"couple_id": 101, "day": "2025-09-12", "anxiety_score": 60, "avoidance_score": 57},
        {"couple_id": 101, "day": "2025-09-15", "anxiety_score": 55, "avoidance_score": 52}
    ],
    "user_feedback": [
        {"couple_id": 101, "advice_id": 2, "label": "good", "ts": "2025-09-10T09:10:00+09:00"},
        {"couple_id": 101, "advice_id": 5, "label": "good", "ts": "2025-09-11T20:40:00+09:00"},
        {"couple_id": 101, "advice_id": 3, "label": "bad",  "ts": "2025-09-12T08:20:00+09:00"}
    ],
    "session_log": []
}

fr_json = {
    "fr_window": {
        "couple_id": 101,
        "from": "2025-08-10",
        "to": "2025-09-14",
        "user_advice_ids": [2,5,3,1],  # (= exposed/selected)
        "good_set": [2,5],
        "bad_set": [3],
        "last_top1": 5
    }
}

# =========================================================
# 1) JSON → 내부 구조 어댑터
#    - item_id = advice_id 로 동일 취급
# =========================================================
def build_interactions_from_feedback(user_state_json, good_val=5.0, bad_val=1.0):
    rows = []
    for fb in user_state_json.get("user_feedback", []):
        v = good_val if fb["label"] == "good" else bad_val
        rows.append({"couple_id": fb["couple_id"], "item_id": fb["advice_id"], "value": v})
    if not rows:
        return pd.DataFrame(columns=["couple_id","item_id","value"])
    return pd.DataFrame(rows)

def build_item_vecs_from_refer(refer_json, ver="v1"):
    vecs = {}
    for f in refer_json.get("advice_feature", []):
        if f.get("ver") == ver:
            vecs[f["advice_id"]] = np.array(f["embed"], dtype=float)
    return vecs

def build_item_meta_from_refer(refer_json):
    # artist 대신 advice tags로 메타 구성
    meta = {}
    for a in refer_json.get("advice", []):
        meta[a["advice_id"]] = {"tags": a.get("tags", []), "comment": a.get("advice_comment","")}
    return meta

def build_item_sim_from_refer(refer_json, ver="v1", item_id_index=None):
    # (선택) 오프라인 유사도 테이블이 있는 경우 바로 스파스화해서 사용
    # 반환: dense matrix (없는 쌍은 0)
    if item_id_index is None:
        return None
    size = len(item_id_index)
    M = np.zeros((size, size), dtype=float)
    for s in refer_json.get("advice_sim", []):
        if s.get("ver") != ver: 
            continue
        a, b, sim = s["advice_id_a"], s["advice_id_b"], float(s["sim"])
        if a in item_id_index and b in item_id_index:
            i = item_id_index[a]; j = item_id_index[b]
            M[i, j] = sim
            M[j, i] = sim
    np.fill_diagonal(M, 0.0)
    return M

# =========================================================
# 2) 전역 상태(원본 코드의 구조 유지 + 어댑터 적용)
# =========================================================
# interactions: feedback → implicit rating (good/bad -> 5/1)
interactions = build_interactions_from_feedback(user_state)

# item_vecs / item_meta
item_vecs = build_item_vecs_from_refer(refer, ver="v1")
item_meta = build_item_meta_from_refer(refer)

# 유틸: 인덱스 매핑 재빌드
def rebuild_indices():
    global uid2idx, iid2idx, idx2uid, idx2iid, num_users, num_items
    couple_ids = sorted(interactions["couple_id"].unique().tolist())
    item_ids = sorted(list(item_vecs.keys()))
    uid2idx = {u:i for i,u in enumerate(couple_ids)}
    iid2idx = {it:i for i,it in enumerate(item_ids)}
    idx2uid = {i:u for u,i in uid2idx.items()}
    idx2iid = {i:it for it,i in iid2idx.items()}
    num_users = len(uid2idx); num_items = len(iid2idx)

def l2_normalize(X, eps=1e-9):
    n = np.linalg.norm(X, axis=1, keepdims=True) + eps
    return X / n

def rebuild_matrices():
    """R, item_mat, user_mat, item_sim을 모두 갱신"""
    global R, item_mat, user_mat, item_sim

    # R
    if len(uid2idx) == 0 or len(iid2idx) == 0:
        R = csr_matrix((0,0), dtype=float)
        item_mat = np.zeros((0,0))
        user_mat = np.zeros((0,0))
        item_sim = np.zeros((0,0))
        return

    rows = interactions["couple_id"].map(uid2idx).values
    cols = interactions["item_id"].map(iid2idx).values
    vals = interactions["value"].astype(float).values
    R = csr_matrix((vals, (rows, cols)), shape=(num_users, num_items))

    # item_mat
    dim = len(next(iter(item_vecs.values()))) if item_vecs else 0
    item_mat = np.zeros((num_items, dim))
    for i_idx, it in idx2iid.items():
        item_mat[i_idx] = item_vecs[it]
    item_mat = l2_normalize(item_mat)

    # user_mat (간이: 가중 평균 → 정규화)
    if num_users > 0 and num_items > 0:
        weights = np.array(R.sum(axis=1)).reshape(-1,1) + 1e-9
        user_mat = (R @ item_mat) / weights
        user_mat = l2_normalize(user_mat)
    else:
        user_mat = np.zeros((0, dim))

    # item_sim: refer.advice_sim 있으면 그걸 사용, 없으면 계산
    preload = build_item_sim_from_refer(refer, ver="v1", item_id_index=iid2idx)
    if preload is not None and preload.shape == (num_items, num_items):
        item_sim = preload
    else:
        item_sim = cosine_similarity(item_mat) if num_items > 0 else np.zeros((0,0))
        np.fill_diagonal(item_sim, 0.0)

# 초기 빌드
rebuild_indices()
rebuild_matrices()

# =========================================================
# 3) 원본 코드의 추천/피드백 로직 (소폭 수정)
# =========================================================
user_blacklist = defaultdict(set)

def map_feedback_to_value(label=None, rating=None, like=None):
    # 우선순위: label -> rating -> like (호환 유지)
    if label in ("good","bad"):
        return 1.0 if label == "good" else 0.0
    if rating is not None:
        return max(0.0, min(1.0, (rating-1)/4))
    if like is True:   return 1.0
    if like is False:  return 0.0
    return None

recent_feedback = defaultdict(lambda: deque(maxlen=200))  # couple_id -> [(item_id, value)]
last_feedback_value = defaultdict(dict)

def update_user_profile(uidx, iidx, value, alpha=0.25, beta=0.15):
    # 사용자 벡터 EMA (좋아요는 당김, 싫어요는 밀어냄)
    if uidx is None or iidx is None: 
        return
    u = user_mat[uidx].copy()
    v = item_mat[iidx]  # L2-normalized
    coef_pos = alpha if value >= 0.5 else -beta
    u_new = (1 - abs(coef_pos)) * u + coef_pos * v
    u_new /= (np.linalg.norm(u_new) + 1e-9)
    user_mat[uidx] = u_new

def register_feedback(couple_id, item_id, label=None, rating=None, like=None):
    """good/bad(label) 또는 rating/like를 받아 상호작용 + 상태 갱신"""
    global interactions
    val01 = map_feedback_to_value(label=label, rating=rating, like=like)
    if val01 is None: 
        return

    # 최근 피드백/마지막값 저장
    recent_feedback[couple_id].append((item_id, val01))
    last_feedback_value[couple_id][item_id] = val01

    # interactions 갱신(1~5 스케일)
    if couple_id in uid2idx and item_id in iid2idx:
        v5 = float(val01)*5.0
        interactions = pd.concat([
            interactions,
            pd.DataFrame({"couple_id":[couple_id], "item_id":[item_id], "value":[v5]})
        ], ignore_index=True)

        # R, user_mat 업데이트 (간이)
        uidx = uid2idx[couple_id]; iidx = iid2idx[item_id]
        # CSR 인플레이스 갱신 대신 간단 재빌드 경로를 사용
        rebuild_matrices()
        # 타이브레이커용 사용자 벡터 소폭 보정
        update_user_profile(uidx, iidx, val01)

def filter_blocked_and_seen(couple_id, scores, like_threshold=0.7, allow_resurface=True):
    seen = set(interactions[interactions["couple_id"]==couple_id]["item_id"])
    blocked = user_blacklist[couple_id]
    if allow_resurface:
        liked = {iid for iid, v in last_feedback_value[couple_id].items() if v >= like_threshold}
        seen -= liked
    for iid in (seen | blocked):
        if iid in iid2idx:
            scores[iid2idx[iid]] = -np.inf

def rerank_with_feedback(couple_id, rec_idx, base_scores, boost=0.15, penalty=0.20):
    if couple_id not in recent_feedback:
        return base_scores
    liked_items   = {iid for iid,val in recent_feedback[couple_id] if val >= 0.7}
    disliked_items= {iid for iid,val in recent_feedback[couple_id] if val <= 0.3}

    # 태그 기반 가산/감산 (artist → tags로 대체)
    def tags(iid): return set(item_meta.get(iid,{}).get("tags", []))
    liked_tags = set().union(*(tags(i) for i in liked_items)) if liked_items else set()
    disliked_tags = set().union(*(tags(i) for i in disliked_items)) if disliked_items else set()

    adj = base_scores.copy()
    for j, iidx in enumerate(rec_idx):
        iid = idx2iid[iidx]
        t = tags(iid)
        if iid in disliked_items or (t & disliked_tags):
            adj[j] *= (1 - penalty)
        if iid in liked_items or (t & liked_tags):
            adj[j] *= (1 + boost)
    return adj

def recommend_for_user(target_user, N=5, topk_item_neighbors=50, normalize_by_sim_sum=True):
    """
    Item-CF 추천:
      - 점수(j) = Σ_{i∈Su} [ sim(i, j) * wu(i) ]
      - Su는 사용자가 본 아이템, wu(i)는 평점(1~5)
    """
    empty_result = []
    if target_user not in uid2idx or item_sim.size == 0:
        return empty_result
    uidx = uid2idx[target_user]

    user_row = R.getrow(uidx).toarray().ravel()  # [I]
    seen_idx = np.where(user_row > 0)[0]
    if seen_idx.size == 0:
        return empty_result

    weights_seen = user_row[seen_idx]  # 1~5

    sims_seen = item_sim[seen_idx]     # [S x I]
    if topk_item_neighbors is not None and topk_item_neighbors < num_items:
        sims_seen_sparse = np.zeros_like(sims_seen)
        k = max(1, topk_item_neighbors)
        part = np.argpartition(-sims_seen, k-1, axis=1)[:, :k]
        rows = np.arange(sims_seen.shape[0])[:, None]
        sims_seen_sparse[rows, part] = sims_seen[rows, part]
        sims_seen = sims_seen_sparse

    scores = weights_seen @ sims_seen  # [I]
    if normalize_by_sim_sum:
        denom = np.abs(sims_seen).sum(axis=0) + 1e-9
        scores = scores / denom

    filter_blocked_and_seen(target_user, scores)

    valid_idx = np.where(np.isfinite(scores) & (scores > -np.inf))[0]
    if valid_idx.size == 0:
        return empty_result

    N_eff = min(N, valid_idx.size)
    cand_scores = scores[valid_idx]
    top_local = np.argpartition(-cand_scores, N_eff-1)[:N_eff]
    top_idx = valid_idx[top_local]
    order = np.argsort(-scores[top_idx])
    top_idx = top_idx[order]

    # 피드백 기반 재랭킹
    new_scores = rerank_with_feedback(target_user, top_idx, scores[top_idx])
    scores[top_idx] = new_scores
    order = np.argsort(-scores[top_idx])
    top_idx = top_idx[order]

    # 동점 타이브레이크: 사용자 프로필과의 코사인
    if top_idx.size > 1:
        uvec = user_mat[uidx:uidx+1]
        item_subset = item_mat[top_idx]
        tie = cosine_similarity(uvec, item_subset).ravel()
        eps = 1e-6
        base = scores[top_idx]
        used = np.zeros(top_idx.size, dtype=bool)
        new_indices = []
        for i in np.argsort(-base):
            if used[i]:
                continue
            mask = np.isclose(base, base[i], atol=eps)
            group = np.where(~used & mask)[0]
            order_g = group[np.argsort(-tie[group])]
            new_indices.extend(order_g.tolist())
            used[group] = True
        top_idx = top_idx[new_indices]

    rec_items = [idx2iid[i] for i in top_idx]
    rec_scores = [float(scores[i]) for i in top_idx]
    return list(zip(rec_items, rec_scores))  # [(advice_id, score)]

# =========================================================
# 4) IR/FR 래퍼(API 응답 형태로 반환)
# =========================================================
def get_latest_scores(couple_id):
    """user_state_daily에서 가장 최근 점수 반환(없으면 None)"""
    rows = [r for r in user_state.get("user_state_daily", []) if r["couple_id"] == couple_id]
    if not rows:
        return None
    rows.sort(key=lambda r: r["day"])
    last = rows[-1]
    return {"anxiety": last["anxiety_score"], "avoidance": last["avoidance_score"]}

def advice_text(advice_id):
    return item_meta.get(advice_id, {}).get("comment", f"advice#{advice_id}")

def neighbors_for(advice_id, K=3):
    """설명용 이웃 (advice_sim에서 상위 K)"""
    if advice_id not in iid2idx:
        return []
    iidx = iid2idx[advice_id]
    sims = item_sim[iidx]
    cand = np.argpartition(-sims, min(K, len(sims)-1))[:K]
    cand = cand[np.argsort(-sims[cand])]
    out = []
    for j in cand:
        if j == iidx: 
            continue
        iid = idx2iid[j]
        s = float(sims[j])
        if s > 0:
            out.append({"advice_id": iid, "sim": round(s, 4)})
    return out

def ir_top1(couple_id):
    """IR Top-1 (현재 상태 기반)"""
    recs = recommend_for_user(couple_id, N=1)
    if not recs:
        return None
    aid, _ = recs[0]
    reason = "과거 good 피드백과 유사 아이템 상위"
    return {
        "advice_id": aid,
        "advice_comment": advice_text(aid),
        "reason": reason,
        "neighbors": neighbors_for(aid, K=3)
    }

def fr_top1(couple_id, window_days=36, mode="itemcf"):
    """FR Top-1 (회고) — 1안(Item-CF) 우선"""
    # 여기서는 간단히: 최근 N일의 good/bad를 이미 interactions에 반영했다고 가정하고
    # recommend_for_user로 동일하게 뽑되, reason만 다르게 표기
    recs = recommend_for_user(couple_id, N=1)
    if not recs:
        return None
    aid, _ = recs[0]
    reason = "36일 good 클러스터 중심 유사도 상위" if mode == "itemcf" else "LLM 조합 요약 상위"
    return {
        "advice_id": aid,
        "advice_comment": advice_text(aid),
        "reason": reason,
        "neighbors": neighbors_for(aid, K=3)
    }

# =========================================================
# 5) 데모
# =========================================================
if __name__ == "__main__":
    target_user = int(input("target user: "))

    for _ in range(36):
        print("=== IR Top-1 (초기) ===")
        print(ir_top1(target_user))

        # feedback이 'good'이면 while만 종료하고, for는 다음 회차로 진행
        while True:
            feedback = input("feedback: ").strip().lower()
            if feedback == "good":
                break  # <-- while만 종료
            print(f"\n[피드백 등록] user={target_user}, advice=4, label={feedback}")
            register_feedback(couple_id=target_user, item_id=4, label=feedback)
            print("=== IR Top-1 (피드백 반영 후) ===")
            print(ir_top1(target_user))

    # FR(1안) 호출
    print("\n=== FR Top-1 (36일 회고, 1안: Item-CF) ===")
    print(fr_top1(target_user, window_days=36, mode="itemcf"))
