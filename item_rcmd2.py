import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from collections import defaultdict, deque
from datetime import datetime, timedelta

# =========================================================
# 0) JSON 파일 로드
# =========================================================
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

refer = load_json("json/refer.json")
user_state = load_json("json/user_state.json")
fr_json = load_json("json/fr.json")



# =========================================================
# 1) JSON → 내부 구조 어댑터
#    - item_id = item_id 로 동일 취급
# =========================================================
def normalize_cid(c):
    """couple_id를 내부 키로 일관되게 문자열화"""
    return str(c)

def user_key(couple_id, member="A"):
    return f"{normalize_cid(couple_id)}_{member.upper()}"

def build_user_info_from_refer(refer_json):
    """couple_id(str) -> {'A': int, 'B': int}"""
    users = {}
    for u in refer_json.get("users", []):
        cid = normalize_cid(u["couple_id"])
        users[cid] = {
            "A": int(u.get("A_attachment_type", 0)),
            "B": int(u.get("B_attachment_type", 0)),
        }
    return users

user_info = build_user_info_from_refer(refer)

def build_interactions_from_feedback(user_state_json, good_val=5.0, bad_val=1.0):
    rows = []
    for fb in user_state_json.get("user_feedback", []):
        v = good_val if fb["label"] == "good" else bad_val
        cid = normalize_cid(fb["couple_id"])
        mem = (fb.get("member") or "A").upper()  # 과거 데이터 호환
        uid = user_key(cid, mem)
        rows.append({"user_id": uid, "couple_id": cid, "member": mem,
                     "item_id": fb["item_id"], "value": v})
    if not rows:
        return pd.DataFrame(columns=["user_id","couple_id","member","item_id","value"])
    return pd.DataFrame(rows)

def build_item_vecs_from_refer(refer_json, ver="v1"):
    vecs = {}
    for f in refer_json.get("advice_feature", []):
        if f.get("ver") == ver:
            vecs[f["item_id"]] = np.array(f["embed"], dtype=float)
    return vecs

def build_item_meta_from_refer(refer_json):
    meta = {}
    for a in refer_json.get("advice", []):
        meta[a["item_id"]] = {"tags": a.get("tags", []), "comment": a.get("advice_comment","")}
    return meta

def build_item_sim_from_refer(refer_json, ver="v1", item_id_index=None):
    if item_id_index is None:
        return None
    size = len(item_id_index)
    M = np.zeros((size, size), dtype=float)
    for s in refer_json.get("advice_sim", []):
        if s.get("ver") != ver: 
            continue
        a, b, sim = s["item_id_a"], s["item_id_b"], float(s["sim"])
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
    user_ids = sorted(interactions["user_id"].unique().tolist())
    item_ids = sorted(list(item_vecs.keys()))
    uid2idx = {u:i for i,u in enumerate(user_ids)}
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

    rows = interactions["user_id"].map(uid2idx).values 
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

ATTACHMENT_TAG_PREF = {
    1: ["마음챙김","호흡","정서"],
    2: ["계획","실행","집중"],
    3: ["관계","경계설정","소셜서포트"],
    4: ["수면","위생","자연"],
}

def apply_member_bias(rec_idx, base_scores, member_type, boost=0.07):
    """애착유형 기반 태그 가중 / rec_idx: 인덱스 배열, base_scores: 해당 스코어 배열"""
    pref = set(ATTACHMENT_TAG_PREF.get(member_type, []))
    if not pref: 
        return base_scores
    adj = base_scores.copy()
    for j, iidx in enumerate(rec_idx):
        iid = idx2iid[iidx]
        t = set(item_meta.get(iid,{}).get("tags", []))
        if t & pref:
            adj[j] *= (1 + boost)
    return adj

# 초기 빌드
rebuild_indices()
rebuild_matrices()

# =========================================================
# 3) 원본 코드의 추천/피드백 로직 (소폭 수정)
# =========================================================
user_blacklist      = defaultdict(set)   # user_id -> {item_id}

def map_feedback_to_value(label=None, rating=None, like=None):
    # 우선순위: label -> rating -> like (호환 유지)
    if label in ("good","bad"):
        return 1.0 if label == "good" else 0.0
    if rating is not None:
        return max(0.0, min(1.0, (rating-1)/4))
    if like is True:   return 1.0
    if like is False:  return 0.0
    return None

recent_feedback     = defaultdict(lambda: deque(maxlen=200))  # user_id -> [...]
last_feedback_value = defaultdict(dict)  # user_id -> {item_id: val}

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

def register_feedback(couple_id, item_id, label=None, rating=None, like=None, member="A"):
    global interactions
    val01 = map_feedback_to_value(label=label, rating=rating, like=like)
    if val01 is None:
        return
    cid = normalize_cid(couple_id)
    uid = user_key(cid, member)

    # 최근/마지막값 갱신
    recent_feedback[uid].append((item_id, val01))
    last_feedback_value[uid][item_id] = val01

    # interactions 추가
    if item_id in iid2idx:
        v5 = float(val01) * 5.0
        new_row = pd.DataFrame({
            "user_id":   [uid],
            "couple_id": [cid],
            "member":    [member],
            "item_id":   [item_id],
            "value":     [v5],
        })
        interactions = pd.concat([interactions, new_row], ignore_index=True)

        # 인덱스/행렬 재생성 후 최신 인덱스로 업데이트
        rebuild_indices()
        uidx = uid2idx.get(uid)
        iidx = iid2idx.get(item_id)
        rebuild_matrices()
        if uidx is not None and iidx is not None:
            update_user_profile(uidx, iidx, val01)

    if val01 <= 0.3:
        user_blacklist[uid].add(item_id)

def filter_blocked_and_seen(user_id, scores, like_threshold=0.7, allow_resurface=False):
    seen = set(interactions[interactions["user_id"]==user_id]["item_id"])
    blocked = user_blacklist[user_id]
    liked = {iid for iid,v in last_feedback_value[user_id].items() if v >= like_threshold}
    bads  = {iid for iid,v in last_feedback_value[user_id].items() if v <= 0.3}
    if allow_resurface:
        seen -= liked
    for iid in (seen | blocked | bads):
        if iid in iid2idx:
            scores[iid2idx[iid]] = -np.inf

def rerank_with_feedback(user_id, rec_idx, base_scores, boost=0.15, penalty=0.20):
    if user_id not in recent_feedback:
        return base_scores
    liked_items    = {iid for iid, val in recent_feedback[user_id] if val >= 0.7}
    disliked_items = {iid for iid, val in recent_feedback[user_id] if val <= 0.3}
    def tags(iid): return set(item_meta.get(iid, {}).get("tags", []))
    liked_tags    = set().union(*(tags(i) for i in liked_items)) if liked_items else set()
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

def recommend_for_user(user_id, N=5, topk_item_neighbors=None, normalize_by_sim_sum=True):
    empty_result = []
    if user_id not in uid2idx or item_sim.size == 0:
        return empty_result

    uidx = uid2idx[user_id]
    user_row = R.getrow(uidx).toarray().ravel()  # [I]
    seen_idx = np.where(user_row > 0)[0]
    if seen_idx.size == 0:
        return empty_result

    weights_seen = user_row[seen_idx]  # 1~5
    sims_seen = item_sim[seen_idx]     # [S x I]

    # 아이템 수 체크
    if sims_seen.shape[1] == 0:
        return empty_result

    # top-k 이웃 제한
    if topk_item_neighbors is not None and topk_item_neighbors < num_items:
        I = sims_seen.shape[1]
        if I == 0:
            return empty_result
        k = max(1, min(topk_item_neighbors, I))
        if I > 1 and k < I:
            sims_seen_sparse = np.zeros_like(sims_seen)
            part = np.argpartition(-sims_seen, k-1, axis=1)[:, :k]
            rows = np.arange(sims_seen.shape[0])[:, None]
            sims_seen_sparse[rows, part] = sims_seen[rows, part]
            sims_seen = sims_seen_sparse

    scores = weights_seen @ sims_seen  # [I]
    if normalize_by_sim_sum:
        denom = np.abs(sims_seen).sum(axis=0) + 1e-9
        scores = scores / denom

    # 필터링 후 빈 배열 방어
    filter_blocked_and_seen(user_id, scores)
    valid_idx = np.where(np.isfinite(scores) & (scores > -np.inf))[0]
    if valid_idx.size == 0:
        return empty_result

    N_eff = min(N, valid_idx.size)
    if N_eff == 0:
        return empty_result

    cand_scores = scores[valid_idx]
    top_local = np.argpartition(-cand_scores, N_eff-1)[:N_eff]
    top_idx = valid_idx[top_local]
    order = np.argsort(-scores[top_idx])
    top_idx = top_idx[order]

    # 빈 배열 방어
    if top_idx.size == 0:
        return empty_result

    rec_items = [idx2iid[i] for i in top_idx]
    rec_scores = [float(scores[i]) for i in top_idx]
    return list(zip(rec_items, rec_scores))

# =========================================================
# 4) IR/FR 래퍼(API 응답 형태로 반환)
# =========================================================
def get_latest_scores(couple_id):
    """user_state_daily에서 가장 최근 점수 반환(없으면 None)"""
    cid = normalize_cid(couple_id)
    rows = [r for r in user_state.get("user_state_daily", []) if normalize_cid(r["couple_id"]) == cid]
    if not rows:
        return None
    rows.sort(key=lambda r: r["day"])
    last = rows[-1]
    return {"anxiety": last["anxiety_score"], "avoidance": last["avoidance_score"]}

def advice_text(item_id):
    return item_meta.get(item_id, {}).get("comment", f"advice#{item_id}")

def safe_neighbors_for(item_id, K=3):
    """설명용 이웃 (advice_sim에서 상위 K) — 소형 케이스 안전"""
    if item_id not in iid2idx or item_sim.size == 0:
        return []
    iidx = iid2idx[item_id]
    sims = item_sim[iidx]
    n = sims.shape[0]
    if n <= 1:
        return []
    k = max(1, min(K, n - 1))
    cand = np.argpartition(-sims, k)[:k+1]  # 여유분 확보
    cand = cand[np.argsort(-sims[cand])]
    out = []
    for j in cand:
        if j == iidx:
            continue
        iid = idx2iid[j]
        s = float(sims[j])
        if s > 0:
            out.append({"item_id": iid, "sim": round(s, 4)})
        if len(out) >= K:
            break
    return out

# neighbors_for 별칭
neighbors_for = safe_neighbors_for

def ir_top1(couple_id, member="A"):
    uid = user_key(couple_id, member)
    recs = recommend_for_user(uid, N=1, topk_item_neighbors=None)
    if not recs:
        return None
    aid, _ = recs[0]
    return {
        "item_id": aid,
        "advice_comment": advice_text(aid),
        "reason": f"{member} 독립 학습 + ItemCF",
        "neighbors": neighbors_for(aid, K=3),
    }

def fr_top1(couple_id, member="A", window_days=36, mode="itemcf"):
    uid = user_key(couple_id, member)
    recs = recommend_for_user(uid, N=1)
    if not recs:
        return None
    aid, _ = recs[0]
    reason = "36일 good 클러스터 중심 유사도 상위" if mode == "itemcf" else "LLM 조합 요약 상위"
    return {
        "item_id": aid,
        "advice_comment": advice_text(aid),
        "reason": reason,
        "neighbors": neighbors_for(aid, K=3)
    }

def ir_top1_member(couple_id, member="A"):
    uid = user_key(couple_id, member)
    recs = recommend_for_user(uid, N=10, topk_item_neighbors=None)
    if not recs:
        return None

    idxs = [iid2idx[aid] for aid, _ in recs if aid in iid2idx]
    scores = [s for _, s in recs]
    if not idxs or not scores:
        return None

    idxs = np.array(idxs, dtype=int)
    scores = np.array(scores, dtype=float)

    # 애착유형 바이어스 반영 (couple_id 정규화 후 조회)
    mtype = user_info.get(normalize_cid(couple_id), {}).get(member, 0)
    if mtype:
        scores = apply_member_bias(idxs, scores, mtype, boost=0.10)

    order = np.argsort(-scores)
    if order.size == 0:
        return None

    top_iidx = idxs[order[0]]
    aid = idx2iid[top_iidx]
    return {
        "item_id": aid,
        "advice_comment": advice_text(aid),
        "reason": f"{member} 독립 학습 + ItemCF + 애착유형 가중",
        "neighbors": neighbors_for(aid, K=3),
    }

def ir_top1_AB(couple_id):
    return {
        "A": ir_top1_member(couple_id, "A"),
        "B": ir_top1_member(couple_id, "B"),
    }

# =========================================================
# 5) 데모
# =========================================================
if __name__ == "__main__":
    # 문자열/숫자 모두 입력 가능 (내부적으로 문자열 키로 통일)
    target_user = input("target couple_id (예: cpl_abc123 또는 101): ").strip()
    if not target_user:
        raise SystemExit("유효한 couple_id를 입력하세요. 예: cpl_abc123")

    for turn in range(36):
        print(f"\n=== IR Top-1 (회차 {turn+1}) ===")
        ab = ir_top1_AB(target_user)  # A와 B 동시에 추천

        a_rec = ab.get("A")
        b_rec = ab.get("B")

        print("A추천:", a_rec)
        print("B추천:", b_rec)
        print("허용 입력 예시: 'A good', 'B bad', 'A bad, B good', 'next', 'quit'")

        current_aid_A = a_rec["item_id"] if isinstance(a_rec, dict) else None
        current_aid_B = b_rec["item_id"] if isinstance(b_rec, dict) else None

        while True:
            feedback = input("feedback [A/B good|bad ... | next | quit]: ").strip().lower()
            if feedback == "quit":
                raise SystemExit(0)
            if feedback in ("next", "skip", ""):
                # 이 회차 종료 → 다음 회차로
                break

            # 한 줄에 여러 건: "A good, B bad" 처럼 쉼표로 구분
            commands = [c.strip() for c in feedback.split(",") if c.strip()]
            for cmd in commands:
                parts = cmd.split()
                if len(parts) != 2 or parts[0] not in ("a","b") or parts[1] not in ("good","bad"):
                    print("형식: 'A good, B bad' 또는 'next'"); 
                    continue

                member, label = parts[0].upper(), parts[1]
                target_aid = current_aid_A if member == "A" else current_aid_B
                if target_aid is None:
                    print(f"[{member}] 현재 추천이 없습니다.")
                    continue

                print(f"[피드백] couple={target_user}, member={member}, advice={target_aid}, label={label}")
                register_feedback(couple_id=target_user, item_id=target_aid, label=label, member=member)

                # 'good'은 같은 회차에서 재추천하지 않음. 'bad'만 즉시 재추천 + 안티 스턱.
                if label == "bad":
                    new_rec = ir_top1_member(target_user, member)
                    if new_rec and new_rec["item_id"] == target_aid:
                        user_blacklist[user_key(target_user, member)].add(target_aid)
                        new_rec = ir_top1_member(target_user, member)

                    if member == "A":
                        current_aid_A = new_rec["item_id"] if new_rec else None
                        if new_rec:
                            print("A 재추천:", new_rec)
                    else:
                        current_aid_B = new_rec["item_id"] if new_rec else None
                        if new_rec:
                            print("B 재추천:", new_rec)

    # FR(1안) 호출
    print("\n=== FR Top-1 (36일 회고, 1안: Item-CF) ===")
    print(fr_top1(target_user, window_days=36, mode="itemcf"))
