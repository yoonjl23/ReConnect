import numpy as np
import pandas as pd  # <-- 오타 방지: 실제로는 'import pandas as pd' 사용하세요
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from collections import defaultdict, deque

# ---------------------------------------
# 0) 입력 데이터 예 (동일)
# ---------------------------------------
interactions = pd.DataFrame({
    "user_id": [1,1,1,2,2,3,3,4],
    "item_id": [10,11,12,10,13,11,14,12],
    "value":   [5,  4,  1,  3,  5,  4,  1,  2]
})

item_vecs = {
    10: np.array([0.9, 0.1, 0.0]),
    11: np.array([0.8, 0.2, 0.0]),
    12: np.array([0.1, 0.9, 0.2]),
    13: np.array([0.0, 0.8, 0.3]),
    14: np.array([0.0, 0.1, 0.9]),
}
item_df = pd.DataFrame(
    [(iid, vec) for iid, vec in item_vecs.items()],
    columns=["item_id", "vec"]
)

item_meta = {iid: {"artist": f"artist_{iid % 3}"} for iid in item_vecs.keys()}

user_vecs = None  # 예: {user_id: np.array([...]), ...}

# ---------------------------------------
# 1) user-item 매트릭스 (희소)
# ---------------------------------------
user_ids = interactions["user_id"].unique()
item_ids = interactions["item_id"].unique()
uid2idx = {u:i for i,u in enumerate(sorted(user_ids))}
iid2idx = {it:i for i,it in enumerate(sorted(item_ids))}
idx2uid = {i:u for u,i in uid2idx.items()}
idx2iid = {i:it for it,i in iid2idx.items()}

rows = interactions["user_id"].map(uid2idx).values
cols = interactions["item_id"].map(iid2idx).values
vals = interactions["value"].astype(float).values
num_users = len(uid2idx); num_items = len(iid2idx)
R = csr_matrix((vals, (rows, cols)), shape=(num_users, num_items))  # [U x I]

# ---------------------------------------
# 2) 사용자 벡터(옵션) & 아이템 행렬
#    - user_mat은 tie-breaker용으로 유지 (선택)
# ---------------------------------------
dim = len(next(iter(item_vecs.values())))
item_mat = np.zeros((num_items, dim))
for i_idx, it in idx2iid.items():
    item_mat[i_idx] = item_vecs[it]  # [I x d]

def l2_normalize(X, eps=1e-9):
    n = np.linalg.norm(X, axis=1, keepdims=True) + eps
    return X / n

item_mat = l2_normalize(item_mat)

if user_vecs is None:
    weights = np.array(R.sum(axis=1)).reshape(-1,1) + 1e-9
    user_mat = (R @ item_mat) / weights  # [U x d]
    user_mat = l2_normalize(user_mat)
else:
    user_mat = np.zeros((num_users, dim))
    for u_idx, u in idx2uid.items():
        user_mat[u_idx] = user_vecs[u]
    user_mat = l2_normalize(user_mat)

# ---------------------------------------
# 3) 아이템-아이템 유사도 (코사인)  ← ★ 변경 핵심
# ---------------------------------------
item_sim = cosine_similarity(item_mat)  # [I x I]
np.fill_diagonal(item_sim, 0.0)         # 자기 자신 제외

# ---------------------------------------
# 4) 추천 함수 (Item-CF)
#    - target_user가 본 아이템들의 가중합으로 후보 점수 계산
#    - seen/blocked 제외, 피드백 기반 재랭킹, tie-breaker 유지
# ---------------------------------------
user_blacklist = defaultdict(set)

def map_feedback_to_value(rating=None, like=None):
    if rating is not None:
        return max(0.0, min(1.0, (rating-1)/4))
    if like is True:   return 1.0         
    if like is False:  return 0.0         
    return None

# 최근 피드백 저장
recent_feedback = defaultdict(lambda: deque(maxlen=200))  # user_id -> [(item_id, value)]
last_feedback_value = defaultdict(dict)

def register_feedback(user_id, item_id, rating=None, like=None):
    val = map_feedback_to_value(rating, like)
    if val is None: 
        return
    recent_feedback[user_id].append((item_id, val))
    last_feedback_value[user_id][item_id] = val

    # interactions에도 반영
    global interactions
    interactions = pd.concat([
        interactions,
        pd.DataFrame({"user_id":[user_id], "item_id":[item_id], "value":[val*5.0]})
    ], ignore_index=True)

    # 점진적 사용자 벡터 업데이트(타이브레이커용)
    update_user_profile(user_id, item_id, val)

def update_user_profile(user_id, item_id, value, alpha=0.25, beta=0.15):
    # 사용자 벡터 EMA (좋아요는 당김, 싫어요는 밀어냄)
    if user_id not in uid2idx or item_id not in iid2idx:
        return
    uidx = uid2idx[user_id]; iidx = iid2idx[item_id]

    # R 갱신 (희소: CSR 인플레이스 갱신은 비용 큼 → 간단히 재할당)
    R[uidx, iidx] = float(value) * 5.0

    u = user_mat[uidx].copy()
    v = item_mat[iidx]  # 이미 L2 정규화됨
    coef_pos = alpha if value >= 0.5 else -beta
    u_new = (1 - abs(coef_pos)) * u + coef_pos * v
    u_new /= (np.linalg.norm(u_new) + 1e-9)
    user_mat[uidx] = u_new

def dislike_and_block(user_id, item_id):
    user_blacklist[user_id].add(item_id)
    register_feedback(user_id, item_id, like=False)

def filter_blocked_and_seen(user_id, scores, like_threshold=0.7, allow_resurface=True):
    seen = set(interactions[interactions["user_id"]==user_id]["item_id"])
    blocked = user_blacklist[user_id]
    if allow_resurface:
        liked = {iid for iid, v in last_feedback_value[user_id].items() if v >= like_threshold}
        seen -= liked
    for iid in (seen | blocked):
        if iid in iid2idx:
            scores[iid2idx[iid]] = -np.inf

def rerank_with_feedback(user_id, rec_idx, base_scores, boost=0.15, penalty=0.20):
    if user_id not in recent_feedback: 
        return base_scores
    liked_items   = {iid for iid,val in recent_feedback[user_id] if val >= 0.7}
    disliked_items= {iid for iid,val in recent_feedback[user_id] if val <= 0.3}

    def artist(iid): return item_meta.get(iid,{}).get("artist")
    liked_artists = {artist(i) for i in liked_items}
    disliked_artists = {artist(i) for i in disliked_items}

    adj = base_scores.copy()
    for j, iidx in enumerate(rec_idx):
        iid = idx2iid[iidx]
        if iid in disliked_items or artist(iid) in disliked_artists:
            adj[j] *= (1 - penalty)
        if iid in liked_items or artist(iid) in liked_artists:
            adj[j] *= (1 + boost)
    return adj

def recommend_for_user(target_user, N=5, topk_item_neighbors=50, normalize_by_sim_sum=True):
    """
    Item-CF 추천:
      - 사용자 u가 본 아이템 집합 Su와 그 가중치 wu(i)를 바탕으로
        점수(j) = Σ_{i∈Su} [ sim(i, j) * wu(i) ]
      - 필요 시 후보별로 상위 Top-K 이웃 sim만 사용(가속)
    """
    empty_result = []
    if target_user not in uid2idx:
        return empty_result
    uidx = uid2idx[target_user]

    # 사용자가 본 아이템 인덱스와 가중치(평점)
    user_row = R.getrow(uidx).toarray().ravel()  # [I]
    seen_idx = np.where(user_row > 0)[0]
    if seen_idx.size == 0 or item_sim.shape[0] == 0:
        return empty_result

    weights_seen = user_row[seen_idx]  # 원래 1~5 스케일

    # 후보 점수: seen 아이템들의 유사도 행만 모아서 가중합
    #   sims_seen: [S x I], weights: [S] → scores: [I]
    sims_seen = item_sim[seen_idx]  # [S x I]

    # 선택적으로 seen별 이웃을 Top-K로 스파스화(속도/잡음 제거)
    if topk_item_neighbors is not None and topk_item_neighbors < num_items:
        sims_seen_sparse = np.zeros_like(sims_seen)
        k = max(1, topk_item_neighbors)
        # 각 행마다 top-k만 남김
        part = np.argpartition(-sims_seen, k-1, axis=1)[:, :k]
        rows = np.arange(sims_seen.shape[0])[:, None]
        sims_seen_sparse[rows, part] = sims_seen[rows, part]
        sims_seen = sims_seen_sparse

    scores = weights_seen @ sims_seen  # [I]

    # 정규화(선택): 유사도 절대값 합으로 나눠 편향 완화
    if normalize_by_sim_sum:
        denom = np.abs(sims_seen).sum(axis=0) + 1e-9
        scores = scores / denom

    # 차단/본 항목 제외
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
    return list(zip(rec_items, rec_scores))

# ---------------------------------------
# 5) 예시 실행 (입/출력 부분은 동일 컨셉)
# ---------------------------------------
if __name__ == "__main__":
    target = int(input("target: "))
    print("초기:", recommend_for_user(target, N=3))

    itemid = int(input("item: "))
    rate = int(input("rate: "))
    register_feedback(target, itemid, rating=rate)
    print("피드백 반영:", recommend_for_user(target, N=3))

    userid = int(input("user: "))
    itemid = int(input("item: "))
    dislike_and_block(userid, itemid)
    print(f"{itemid} 블락 후:", recommend_for_user(target, N=3))
