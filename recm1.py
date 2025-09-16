import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

# ---------------------------------------
# 0) 입력 데이터 예
# ---------------------------------------
# interactions: 사용자-아이템 상호작용 (명시적 rating 또는 암시적 0/1)
interactions = pd.DataFrame({
    "user_id": [1,1,1,2,2,3,3,4],
    "item_id": [10,11,12,10,13,11,14,12],
    "value":   [5,  4,  1,  3,  5,  4,  1,  2]  # 점수(명시적) 또는 1(암시적)
})

# 아이템 임베딩(벡터) 예시: item_id -> d차원 벡터
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

# 아이템 메타(예시) — 실제 서비스에선 진짜 메타데이터 사용
item_meta = {iid: {"artist": f"artist_{iid % 3}"} for iid in item_vecs.keys()}

user_vecs = None  # 예: {user_id: np.array([...]), ...}


# ---------------------------------------
# 1) user-item 매트릭스 만들기 (희소)
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
# 2) 사용자 벡터 만들기 (아이템 임베딩만 있는 경우)
#    - 각 사용자의 아이템 벡터를 시청/평가 value로 가중 평균
#    - 정규화(ℓ2) 권장
# ---------------------------------------
dim = len(next(iter(item_vecs.values())))
item_mat = np.zeros((num_items, dim))
for i_idx, it in idx2iid.items():
    item_mat[i_idx] = item_vecs[it]  # [I x d]

def l2_normalize(X, eps=1e-9):
    n = np.linalg.norm(X, axis=1, keepdims=True) + eps
    return X / n

if user_vecs is None:
    # R: [U x I], item_mat: [I x d]  => user_mat = R * item_mat / (sum of weights)
    weights = np.array(R.sum(axis=1)).reshape(-1,1) + 1e-9
    user_mat = (R @ item_mat) / weights  # [U x d]
    user_mat = l2_normalize(user_mat)
else:
    # 주어진 user_vecs 사용
    user_mat = np.zeros((num_users, dim))
    for u_idx, u in idx2uid.items():
        user_mat[u_idx] = user_vecs[u]
    user_mat = l2_normalize(user_mat)

# ---------------------------------------
# 3) 사용자-사용자 유사도 (코사인)
#    메모리 고려 시 전체 계산 대신 Top-K 최근접 이웃만 추출(ANN/Faiss 권장)
# ---------------------------------------
user_sim = cosine_similarity(user_mat)  # [U x U]
np.fill_diagonal(user_sim, 0.0)         # 자기 자신 제외

# ---------------------------------------
# 4) 추천 함수
#    - target_user: user_id
#    - K: 이웃 수
#    - N: 추천 아이템 수
#    - seen 제외
#    - 이웃 가중치로 아이템 점수 집계 (유사도 × 이웃의 value)
# ---------------------------------------
def recommend_for_user(target_user, K=20, N=5, min_neighbors=1):
    empty_result = []
    if target_user not in uid2idx:
        return empty_result

    uidx = uid2idx[target_user]
    U = user_mat.shape[0]; I = item_mat.shape[0]
    if U <= 1 or I == 0:
        return empty_result
    K = max(1, min(K, U - 1))

    sims = user_sim[uidx].copy()
    sims[uidx] = 0.0  # 자기 자신 0

    # 이웃 선택
    pos_idx = np.where(sims > 0)[0]
    if pos_idx.size == 0:
        neighbor_idx = np.argsort(-sims)[:K]
    else:
        neighbor_idx = pos_idx[np.argpartition(-sims[pos_idx], min(K, pos_idx.size)-1)[:min(K, pos_idx.size)]]
    neighbor_idx = neighbor_idx[neighbor_idx != uidx]
    if neighbor_idx.size < min_neighbors:
        return empty_result

    # 희소 연산으로 점수 집계
    R_neighbors = R.tocsr()[neighbor_idx]
    weights = sims[neighbor_idx]
    scores = R_neighbors.multiply(weights[:, None]).sum(axis=0).A1  # [I]

    # 차단/본 항목 제외 (여기서 한 번만)
    filter_blocked_and_seen(target_user, scores)

    # 후보 필터
    valid_idx = np.where(np.isfinite(scores) & (scores > -np.inf))[0]
    if valid_idx.size == 0:
        return empty_result

    # 상위 N 1차 선택
    N_eff = min(N, valid_idx.size)
    cand_scores = scores[valid_idx]
    top_local = np.argpartition(-cand_scores, N_eff-1)[:N_eff]
    top_idx = valid_idx[top_local]
    order = np.argsort(-scores[top_idx])
    top_idx = top_idx[order]

    # 피드백 기반 재랭킹
    new_scores = rerank_with_feedback(target_user, top_idx, scores[top_idx])
    # 재랭킹 반영
    scores[top_idx] = new_scores
    order = np.argsort(-scores[top_idx])
    top_idx = top_idx[order]

    # 동점군만 코사인 타이브레이크
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


# 예시: 별점[1~5], 좋아요/싫어요
def map_feedback_to_value(rating=None, like=None):
    if rating is not None:                 # 1~5점 → 0~1 정규화
        return max(0.0, min(1.0, (rating-1)/4))
    if like is True:   return 1.0         
    if like is False:  return 0.0         
    return None

def update_user_profile(user_id, item_id, value, alpha=0.25, beta=0.15):
    if user_id not in uid2idx or item_id not in iid2idx:
        return
    uidx = uid2idx[user_id]; iidx = iid2idx[item_id]

    # 희소 업데이트 최적화: LIL이면 빠름
    if not isinstance(R, csr_matrix):
        R[uidx, iidx] = float(value) * 5.0
    else:
        # CSR이면 비용↑. 가능하면 외부에서 LIL로 유지 권장
        R[uidx, iidx] = float(value) * 5.0

    # 사용자 임베딩 EMA
    u = user_mat[uidx].copy()
    v = item_mat[iidx] / (np.linalg.norm(item_mat[iidx]) + 1e-9)
    u_new = (1 - (alpha if value >= 0.5 else beta)) * u + ((alpha if value >= 0.5 else -beta) * v)
    u_new /= (np.linalg.norm(u_new) + 1e-9)
    user_mat[uidx] = u_new

    # 해당 유저의 유사도만 갱신 + 대각선 0
    sims = cosine_similarity(u_new.reshape(1, -1), user_mat).ravel()
    sims[uidx] = 0.0
    user_sim[uidx] = sims
    user_sim[:, uidx] = sims
    user_sim[uidx, uidx] = 0.0

# 예: 최근 200개 피드백 메모리(또는 DB에서 가져옴)
from collections import defaultdict, deque
recent_feedback = defaultdict(lambda: deque(maxlen=200))  # user_id -> [(item_id, value)]
last_feedback_value = defaultdict(dict) 

def register_feedback(user_id, item_id, rating=None, like=None):
    val = map_feedback_to_value(rating, like)
    if val is None: 
        return
    recent_feedback[user_id].append((item_id, val))
    last_feedback_value[user_id][item_id] = val  
    # interactions에도 반영 (보았음/평가 반영)
    global interactions
    interactions = pd.concat([
        interactions,
        pd.DataFrame({"user_id":[user_id], "item_id":[item_id], "value":[val*5.0]})
    ], ignore_index=True)
    update_user_profile(user_id, item_id, val)

def rerank_with_feedback(user_id, rec_idx, base_scores, boost=0.15, penalty=0.20):
    """rec_idx: 추천된 아이템 인덱스 배열, base_scores: 같은 길이"""
    if user_id not in recent_feedback: 
        return base_scores
    liked_items   = {iid for iid,val in recent_feedback[user_id] if val >= 0.7}
    disliked_items= {iid for iid,val in recent_feedback[user_id] if val <= 0.3}

    # 메타(아티스트/장르) 사전 예시
    def artist(iid): return item_meta.get(iid,{}).get("artist")
    liked_artists = {artist(i) for i in liked_items}
    disliked_artists = {artist(i) for i in disliked_items}

    adj = base_scores.copy()
    for j, iidx in enumerate(rec_idx):
        iid = idx2iid[iidx]
        if iid in disliked_items or artist(iid) in disliked_artists:
            adj[j] *= (1 - penalty)                # 하향
        if iid in liked_items or artist(iid) in liked_artists:
            adj[j] *= (1 + boost)                  # 상향
    return adj

user_blacklist = defaultdict(set)

def dislike_and_block(user_id, item_id):
    user_blacklist[user_id].add(item_id)
    register_feedback(user_id, item_id, like=False)

# 추천 직전 필터
def filter_blocked_and_seen(user_id, scores, like_threshold=0.7, allow_resurface=True):
    seen = set(interactions[interactions["user_id"]==user_id]["item_id"])
    blocked = user_blacklist[user_id]

    # 좋아요(또는 고평점) 항목은 재노출 허용
    if allow_resurface:
        liked = {iid for iid, v in last_feedback_value[user_id].items() if v >= like_threshold}
        seen -= liked  # 좋아요한 건 seen에서 제외

    for iid in (seen | blocked):
        if iid in iid2idx:
            scores[iid2idx[iid]] = -np.inf



# ---------------------------------------
# 5) 예시 실행
# ---------------------------------------
target = int(input("target: "))
print("초기:", recommend_for_user(target, K=5, N=3))

itemid = int(input("item: "))
rate = int(input("rate: "))
register_feedback(target, itemid, rating=rate)
print("좋아요(10) 반영:", recommend_for_user(target, K=5, N=3))

userid = int(input("user: "))
itemid = int(input("item: "))

dislike_and_block(userid, itemid)
print(f"{itemid} 블락 후:", recommend_for_user(target, K=5, N=3))
