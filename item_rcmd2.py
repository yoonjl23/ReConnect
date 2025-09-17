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
    {"couple_id": "cpl_abc123", "A_attachment_type": 2, "B_attachment_type": 1},
    {"couple_id": "cpl_abc124", "A_attachment_type": 3, "B_attachment_type": 2}
    ],
    "advice": [
      {"advice_id": 1, "advice_comment": "호흡 4-7-8로 3회 반복해 긴장을 낮춰보세요.", "tags": ["호흡","즉시완화"]},
      {"advice_id": 2, "advice_comment": "오늘 걱정거리 3가지를 종이에 쓰고 현실성 점수를 매겨요.", "tags": ["기록","인지"]},
      {"advice_id": 3, "advice_comment": "중요도-긴급도 매트릭스로 오늘 할 일 3개만 선정해요.", "tags": ["우선순위","실행"]},
      {"advice_id": 4, "advice_comment": "잠들기 1시간 전 스크린 오프 규칙을 지켜요.", "tags": ["수면","위생"]},
      {"advice_id": 5, "advice_comment": "감사 일기: 오늘 고마웠던 3가지를 적어보세요.", "tags": ["기록","정서"]},
      {"advice_id": 6, "advice_comment": "5분 타이머를 켜고 책상 위 불필요한 물건을 치워요.", "tags": ["정리","실행"]},
      {"advice_id": 7, "advice_comment": "10분 산책으로 심박수를 부드럽게 올리고 내려요.", "tags": ["운동","자연"]},
      {"advice_id": 8, "advice_comment": "물 한 컵을 천천히 마시며 현재 몸 감각에 주의 집중.", "tags": ["수분","마음챙김"]},
      {"advice_id": 9, "advice_comment": "오늘 해야 할 최소 행동 1가지를 정하고 바로 시작.", "tags": ["미세습관","실행"]},
      {"advice_id": 10, "advice_comment": "몸 스캔 명상 3분: 발끝→머리까지 천천히 관찰.", "tags": ["명상","마음챙김"]},
      {"advice_id": 11, "advice_comment": "루미네이션(반복 생각)을 멈추는 신호어를 정해요.", "tags": ["인지","루미네이션중단"]},
      {"advice_id": 12, "advice_comment": "하루 카페인 섭취 시간을 오후 2시 이전으로 제한.", "tags": ["위생","카페인절감"]},
      {"advice_id": 13, "advice_comment": "자기연민 문장: ‘그럴 수 있어, 나는 최선을 다했어’.", "tags": ["자기연민","정서"]},
      {"advice_id": 14, "advice_comment": "태양광 10분 쬐기: 아침 햇빛으로 각성 리듬 세팅.", "tags": ["수면","자연"]},
      {"advice_id": 15, "advice_comment": "포모도로 25분 집중 + 5분 휴식 1세트 진행.", "tags": ["집중","타임박싱"]},
      {"advice_id": 16, "advice_comment": "복식호흡 10회: 배가 부풀었다가 납작해지는 감각에 집중.", "tags": ["호흡","마음챙김"]},
      {"advice_id": 17, "advice_comment": "부정 자동사고를 증거 기반으로 재구성해요.", "tags": ["인지","재구성"]},
      {"advice_id": 18, "advice_comment": "오늘의 ‘해야 할 일’ 대신 ‘할 수 있는 일’ 목록 작성.", "tags": ["인지","기록"]},
      {"advice_id": 19, "advice_comment": "스트레칭 3분: 목, 어깨, 허리 순서로 이완.", "tags": ["운동","이완"]},
      {"advice_id": 20, "advice_comment": "저녁 식사에 단백질 한 가지를 추가해요.", "tags": ["식사","위생"]},
      {"advice_id": 21, "advice_comment": "디지털 디톡스: 알림 끄고 20분 무알림 구간 만들기.", "tags": ["디지털디톡스","집중"]},
      {"advice_id": 22, "advice_comment": "스스로에게 격려 편지 한 줄을 써요.", "tags": ["정서","기록"]},
      {"advice_id": 23, "advice_comment": "할 일을 2분이면 시작 가능한 ‘첫 동작’으로 쪼개요.", "tags": ["실행","미세습관"]},
      {"advice_id": 24, "advice_comment": "감정 라벨링: 지금 감정을 단어 1~2개로 이름 붙이기.", "tags": ["감정레이블링","인지"]},
      {"advice_id": 25, "advice_comment": "수면 루틴 신호(세안→낮은 조명→차분한 음악) 고정.", "tags": ["수면","루틴"]},
      {"advice_id": 26, "advice_comment": "걱정일기 10분, 끝나면 ‘내일 확인’ 포스트잇 붙이기.", "tags": ["기록","경계설정"]},
      {"advice_id": 27, "advice_comment": "사회적 지지 1명에게 안부 메시지 보내기.", "tags": ["관계","소셜서포트"]},
      {"advice_id": 28, "advice_comment": "오늘의 에너지 10점 만점 자가평가 후 일정 조정.", "tags": ["자기점검","계획"]},
      {"advice_id": 29, "advice_comment": "미루는 일 1개만 5분만 해보기 규칙.", "tags": ["미루기대응","실행"]},
      {"advice_id": 30, "advice_comment": "핵심가치 3개를 적고 오늘 행동과의 연결점 찾기.", "tags": ["가치정렬","인지"]},
      {"advice_id": 31, "advice_comment": "박자 호흡(4초 들숨/4초 멈춤/4초 날숨/4초 멈춤) 2분.", "tags": ["호흡","즉시완화"]},
      {"advice_id": 32, "advice_comment": "내일의 나에게 보내는 할 일 메모 1줄 남기기.", "tags": ["기록","계획"]},
      {"advice_id": 33, "advice_comment": "산책 중 주변에서 파란색 5가지를 찾아보기.", "tags": ["마음챙김","자연"]},
      {"advice_id": 34, "advice_comment": "부정 피드백을 성장 질문으로 바꿔 보기.", "tags": ["인지","재구성"]},
      {"advice_id": 35, "advice_comment": "수분 목표: 물 6~8컵 중, 지금 1컵 실천.", "tags": ["수분","위생"]},
      {"advice_id": 36, "advice_comment": "업무 시작 전 1분간 오늘의 의도(intention) 정하기.", "tags": ["의도설정","집중"]},
      {"advice_id": 37, "advice_comment": "완벽주의 체크: ‘충분히 괜찮음’ 기준 정하기.", "tags": ["인지","완벽주의"]},
      {"advice_id": 38, "advice_comment": "소음 차단(이어플러그/화이트노이즈)로 집중 환경 만들기.", "tags": ["집중","환경"]},
      {"advice_id": 39, "advice_comment": "업무 마무리 5분 리뷰: 잘한 1가지 기록.", "tags": ["기록","회고"]},
      {"advice_id": 40, "advice_comment": "하루 한 끼는 천천히 20분 이상 씹어 먹기.", "tags": ["식사","마음챙김"]},
      {"advice_id": 41, "advice_comment": "SNS 사용시간 목표 30분, 초과 시 알림 설정.", "tags": ["디지털디톡스","경계설정"]},
      {"advice_id": 42, "advice_comment": "작은 약속 지키기: 오늘 스스로 한 약속 1개 체크.", "tags": ["자기효능감","미세습관"]},
      {"advice_id": 43, "advice_comment": "‘할 일’→‘할 이유’로 문장을 바꿔 동기 부여.", "tags": ["동기","인지"]},
      {"advice_id": 44, "advice_comment": "업무 블록 사이 3분 창밖 보기로 시각 피로 해소.", "tags": ["휴식","자연"]},
      {"advice_id": 45, "advice_comment": "오늘 배운 점 1가지를 기록해 내일에 연결.", "tags": ["학습","기록"]},
      {"advice_id": 46, "advice_comment": "관계 경계 한 문장 연습: ‘지금은 어려워’.", "tags": ["관계","경계설정"]},
      {"advice_id": 47, "advice_comment": "자원 목록 만들기: 나를 도와주는 사람/장소/활동 5개.", "tags": ["자원목록","회복력"]},
      {"advice_id": 48, "advice_comment": "심박수 올리는 계단 오르기 2층만 실천.", "tags": ["운동","즉시활성"]},
      {"advice_id": 49, "advice_comment": "하루 목표 1문장: ‘오늘은 ~만 한다’.", "tags": ["집중","계획"]},
      {"advice_id": 50, "advice_comment": "자기긍정 확언 3회: ‘나는 점진적으로 나아지고 있어’.", "tags": ["긍정확언","정서"]}
    ],

    "advice_feature": [
        {"advice_id": 1, "embed": [0.12,0.84,0.55,0.21], "ver": "v1"},
        {"advice_id": 2, "embed": [0.67,0.14,0.88,0.09], "ver": "v1"},
        {"advice_id": 3, "embed": [0.32,0.45,0.71,0.52], "ver": "v1"},
        {"advice_id": 4, "embed": [0.93,0.12,0.44,0.28], "ver": "v1"},
        {"advice_id": 5, "embed": [0.18,0.77,0.39,0.61], "ver": "v1"},
        {"advice_id": 6, "embed": [0.41,0.23,0.94,0.07], "ver": "v1"},
        {"advice_id": 7, "embed": [0.62,0.15,0.48,0.81], "ver": "v1"},
        {"advice_id": 8, "embed": [0.75,0.34,0.21,0.59], "ver": "v1"},
        {"advice_id": 9, "embed": [0.52,0.88,0.19,0.23], "ver": "v1"},
        {"advice_id": 10, "embed": [0.13,0.55,0.68,0.44], "ver": "v1"},
        {"advice_id": 11, "embed": [0.81,0.27,0.52,0.34], "ver": "v1"},
        {"advice_id": 12, "embed": [0.36,0.93,0.18,0.42], "ver": "v1"},
        {"advice_id": 13, "embed": [0.25,0.61,0.77,0.19], "ver": "v1"},
        {"advice_id": 14, "embed": [0.44,0.82,0.13,0.68], "ver": "v1"},
        {"advice_id": 15, "embed": [0.92,0.33,0.28,0.51], "ver": "v1"},
        {"advice_id": 16, "embed": [0.47,0.59,0.61,0.36], "ver": "v1"},
        {"advice_id": 17, "embed": [0.83,0.12,0.22,0.77], "ver": "v1"},
        {"advice_id": 18, "embed": [0.61,0.74,0.49,0.31], "ver": "v1"},
        {"advice_id": 19, "embed": [0.33,0.45,0.71,0.86], "ver": "v1"},
        {"advice_id": 20, "embed": [0.18,0.94,0.25,0.66], "ver": "v1"},
        {"advice_id": 21, "embed": [0.75,0.22,0.64,0.41], "ver": "v1"},
        {"advice_id": 22, "embed": [0.27,0.73,0.56,0.39], "ver": "v1"},
        {"advice_id": 23, "embed": [0.55,0.62,0.41,0.22], "ver": "v1"},
        {"advice_id": 24, "embed": [0.64,0.15,0.83,0.29], "ver": "v1"},
        {"advice_id": 25, "embed": [0.21,0.84,0.19,0.73], "ver": "v1"},
        {"advice_id": 26, "embed": [0.91,0.45,0.23,0.35], "ver": "v1"},
        {"advice_id": 27, "embed": [0.44,0.72,0.38,0.56], "ver": "v1"},
        {"advice_id": 28, "embed": [0.36,0.15,0.94,0.81], "ver": "v1"},
        {"advice_id": 29, "embed": [0.53,0.22,0.62,0.48], "ver": "v1"},
        {"advice_id": 30, "embed": [0.61,0.39,0.45,0.71], "ver": "v1"},
        {"advice_id": 31, "embed": [0.24,0.82,0.19,0.55], "ver": "v1"},
        {"advice_id": 32, "embed": [0.77,0.34,0.66,0.12], "ver": "v1"},
        {"advice_id": 33, "embed": [0.38,0.53,0.81,0.27], "ver": "v1"},
        {"advice_id": 34, "embed": [0.45,0.91,0.25,0.62], "ver": "v1"},
        {"advice_id": 35, "embed": [0.66,0.28,0.74,0.41], "ver": "v1"},
        {"advice_id": 36, "embed": [0.19,0.63,0.42,0.85], "ver": "v1"},
        {"advice_id": 37, "embed": [0.83,0.14,0.59,0.33], "ver": "v1"},
        {"advice_id": 38, "embed": [0.52,0.71,0.36,0.48], "ver": "v1"},
        {"advice_id": 39, "embed": [0.62,0.44,0.23,0.69], "ver": "v1"},
        {"advice_id": 40, "embed": [0.14,0.75,0.81,0.35], "ver": "v1"},
        {"advice_id": 41, "embed": [0.47,0.62,0.39,0.52], "ver": "v1"},
        {"advice_id": 42, "embed": [0.55,0.18,0.77,0.41], "ver": "v1"},
        {"advice_id": 43, "embed": [0.23,0.88,0.19,0.63], "ver": "v1"},
        {"advice_id": 44, "embed": [0.36,0.54,0.82,0.21], "ver": "v1"},
        {"advice_id": 45, "embed": [0.72,0.13,0.45,0.84], "ver": "v1"},
        {"advice_id": 46, "embed": [0.34,0.81,0.63,0.27], "ver": "v1"},
        {"advice_id": 47, "embed": [0.65,0.44,0.28,0.71], "ver": "v1"},
        {"advice_id": 48, "embed": [0.29,0.59,0.83,0.42], "ver": "v1"},
        {"advice_id": 49, "embed": [0.19,0.77,0.34,0.66], "ver": "v1"},
        {"advice_id": 50, "embed": [0.82,0.26,0.48,0.33], "ver": "v1"}
    ],

    "advice_sim": [
      {"advice_id_a": 1, "advice_id_b": 4, "sim": 0.81, "ver": "v1"},
      {"advice_id_a": 1, "advice_id_b": 16, "sim": 0.67, "ver": "v1"},
      {"advice_id_a": 2, "advice_id_b": 5, "sim": 0.78, "ver": "v1"},
      {"advice_id_a": 2, "advice_id_b": 18, "sim": 0.55, "ver": "v1"},
      {"advice_id_a": 3, "advice_id_b": 9, "sim": 0.62, "ver": "v1"},
      {"advice_id_a": 6, "advice_id_b": 29, "sim": 0.59, "ver": "v1"},
      {"advice_id_a": 7, "advice_id_b": 19, "sim": 0.74, "ver": "v1"},
      {"advice_id_a": 8, "advice_id_b": 33, "sim": 0.71, "ver": "v1"},
      {"advice_id_a": 10, "advice_id_b": 24, "sim": 0.68, "ver": "v1"},
      {"advice_id_a": 12, "advice_id_b": 25, "sim": 0.77, "ver": "v1"},
      {"advice_id_a": 14, "advice_id_b": 40, "sim": 0.69, "ver": "v1"},
      {"advice_id_a": 15, "advice_id_b": 36, "sim": 0.65, "ver": "v1"},
      {"advice_id_a": 17, "advice_id_b": 34, "sim": 0.58, "ver": "v1"},
      {"advice_id_a": 20, "advice_id_b": 35, "sim": 0.63, "ver": "v1"},
      {"advice_id_a": 21, "advice_id_b": 41, "sim": 0.72, "ver": "v1"},
      {"advice_id_a": 22, "advice_id_b": 45, "sim": 0.61, "ver": "v1"},
      {"advice_id_a": 26, "advice_id_b": 46, "sim": 0.64, "ver": "v1"},
      {"advice_id_a": 27, "advice_id_b": 47, "sim": 0.73, "ver": "v1"},
      {"advice_id_a": 30, "advice_id_b": 50, "sim": 0.57, "ver": "v1"},
      {"advice_id_a": 31, "advice_id_b": 42, "sim": 0.66, "ver": "v1"}
    ]
}

user_state = {
    "user_state_daily": [
        {"couple_id": 101, "day": "2025-09-12", "anxiety_score": 60, "avoidance_score": 57},
        {"couple_id": 101, "day": "2025-09-15", "anxiety_score": 55, "avoidance_score": 52}
    ],
    "user_feedback": [
        {"couple_id": 101, "member": "A", "advice_id": 2, "label": "good", "ts": "..."},
        {"couple_id": 101, "member": "B", "advice_id": 5, "label": "bad",  "ts": "..."}
    ],
    "session_log": []
}

fr_json = {
    "fr_window": {
        "couple_id": 101,
        "from": "2025-08-10",
        "to": "2025-09-14",
        "user_advice_ids": [2,5,3,1], 
        "good_set": [2,5],
        "bad_set": [3],
        "last_top1": 5
    }
}

# =========================================================
# 1) JSON → 내부 구조 어댑터
#    - item_id = advice_id 로 동일 취급
# =========================================================
def user_key(couple_id, member="A"):
    return f"{couple_id}_{member.upper()}"

def build_user_info_from_refer(refer_json):
    """couple_id -> {'A': int, 'B': int}"""
    users = {}
    for u in refer_json.get("users", []):
        cid = u["couple_id"]
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
        cid = fb["couple_id"]
        mem = (fb.get("member") or "A").upper()  # 과거 데이터 호환
        uid = user_key(cid, mem)
        rows.append({"user_id": uid, "couple_id": cid, "member": mem,
                     "item_id": fb["advice_id"], "value": v})
    if not rows:
        return pd.DataFrame(columns=["user_id","couple_id","member","item_id","value"])
    return pd.DataFrame(rows)


def build_item_vecs_from_refer(refer_json, ver="v1"):
    vecs = {}
    for f in refer_json.get("advice_feature", []):
        if f.get("ver") == ver:
            vecs[f["advice_id"]] = np.array(f["embed"], dtype=float)
    return vecs

def build_item_meta_from_refer(refer_json):
    meta = {}
    for a in refer_json.get("advice", []):
        meta[a["advice_id"]] = {"tags": a.get("tags", []), "comment": a.get("advice_comment","")}
    return meta

def build_item_sim_from_refer(refer_json, ver="v1", item_id_index=None):
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
    uid = user_key(couple_id, member)

    # 최근/마지막값 갱신
    recent_feedback[uid].append((item_id, val01))
    last_feedback_value[uid][item_id] = val01

    # interactions 추가 (여기가 포인트)
    if item_id in iid2idx:
        v5 = float(val01) * 5.0
        new_row = pd.DataFrame({
            "user_id":   [uid],
            "couple_id": [couple_id],
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

def ir_top1(couple_id, member="A"):
    uid = user_key(couple_id, member)
    recs = recommend_for_user(uid, N=1, topk_item_neighbors=None)
    if not recs:
        return None
    aid, _ = recs[0]
    return {
        "advice_id": aid,
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
        "advice_id": aid,
        "advice_comment": advice_text(aid),
        "reason": reason,
        "neighbors": neighbors_for(aid, K=3)
    }


def safe_neighbors_for(advice_id, K=3):
    """설명용 이웃 (advice_sim에서 상위 K) — 소형 케이스 안전"""
    if advice_id not in iid2idx or item_sim.size == 0:
        return []
    iidx = iid2idx[advice_id]
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
            out.append({"advice_id": iid, "sim": round(s, 4)})
        if len(out) >= K:
            break
    return out

# 기존 neighbors_for를 safe_neighbors_for로 대체
neighbors_for = safe_neighbors_for

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

    # 애착유형 바이어스 반영
    mtype = user_info.get(couple_id, {}).get(member, 0)
    if mtype:
        scores = apply_member_bias(idxs, scores, mtype, boost=0.10)

    order = np.argsort(-scores)
    if order.size == 0:
        return None

    top_iidx = idxs[order[0]]
    aid = idx2iid[top_iidx]
    return {
        "advice_id": aid,
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
    try:
        target_user = int(input("target user (예: 101): ").strip())
    except ValueError:
        raise SystemExit("유효한 정수를 입력하세요. 예: 101")

    # (위는 동일)
    for turn in range(36):
        print(f"\n=== IR Top-1 (회차 {turn+1}) ===")
        ab = ir_top1_AB(target_user)  # A와 B 동시에 추천

        a_rec = ab.get("A")
        b_rec = ab.get("B")

        print("A추천:", a_rec)
        print("B추천:", b_rec)

        current_aid_A = a_rec["advice_id"] if isinstance(a_rec, dict) else None
        current_aid_B = b_rec["advice_id"] if isinstance(b_rec, dict) else None

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
                    if new_rec and new_rec["advice_id"] == target_aid:
                        user_blacklist[user_key(target_user, member)].add(target_aid)
                        new_rec = ir_top1_member(target_user, member)

                    if member == "A":
                        current_aid_A = new_rec["advice_id"] if new_rec else None
                        if new_rec:
                            print("A 재추천:", new_rec)
                    else:
                        current_aid_B = new_rec["advice_id"] if new_rec else None
                        if new_rec:
                            print("B 재추천:", new_rec)



    print("허용 입력: good / bad / skip / quit")


    # FR(1안) 호출
    print("\n=== FR Top-1 (36일 회고, 1안: Item-CF) ===")
    print(fr_top1(target_user, window_days=36, mode="itemcf"))
