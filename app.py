# app.py
import os
import logging
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI, APIConnectionError, BadRequestError, AuthenticationError, RateLimitError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

app = FastAPI()

# ▶️ 환경변수에서 키 읽기 권장: export OPENAI_API_KEY=...
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

MODEL_NAME = "gpt-4o-mini"  # 필요시 gpt-4o 또는 gpt-4.1-mini 등으로 교체

class Diary(BaseModel):
    user_id: str
    text: str

class Feedback(BaseModel):
    user_id: str
    report_id: str
    rating: int
    comment: str

@app.get("/health")
def health():
    return {"status": "ok"}

def _first_text(oi) -> Optional[str]:
    try:
        # 안전 파싱: choices가 없거나 content가 None일 수 있으므로 대비
        return (oi.choices[0].message.content or "").strip()
    except Exception as e:
        logging.exception("Parsing error: %s", e)
        return None

@app.post("/report")
def generate_report(diary: Diary):
    try:
        messages = [
            {"role": "system", "content": "당신은 심리 보고서 작성 전문가입니다."},
            {"role": "user", "content": f"다음 일기를 분석하고 보고서를 작성하세요:\n{diary.text}"}
        ]
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=500,
            temperature=0.3,
        )
        logging.info("OpenAI raw usage: %s", getattr(resp, "usage", None))
        text = _first_text(resp)
        if not text:
            raise HTTPException(status_code=502, detail={"error": "EMPTY_RESPONSE", "raw": resp.model_dump()})
        return {"report": text, "usage": getattr(resp, "usage", None)}
    except AuthenticationError as e:
        raise HTTPException(status_code=401, detail={"error": "AUTH_ERROR", "message": str(e)})
    except RateLimitError as e:
        raise HTTPException(status_code=429, detail={"error": "RATE_LIMIT", "message": str(e)})
    except BadRequestError as e:
        raise HTTPException(status_code=400, detail={"error": "BAD_REQUEST", "message": str(e)})
    except APIConnectionError as e:
        raise HTTPException(status_code=503, detail={"error": "CONNECTION_ERROR", "message": str(e)})
    except Exception as e:
        logging.exception("UNKNOWN_ERROR")
        raise HTTPException(status_code=500, detail={"error": "UNKNOWN_ERROR", "message": str(e)})

@app.post("/report/feedback")
def regenerate_report(feedback: Feedback):
    # 실제 서비스에서는 DB에서 가져오세요. 여기선 예시 값.
    prev_report = "이전 보고서 예시"
    diary_text = "사용자 일기 예시"

    try:
        messages = [
            {"role": "system", "content": "당신은 심리 보고서를 개선하는 전문가입니다."},
            {"role": "user", "content": f"""
사용자 일기:
{diary_text}

이전 보고서:
{prev_report}

사용자 피드백:
평점: {feedback.rating}
코멘트: {feedback.comment}

위 피드백을 반영해 새로운 보고서를 작성하세요.
"""},
        ]
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=500,
            temperature=0.3,
        )
        logging.info("OpenAI raw usage: %s", getattr(resp, "usage", None))
        text = _first_text(resp)
        if not text:
            raise HTTPException(status_code=502, detail={"error": "EMPTY_RESPONSE", "raw": resp.model_dump()})
        return {"new_report": text, "usage": getattr(resp, "usage", None)}
    except AuthenticationError as e:
        raise HTTPException(status_code=401, detail={"error": "AUTH_ERROR", "message": str(e)})
    except RateLimitError as e:
        raise HTTPException(status_code=429, detail={"error": "RATE_LIMIT", "message": str(e)})
    except BadRequestError as e:
        raise HTTPException(status_code=400, detail={"error": "BAD_REQUEST", "message": str(e)})
    except APIConnectionError as e:
        raise HTTPException(status_code=503, detail={"error": "CONNECTION_ERROR", "message": str(e)})
    except Exception as e:
        logging.exception("UNKNOWN_ERROR")
        raise HTTPException(status_code=500, detail={"error": "UNKNOWN_ERROR", "message": str(e)})
