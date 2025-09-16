# fb.py
import os
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI()

# ✅ 환경변수에서 읽기: export OPENAI_API_KEY=...
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
client = OpenAI(api_key=OPENAI_API_KEY)

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

@app.get("/")
def root():
    return {"message": "Use GET /health, POST /report, POST /report/feedback"}

@app.post("/report")
def generate_report(diary: Diary):
    messages = [
        {"role": "system", "content": "당신은 심리 보고서 작성 전문가입니다."},
        {"role": "user", "content": f"다음 일기를 분석하고 보고서를 작성하세요:\n{diary.text}"}
    ]
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=500,
        timeout=30,
    )
    return {"report": resp.choices[0].message.content.strip()}

@app.post("/report/feedback")
def regenerate_report(feedback: Feedback):
    prev_report = "이전 보고서 예시"
    diary_text = "사용자 일기 예시"
    messages = [
        {"role": "system", "content": "당신은 심리 보고서를 개선하는 전문가입니다."},
        {"role": "user", "content": f"""사용자 일기:
{diary_text}

이전 보고서:
{prev_report}

사용자 피드백:
평점: {feedback.rating}
코멘트: {feedback.comment}

위 피드백을 반영해 새로운 보고서를 작성하세요.
"""}
    ]
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=500,
        timeout=30,
    )
    return {"new_report": resp.choices[0].message.content.strip()}
