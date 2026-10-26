"""FastAPI 진입점.

POST /analyze: 공고 텍스트를 받아 멀티에이전트 분석 리포트를 반환한다.
OPENAI_API_KEY가 없으면 지어낸 리포트 대신 503을 반환한다.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app import agents
from app.graph import build_graph
from app.profile import load_profile
from app.schemas import Report

app = FastAPI(title="JobFit Agent")
_graph = build_graph()


class AnalyzeRequest(BaseModel):
    posting_text: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/analyze", response_model=Report)
def analyze(request: AnalyzeRequest) -> Report:
    if not request.posting_text.strip():
        raise HTTPException(status_code=400, detail="posting_text가 비어 있습니다")
    try:
        profile_chunks = load_profile()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    try:
        final_state = _graph.invoke(
            {
                "posting_text": request.posting_text,
                "profile_chunks": profile_chunks,
                "attempts": 0,
                "llm_calls": 0,
            }
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return agents.build_report(final_state)
