"""FastAPI 진입점.

GET  /         : 웹 폼 (공고 붙여넣기 → 리포트). 프론트 프레임워크 없이 정적 HTML 1장.
GET  /chunks   : 이력서 근거 청크 목록 (리포트의 인용 id를 원문과 대조하는 용도).
POST /analyze  : 공고 텍스트를 받아 멀티에이전트 분석 리포트를 반환한다.
OPENAI_API_KEY가 없으면 지어낸 리포트 대신 503을 반환한다.
RECORD_WEBHOOK_URL이 있으면 응답 후 분석 기록을 보낸다 (app/recording.py).
"""

from __future__ import annotations

from pathlib import Path

import openai
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app import agents, recording
from app.graph import build_graph, initial_state
from app.profile import load_profile
from app.schemas import Report

MAX_POSTING_CHARS = 20_000  # 공개 배포 시 요청 1건의 비용 상한
INDEX_HTML = Path(__file__).resolve().parent / "static" / "index.html"

app = FastAPI(title="JobFit Agent")
_graph = build_graph()


class AnalyzeRequest(BaseModel):
    posting_text: str = Field(max_length=MAX_POSTING_CHARS)


class Chunk(BaseModel):
    id: str
    text: str


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(INDEX_HTML, media_type="text/html; charset=utf-8")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/chunks", response_model=list[Chunk])
def chunks() -> list[dict]:
    try:
        return load_profile()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/analyze", response_model=Report)
def analyze(request: AnalyzeRequest, background_tasks: BackgroundTasks) -> Report:
    if not request.posting_text.strip():
        raise HTTPException(status_code=400, detail="posting_text가 비어 있습니다")
    try:
        profile_chunks = load_profile()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    try:
        final_state = _graph.invoke(initial_state(request.posting_text, profile_chunks))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except agents.LLMOutputError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except openai.OpenAIError as exc:
        raise HTTPException(status_code=502, detail=f"LLM 호출 실패: {exc}") from exc
    report = agents.build_report(final_state)
    if recording.enabled():
        # 응답을 보낸 뒤 전송한다. 전송 실패는 응답에 영향을 주지 않는다.
        background_tasks.add_task(
            recording.send_record, recording.build_record(request.posting_text, report)
        )
    return report
