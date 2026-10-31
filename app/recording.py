"""분석 기록 전송. 평가셋 수집용 (docs/n8n-기록.md).

RECORD_WEBHOOK_URL이 설정돼 있으면, 분석 1건을 Google Sheets 행 형식으로 만들어
그 주소로 보낸다. 받는 쪽(n8n 워크플로 workflows/jobfit-record.json)은 행을 시트에
그대로 추가한다. 행 형식은 여기서 정하고 테스트한다 (n8n에는 변환 로직을 두지 않는다).

- 검증 원장을 통과한 리포트만 보낸다. 미검증 요건은 판정 대신 '미검증'으로 적는다.
- 이력서 본문은 보내지 않는다. 청크 id와 근거 문장만 보낸다.
- 전송 실패는 분석 응답에 영향을 주지 않는다 (경고 로그만 남긴다).
"""

from __future__ import annotations

import hashlib
import logging
import os
from datetime import datetime, timedelta, timezone

import httpx

from app.schemas import Report

logger = logging.getLogger(__name__)

KST = timezone(timedelta(hours=9), "KST")
UNVERIFIED = "미검증"
TOKEN_HEADER = "X-JobFit-Token"

# 시트 열 순서. docs/n8n-기록.md의 시트 머리글과 같아야 한다.
POSTING_COLUMNS = [
    "case_id",
    "analyzed_at",
    "posting_text",
    "requirements",
    "unverified",
    "blocked_claims",
    "llm_calls",
    "input_tokens",
    "output_tokens",
]
PREDICTION_COLUMNS = [
    "case_id",
    "requirement_id",
    "category",
    "requirement",
    "verdict",
    "citations",
    "quotes",
    "rationale",
    "gap_plan",
]


def enabled() -> bool:
    return bool(os.environ.get("RECORD_WEBHOOK_URL"))


def case_id_for(posting_text: str, analyzed_at: datetime) -> str:
    """분석 시각(KST) + 공고 본문 해시 앞 6자리. 같은 공고를 다시 분석해도 시각으로 구분된다."""
    digest = hashlib.sha256(posting_text.encode("utf-8")).hexdigest()[:6]
    return f"{analyzed_at.astimezone(KST):%Y%m%d-%H%M%S}-{digest}"


def build_record(posting_text: str, report: Report, analyzed_at: datetime | None = None) -> dict:
    """분석 1건을 시트 행으로 만든다: postings 탭 1행, predictions 탭 요건당 1행."""
    analyzed_at = (analyzed_at or datetime.now(KST)).astimezone(KST)
    case_id = case_id_for(posting_text, analyzed_at)
    matches = {m.requirement_id: m for m in report.matches}
    gaps = {g.requirement_id: g.plan for g in report.gaps}
    unverified = set(report.unverified)

    predictions = []
    for req in report.requirements:
        match = None if req.id in unverified else matches.get(req.id)
        predictions.append(
            {
                "case_id": case_id,
                "requirement_id": req.id,
                "category": req.category,
                "requirement": req.text,
                "verdict": match.verdict if match else UNVERIFIED,
                "citations": ",".join(match.citations) if match else "",
                "quotes": "\n".join(f"[{q.chunk_id}] {q.text}" for q in match.quotes) if match else "",
                "rationale": match.rationale if match else "",
                "gap_plan": gaps.get(req.id, ""),
            }
        )
    posting = {
        "case_id": case_id,
        "analyzed_at": analyzed_at.isoformat(timespec="seconds"),
        "posting_text": posting_text,
        "requirements": len(report.requirements),
        "unverified": ",".join(report.unverified),
        "blocked_claims": report.blocked_claims,
        "llm_calls": report.llm_calls,
        "input_tokens": report.input_tokens,
        "output_tokens": report.output_tokens,
    }
    return {"case_id": case_id, "postings": [posting], "predictions": predictions}


def send_record(record: dict, client: httpx.Client | None = None) -> bool:
    """기록을 RECORD_WEBHOOK_URL로 보낸다. 성공하면 True, 실패하면 경고를 남기고 False."""
    url = os.environ.get("RECORD_WEBHOOK_URL")
    if not url:
        return False
    headers = {}
    token = os.environ.get("RECORD_WEBHOOK_TOKEN")
    if token:
        headers[TOKEN_HEADER] = token
    try:
        if client is None:
            with httpx.Client(timeout=10) as owned:
                response = owned.post(url, json=record, headers=headers)
        else:
            response = client.post(url, json=record, headers=headers)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        logger.warning("분석 기록 전송 실패 (%s): %s", record.get("case_id"), exc)
        return False
    return True
