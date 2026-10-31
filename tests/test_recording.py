"""분석 기록 전송 테스트. 네트워크 없이 실행된다."""

import json
import re
from datetime import datetime, timezone
from types import SimpleNamespace

import httpx
from fastapi.testclient import TestClient

from app import agents, recording
from app.main import app
from app.recording import POSTING_COLUMNS, PREDICTION_COLUMNS, build_record, send_record
from app.schemas import GapPlan, Match, Report, Requirement

REPORT = Report(
    requirements=[
        Requirement(id="req-01", category="필수", text="FastAPI 개발 경험"),
        Requirement(id="req-02", category="우대", text="CI/CD 구축 경험"),
        Requirement(id="req-03", category="우대", text="Kubernetes 운영 경험"),
    ],
    matches=[
        Match(
            requirement_id="req-01",
            verdict="매칭",
            citations=["chunk-001"],
            quotes=[{"chunk_id": "chunk-001", "text": "Python FastAPI 백엔드 개발"}],
            rationale="FastAPI 개발 경험",
        ),
        Match(requirement_id="req-03", verdict="없음", rationale="근거 없음"),
    ],
    gaps=[GapPlan(requirement_id="req-03", plan="Kubernetes 학습 중")],
    unverified=["req-02"],
    blocked_claims=4,
    llm_calls=6,
    input_tokens=1200,
    output_tokens=300,
)
AT = datetime(2026, 10, 31, 4, 5, 6, tzinfo=timezone.utc)  # KST 13:05:06


def test_build_record_rows_match_sheet_columns():
    record = build_record("백엔드 개발자 채용 공고", REPORT, analyzed_at=AT)

    assert re.fullmatch(r"20261031-130506-[0-9a-f]{6}", record["case_id"])
    [posting] = record["postings"]
    assert list(posting) == POSTING_COLUMNS
    assert posting["analyzed_at"] == "2026-10-31T13:05:06+09:00"
    assert posting["unverified"] == "req-02"
    assert (posting["blocked_claims"], posting["llm_calls"], posting["input_tokens"]) == (4, 6, 1200)

    rows = record["predictions"]
    assert all(list(row) == PREDICTION_COLUMNS for row in rows)
    assert all(row["case_id"] == record["case_id"] for row in rows)
    by_id = {row["requirement_id"]: row for row in rows}
    assert by_id["req-01"]["verdict"] == "매칭"
    assert by_id["req-01"]["citations"] == "chunk-001"
    assert by_id["req-01"]["quotes"] == "[chunk-001] Python FastAPI 백엔드 개발"
    # 검증 원장이 막은 요건은 판정 없이 '미검증'으로만 기록한다.
    assert by_id["req-02"]["verdict"] == "미검증"
    assert by_id["req-02"]["citations"] == "" and by_id["req-02"]["rationale"] == ""
    assert by_id["req-03"]["gap_plan"] == "Kubernetes 학습 중"


def test_case_id_is_stable_for_same_posting_and_time():
    a = build_record("공고", REPORT, analyzed_at=AT)["case_id"]
    b = build_record("공고", REPORT, analyzed_at=AT)["case_id"]
    c = build_record("다른 공고", REPORT, analyzed_at=AT)["case_id"]
    assert a == b and a != c


def _client(handler, captured):
    def record(request):
        captured.append(request)
        return handler(request)

    return httpx.Client(transport=httpx.MockTransport(record))


def test_send_record_posts_json_with_token(monkeypatch):
    monkeypatch.setenv("RECORD_WEBHOOK_URL", "http://n8n:5678/webhook/jobfit-record")
    monkeypatch.setenv("RECORD_WEBHOOK_TOKEN", "secret")
    captured = []
    record = build_record("공고", REPORT, analyzed_at=AT)

    assert send_record(record, client=_client(lambda r: httpx.Response(200), captured)) is True
    [request] = captured
    assert request.url == "http://n8n:5678/webhook/jobfit-record"
    assert request.headers["X-JobFit-Token"] == "secret"
    assert json.loads(request.content)["case_id"] == record["case_id"]


def test_send_record_failure_does_not_raise(monkeypatch):
    monkeypatch.setenv("RECORD_WEBHOOK_URL", "http://n8n:5678/webhook/jobfit-record")
    monkeypatch.delenv("RECORD_WEBHOOK_TOKEN", raising=False)
    record = build_record("공고", REPORT, analyzed_at=AT)

    assert send_record(record, client=_client(lambda r: httpx.Response(500), [])) is False

    def refuse(request):
        raise httpx.ConnectError("connection refused")

    assert send_record(record, client=_client(refuse, [])) is False


def test_send_record_disabled_without_url(monkeypatch):
    monkeypatch.delenv("RECORD_WEBHOOK_URL", raising=False)
    assert send_record({"case_id": "x"}) is False


class FakeLLM:
    def __init__(self, outputs):
        self.outputs = list(outputs)

    def invoke(self, prompt):
        return SimpleNamespace(content=json.dumps(self.outputs.pop(0), ensure_ascii=False))


def _analyze(monkeypatch, sent):
    llm = FakeLLM(
        [
            [{"id": "req-01", "category": "필수", "text": "Kubernetes 운영 경험"}],
            [{"requirement_id": "req-01", "verdict": "없음", "citations": []}],
            [{"requirement_id": "req-01", "plan": "학습 중"}],
        ]
    )
    monkeypatch.setattr(agents, "_get_llm", lambda: llm)
    monkeypatch.setattr("app.main.load_profile", lambda: [{"id": "chunk-001", "text": "Python"}])
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.setattr(recording, "send_record", lambda record: sent.append(record))
    return TestClient(app).post("/analyze", json={"posting_text": "인프라 엔지니어 채용"})


def test_analyze_sends_record_when_webhook_configured(monkeypatch):
    monkeypatch.setenv("RECORD_WEBHOOK_URL", "http://n8n:5678/webhook/jobfit-record")
    sent = []
    response = _analyze(monkeypatch, sent)
    assert response.status_code == 200
    [record] = sent
    assert record["postings"][0]["posting_text"] == "인프라 엔지니어 채용"
    assert record["predictions"][0]["verdict"] == "없음"


def test_analyze_does_not_send_without_webhook(monkeypatch):
    monkeypatch.delenv("RECORD_WEBHOOK_URL", raising=False)
    sent = []
    assert _analyze(monkeypatch, sent).status_code == 200
    assert sent == []
