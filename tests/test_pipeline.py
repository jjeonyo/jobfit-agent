"""그래프 엔드투엔드 테스트. 가짜 LLM으로 실행되며 네트워크를 쓰지 않는다."""

import json
import logging
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app import agents
from app.graph import build_graph
from app.main import app

CHUNKS = [
    {"id": "chunk-001", "text": "Python FastAPI 백엔드 개발, LangChain RAG 하이브리드 검색"},
    {"id": "chunk-002", "text": "Docker Compose 실행 환경 구성"},
]

# chunk-001 원문에 글자 그대로 있는 근거 문장
QUOTE_001 = [{"chunk_id": "chunk-001", "text": "Python FastAPI 백엔드 개발"}]

REQUIREMENTS = [
    {"id": "req-01", "category": "필수", "text": "Python FastAPI 개발 경험"},
    {"id": "req-02", "category": "필수", "text": "Kafka 운영 경험"},
    {"id": "req-03", "category": "우대", "text": "Kubernetes 운영 경험"},
]


class FakeLLM:
    """프롬프트 종류별로 정해진 응답을 돌려주고, 받은 프롬프트를 기록한다."""

    def __init__(self, matcher_outputs, gap_output=None, parser_output=None, usage=None):
        self.matcher_outputs = list(matcher_outputs)
        self.gap_output = gap_output if gap_output is not None else []
        self.parser_output = parser_output if parser_output is not None else REQUIREMENTS
        self.usage = usage  # 응답마다 붙이는 usage_metadata (None이면 붙이지 않음)
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        if prompt.startswith("다음 채용공고"):
            payload = self.parser_output
        elif prompt.startswith("다음 요건"):
            payload = self.matcher_outputs.pop(0)
        else:
            payload = self.gap_output
        content = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
        return SimpleNamespace(content=content, usage_metadata=self.usage)


def run(llm, monkeypatch):
    monkeypatch.setattr(agents, "_get_llm", lambda: llm)
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    state = build_graph().invoke(
        {
            "posting_text": "공고",
            "profile_chunks": CHUNKS,
            "attempts": 0,
            "blocked_claims": 0,
            "llm_calls": 0,
        }
    )
    return agents.build_report(state)


def test_fabricated_claim_is_retried_then_marked_unverified(monkeypatch):
    first = [
        {"requirement_id": "req-01", "verdict": "매칭", "citations": ["chunk-001"], "quotes": QUOTE_001, "rationale": "FastAPI 개발"},
        {"requirement_id": "req-02", "verdict": "부분 매칭", "citations": [], "rationale": "Kafka 일부 경험"},
        {"requirement_id": "req-03", "verdict": "없음", "citations": [], "rationale": "근거 없음"},
    ]
    retry = [{"requirement_id": "req-02", "verdict": "매칭", "citations": [], "rationale": "Kafka 경험"}]
    llm = FakeLLM(
        [first, retry, retry, retry],
        gap_output=[{"requirement_id": "req-03", "plan": "Kubernetes 학습 중"}],
    )
    report = run(llm, monkeypatch)

    assert report.unverified == ["req-02"]
    assert [m.requirement_id for m in report.matches] == ["req-01", "req-03"]
    assert report.blocked_claims == 4  # 최초 1회 + 재생성 3회 모두 차단
    # parser 1 + matcher 4 + gap 1
    assert report.llm_calls == 6
    # 재시도는 차단된 요건만, 재검증 지시와 함께 다시 판정한다.
    retry_prompt = llm.prompts[2]
    assert "재검증 요청" in retry_prompt
    assert "req-02" in retry_prompt and "req-01" not in retry_prompt


def test_gap_planner_never_sees_blocked_rationale(monkeypatch):
    first = [
        {"requirement_id": "req-01", "verdict": "매칭", "citations": ["chunk-001"], "quotes": QUOTE_001},
        {"requirement_id": "req-02", "verdict": "부분 매칭", "citations": [], "rationale": "Kafka 실무 3년"},
        {"requirement_id": "req-03", "verdict": "없음", "citations": []},
    ]
    retry = [{"requirement_id": "req-02", "verdict": "부분 매칭", "citations": [], "rationale": "Kafka 실무 3년"}]
    llm = FakeLLM(
        [first, retry, retry, retry],
        gap_output=[
            {"requirement_id": "req-02", "plan": "Kafka 실무 3년 강조"},
            {"requirement_id": "req-03", "plan": "Kubernetes 학습 중"},
        ],
    )
    report = run(llm, monkeypatch)

    gap_prompt = llm.prompts[-1]
    assert "Kafka 실무 3년" not in gap_prompt
    # 대상이 아닌 요건에 대한 계획은 리포트에 넣지 않는다.
    assert [g.requirement_id for g in report.gaps] == ["req-03"]


def test_citation_to_unknown_chunk_is_blocked(monkeypatch):
    fabricated = [
        {"requirement_id": "req-01", "verdict": "매칭", "citations": ["chunk-001", "chunk-999"]},
        {"requirement_id": "req-02", "verdict": "없음", "citations": []},
        {"requirement_id": "req-03", "verdict": "없음", "citations": []},
    ]
    fixed = [{"requirement_id": "req-01", "verdict": "매칭", "citations": ["chunk-001"], "quotes": QUOTE_001}]
    report = run(FakeLLM([fabricated, fixed]), monkeypatch)

    assert report.unverified == []
    assert report.blocked_claims == 1
    assert report.matches[0].citations == ["chunk-001"]


def test_missing_and_invalid_items_are_regenerated(monkeypatch):
    first = [
        {"requirement_id": "req-01", "verdict": "아마도", "citations": []},  # 스키마 위반
        {"requirement_id": "req-02", "verdict": "없음", "citations": []},
        {"requirement_id": "req-99", "verdict": "매칭", "citations": ["chunk-001"], "quotes": QUOTE_001},  # 없는 요건
    ]
    second = [
        {"requirement_id": "req-01", "verdict": "매칭", "citations": ["chunk-001"], "quotes": QUOTE_001},
        {"requirement_id": "req-03", "verdict": "없음", "citations": []},
    ]
    report = run(FakeLLM([first, second]), monkeypatch)

    assert [m.requirement_id for m in report.matches] == ["req-01", "req-02", "req-03"]
    assert report.unverified == []
    assert report.blocked_claims == 0


def test_fenced_json_is_parsed(monkeypatch):
    matches = [
        {"requirement_id": r["id"], "verdict": "없음", "citations": []} for r in REQUIREMENTS
    ]
    fenced = "```json\n" + json.dumps(matches, ensure_ascii=False) + "\n```"
    report = run(FakeLLM([fenced]), monkeypatch)
    assert len(report.matches) == 3
    assert report.llm_calls == 3  # 펜스 때문에 재시도하지 않는다


def test_token_usage_is_summed_over_all_calls_including_retries(monkeypatch):
    no_match = [
        {"requirement_id": r["id"], "verdict": "없음", "citations": []} for r in REQUIREMENTS
    ]
    usage = {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}
    report = run(FakeLLM(["JSON 아님", no_match], usage=usage), monkeypatch)
    # parser 1 + matcher 2 (JSON 파싱 재시도) + gap 1
    assert report.llm_calls == 4
    assert (report.input_tokens, report.output_tokens) == (400, 80)


def test_missing_token_usage_counts_zero_and_warns(monkeypatch, caplog):
    no_match = [
        {"requirement_id": r["id"], "verdict": "없음", "citations": []} for r in REQUIREMENTS
    ]
    with caplog.at_level(logging.WARNING, logger="app.agents"):
        report = run(FakeLLM([no_match]), monkeypatch)
    assert report.llm_calls == 3
    assert (report.input_tokens, report.output_tokens) == (0, 0)
    assert "usage_metadata" in caplog.text


def test_unparseable_output_raises_after_one_retry():
    llm = FakeLLM([], parser_output="요건은 다음과 같습니다")
    with pytest.raises(agents.LLMOutputError):
        agents._call_json(llm, "다음 채용공고")
    assert len(llm.prompts) == 2


def test_analyze_endpoint_returns_verified_report(monkeypatch):
    matches = [
        {"requirement_id": "req-01", "verdict": "매칭", "citations": ["chunk-001"], "quotes": QUOTE_001},
        {"requirement_id": "req-02", "verdict": "없음", "citations": []},
        {"requirement_id": "req-03", "verdict": "없음", "citations": []},
    ]
    llm = FakeLLM([matches], gap_output=[{"requirement_id": "req-02", "plan": "학습 중"}])
    monkeypatch.setattr(agents, "_get_llm", lambda: llm)
    monkeypatch.setattr("app.main.load_profile", lambda: CHUNKS)
    monkeypatch.delenv("SUPABASE_URL", raising=False)

    response = TestClient(app).post("/analyze", json={"posting_text": "Python 개발자 채용"})
    assert response.status_code == 200
    body = response.json()
    assert body["unverified"] == [] and body["blocked_claims"] == 0
    assert body["llm_calls"] == 3
    assert body["input_tokens"] == 0 and body["output_tokens"] == 0


def test_analyze_unparseable_llm_output_returns_502(monkeypatch):
    llm = FakeLLM([], parser_output="JSON 아님")
    monkeypatch.setattr(agents, "_get_llm", lambda: llm)
    response = TestClient(app).post("/analyze", json={"posting_text": "공고"})
    assert response.status_code == 502
