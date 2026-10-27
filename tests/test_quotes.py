"""근거 문장(quotes) 검사 테스트. LLM 없이 실행된다."""

import json
from types import SimpleNamespace

import pytest

from app import agents
from app.agents import ProfileText, block_reason, verify
from app.chunking import chunk_text
from app.graph import build_graph, initial_state
from app.schemas import Match, Requirement

CHUNKS = {
    "chunk-001": "Python FastAPI 백엔드 개발, LangChain RAG 하이브리드 검색",
    "chunk-002": "학습 중 (실무 경험으로 표기하지 않음): Airflow, 클라우드, CI/CD",
    # data/profile.md 기술 스택 형식 (청킹 후 줄바꿈이 공백으로 접힌 상태)
    "chunk-003": (
        "## 기술 스택 - 백엔드: Python, FastAPI (WebSocket), Java 17, Spring Boot "
        "- 학습 중 (실무 경험으로 표기하지 않음): Airflow, 클라우드, CI/CD "
        "— 잡핏 에이전트 프로젝트로 실전 경험으로 전환 중 ## 경력"
    ),
}
PROFILE = ProfileText([{"id": cid, "text": text} for cid, text in CHUNKS.items()])


def claim(citations, quotes, verdict="매칭"):
    return Match(
        requirement_id="req-01",
        verdict=verdict,
        citations=citations,
        quotes=[{"chunk_id": cid, "text": text} for cid, text in quotes],
    )


def test_verbatim_quote_passes():
    match = claim(["chunk-001"], [("chunk-001", "Python FastAPI 백엔드 개발")])
    assert block_reason(match, PROFILE) is None


def test_whitespace_differences_are_ignored():
    match = claim(["chunk-001"], [("chunk-001", "Python  FastAPI\n백엔드 개발")])
    assert block_reason(match, PROFILE) is None


def test_no_evidence_verdict_needs_no_quote():
    assert block_reason(claim([], [], verdict="없음"), PROFILE) is None


@pytest.mark.parametrize(
    ("citations", "quotes", "expected"),
    [
        ([], [], "인용 없음"),
        (["chunk-999"], [], "이력서에 없는 청크"),
        (["chunk-001"], [], "근거 문장 없는 인용"),
        # 청크에 없는 문장 (요약·의역 포함)
        (["chunk-001"], [("chunk-001", "Python FastAPI로 백엔드를 개발함")], "원문에 없음"),
        # 단어 하나만 인용
        (["chunk-001"], [("chunk-001", "Python")], "자 미만"),
        # 인용하지 않은 청크의 문장
        (["chunk-001"], [("chunk-002", "Airflow, 클라우드, CI/CD")], "citations에 없는 청크"),
        # 두 청크를 인용했는데 한 청크에만 근거 문장이 있음
        (["chunk-001", "chunk-002"], [("chunk-001", "Python FastAPI 백엔드 개발")], "근거 문장 없는 인용"),
        # P9 재현: '학습 중' 문장을 근거로 CI/CD 경험을 주장
        (["chunk-002"], [("chunk-002", "학습 중 (실무 경험으로 표기하지 않음): Airflow, 클라우드, CI/CD")], "학습 중"),
    ],
)
def test_unsupported_claims_are_blocked(citations, quotes, expected):
    for verdict in ("매칭", "부분 매칭"):
        reason = block_reason(claim(citations, quotes, verdict=verdict), PROFILE)
        assert reason is not None and expected in reason


def test_learning_item_is_blocked_even_when_quote_skips_the_marker():
    # 표기('학습 중') 뒤 부분만 인용해도 같은 목록 항목이면 차단한다.
    reason = block_reason(
        claim(["chunk-003"], [("chunk-003", "Airflow, 클라우드, CI/CD")], verdict="부분 매칭"), PROFILE
    )
    assert reason is not None and "학습 중" in reason


def test_other_items_in_same_chunk_still_pass():
    match = claim(["chunk-003"], [("chunk-003", "백엔드: Python, FastAPI (WebSocket)")])
    assert block_reason(match, PROFILE) is None


def test_learning_item_split_across_chunks_is_blocked():
    # 실제 이력서에서 '학습 중' 항목이 청크 경계에 걸려, 뒤 청크에는 표기 없이
    # "Airflow, 클라우드, CI/CD"만 남는 경우. 전체 본문 기준으로 항목을 찾아야 잡힌다.
    text = (
        "## 기술 스택\n- 백엔드: Python, FastAPI (WebSocket), Java 17\n"
        "- 학습 중 (실무 경험으로 표기하지 않음): Airflow, 클라우드, CI/CD\n## 경력\n- 지엔텔"
    )
    chunks = chunk_text(text, size=90, overlap=20)
    quote = "Airflow, 클라우드, CI/CD"
    tail = next(c for c in chunks if quote in c["text"] and "학습 중" not in c["text"])

    match = claim([tail["id"]], [(tail["id"], quote)], verdict="부분 매칭")
    reason = block_reason(match, ProfileText(chunks))
    assert reason is not None and "학습 중" in reason
    # 청크 위치(start)가 없으면 청크 안에서만 보므로 놓친다 (복원이 필요한 이유).
    without_start = [{"id": c["id"], "text": c["text"]} for c in chunks]
    assert block_reason(match, ProfileText(without_start)) is None


def test_verify_passes_reasons_for_retry():
    state = {
        "requirements": [Requirement(id="req-01", category="필수", text="CI/CD 경험")],
        "matches": [claim(["chunk-001"], [("chunk-001", "Python")])],
        "profile_chunks": [{"id": cid, "text": text} for cid, text in CHUNKS.items()],
        "attempts": 0,
    }
    result = verify(state)
    assert result["route"] == "retry"
    assert result["blocked_claims"] == 1
    assert "자 미만" in result["fix_reasons"]["req-01"]


class FakeLLM:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return SimpleNamespace(content=json.dumps(self.outputs.pop(0), ensure_ascii=False))


def test_paraphrased_quote_is_retried_with_reason_then_fixed(monkeypatch):
    requirements = [{"id": "req-01", "category": "필수", "text": "FastAPI 개발 경험"}]
    paraphrased = [
        {
            "requirement_id": "req-01",
            "verdict": "매칭",
            "citations": ["chunk-001"],
            "quotes": [{"chunk_id": "chunk-001", "text": "FastAPI로 백엔드를 개발한 경험"}],
        }
    ]
    verbatim = [
        {
            "requirement_id": "req-01",
            "verdict": "매칭",
            "citations": ["chunk-001"],
            "quotes": [{"chunk_id": "chunk-001", "text": "Python FastAPI 백엔드 개발"}],
        }
    ]
    llm = FakeLLM([requirements, paraphrased, verbatim])
    monkeypatch.setattr(agents, "_get_llm", lambda: llm)
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    chunks = [{"id": cid, "text": text} for cid, text in CHUNKS.items()]
    report = agents.build_report(build_graph().invoke(initial_state("공고", chunks)))

    retry_prompt = llm.prompts[2]
    assert "req-01: 근거 문장이 chunk-001 원문에 없음" in retry_prompt
    assert report.unverified == []
    assert report.blocked_claims == 1
    assert report.matches[0].quotes[0].text == "Python FastAPI 백엔드 개발"
