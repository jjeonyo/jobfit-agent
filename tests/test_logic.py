"""스키마·평가·검증 원장 테스트. LLM 없이 실행된다."""

import pytest

from app.agents import verify
from app.schemas import Match, Report
from eval.run import citation_rate, extraction_accuracy


def test_report_rejects_unknown_verdict():
    with pytest.raises(ValueError):
        Match(requirement_id="req-01", verdict="아마도", citations=[])


def test_extraction_accuracy():
    assert extraction_accuracy(["a", "b"], ["a", "c"]) == 0.5
    assert extraction_accuracy([], ["a"]) == 0.0


def test_citation_rate_counts_only_judged_matches():
    matches = [
        {"requirement_id": "r1", "verdict": "매칭", "citations": ["chunk-001"]},
        {"requirement_id": "r2", "verdict": "매칭", "citations": []},
        {"requirement_id": "r3", "verdict": "없음", "citations": []},
    ]
    assert citation_rate(matches) == 0.5


def test_verifier_blocks_citationless_claim_and_retries():
    state = {
        "matches": [Match(requirement_id="req-01", verdict="매칭", citations=[])],
        "attempts": 0,
        "unverified": [],
    }
    result = verify(state)
    assert result["route"] == "retry"
    assert result["fix_requests"] == ["req-01"]


def test_verifier_marks_unverified_after_max_attempts():
    state = {
        "matches": [Match(requirement_id="req-01", verdict="매칭", citations=[])],
        "attempts": 3,
        "unverified": [],
    }
    result = verify(state)
    assert result["route"] == "done"
    assert result["matches"] == []
    assert result["unverified"] == ["req-01"]


def test_report_defaults():
    report = Report()
    assert report.matches == [] and report.llm_calls == 0
