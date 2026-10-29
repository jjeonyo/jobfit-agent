"""스키마·평가·검증 원장 테스트. LLM 없이 실행된다."""

import pytest

from app.agents import verify
from app.schemas import Match, Report
from eval.run import align_requirements, citation_rate, evaluate, extraction_accuracy


def test_report_rejects_unknown_verdict():
    with pytest.raises(ValueError):
        Match(requirement_id="req-01", verdict="아마도", citations=[])


def _reqs(*texts, prefix="req"):
    return [{"id": f"{prefix}-{i:02d}", "text": t} for i, t in enumerate(texts, start=1)]


def test_extraction_accuracy():
    gold = _reqs("Python 백엔드 개발 경험", "Docker 배포 경험")
    predicted = _reqs("Python 백엔드 개발 경험", "Kubernetes 운영 경험")
    assert extraction_accuracy(gold, predicted) == 0.5
    assert extraction_accuracy([], predicted) == 0.0


def test_extraction_accuracy_aligns_by_content_not_id():
    # 같은 id라도 내용이 다르면 대응하지 않는다 (기존 id 일치 방식의 오류).
    gold = _reqs("LLM을 활용한 서비스 개발 경험", "Docker 등 컨테이너 기반 배포 경험")
    shuffled = _reqs("Kubernetes 운영 경험", "SQL 튜닝 경험")
    assert extraction_accuracy(gold, shuffled) == 0.0
    # id와 순서가 달라도 표기만 다른 같은 요건은 대응한다.
    paraphrased = [
        {"id": "req-07", "text": "Docker 컨테이너 배포 경험"},
        {"id": "req-03", "text": "LLM 활용 서비스 개발 경험"},
    ]
    pairs = align_requirements(gold, paraphrased)
    assert [(p["gold_id"], p["predicted_id"]) for p in pairs] == [
        ("req-01", "req-03"),
        ("req-02", "req-07"),
    ]


def test_alignment_is_one_to_one():
    # 예측 요건 하나가 정답 요건 둘을 동시에 채우지 않는다.
    gold = _reqs("Python 개발 경험", "Python 개발 경험 3년 이상")
    predicted = _reqs("Python 개발 경험")
    assert extraction_accuracy(gold, predicted) == 0.5


def test_alignment_threshold():
    gold = _reqs("FastAPI 또는 Django 사용 경험")
    predicted = _reqs("Python 웹 프레임워크(FastAPI, Django) 경험")
    assert extraction_accuracy(gold, predicted, threshold=0.5) == 1.0
    assert extraction_accuracy(gold, predicted, threshold=0.6) == 0.0


def test_evaluate_reports_unmatched_gold():
    result = evaluate(
        [
            {
                "id": "case-01",
                "gold_requirements": _reqs("Python 백엔드 개발 경험", "원활한 커뮤니케이션 능력"),
                "requirements": _reqs("Python 백엔드 개발 경험", "원활한 의사소통 능력"),
                "matches": [{"requirement_id": "req-01", "verdict": "매칭", "citations": ["chunk-001"]}],
            }
        ]
    )
    assert result["mean_extraction_accuracy"] == 0.5
    assert result["mean_citation_rate"] == 1.0
    assert result["per_case"][0]["unmatched_gold_ids"] == ["req-02"]


def test_evaluate_reports_mean_token_usage_excluding_error_cases():
    ok = {"llm_calls": 4, "input_tokens": 1000, "output_tokens": 200, "elapsed_sec": 2.0}
    result = evaluate(
        [
            {"id": "case-01", **ok},
            {"id": "case-02", **ok, "input_tokens": 3000, "output_tokens": 400},
            {"id": "case-03", "error": "LLM 호출 실패", "requirements": [], "matches": []},
        ]
    )
    pipeline = result["pipeline"]
    assert pipeline["error_case_ids"] == ["case-03"]
    assert pipeline["mean_input_tokens"] == 2000
    assert pipeline["mean_output_tokens"] == 300


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


def _verdict_case(gold_verdicts, predicted, unverified=()):
    """gold_verdicts: [(verdict, citations)], predicted: [(verdict, citations) | None]."""
    texts = [f"요건 문장 {chr(0xAC00 + i * 100)}{i}" for i in range(len(gold_verdicts))]
    gold = [
        {"id": f"g-{i}", "text": t, "verdict": v, "citations": c}
        for i, (t, (v, c)) in enumerate(zip(texts, gold_verdicts))
    ]
    requirements = [{"id": f"req-{i}", "text": t} for i, t in enumerate(texts)]
    matches = [
        {"requirement_id": f"req-{i}", "verdict": p[0], "citations": p[1]}
        for i, p in enumerate(predicted)
        if p is not None
    ]
    return {
        "id": "case-01",
        "gold_requirements": gold,
        "requirements": requirements,
        "matches": matches,
        "unverified": list(unverified),
    }


def test_verdict_metrics_confusion_and_fabrication():
    case = _verdict_case(
        gold_verdicts=[
            ("매칭", ["chunk-001"]),
            ("부분 매칭", ["chunk-002"]),
            ("없음", []),
            ("없음", []),
            ("매칭", ["chunk-003"]),
        ],
        predicted=[
            ("매칭", ["chunk-001"]),  # 정답, 인용 일치
            ("매칭", ["chunk-004"]),  # 부분 매칭 → 매칭, 인용 불일치
            ("부분 매칭", ["chunk-001"]),  # 없음 → 부분 매칭 = 지어내기
            ("없음", []),  # 정답
            None,  # 미검증
        ],
        unverified=["req-4"],
    )
    verdict = evaluate([case])["verdict"]
    assert verdict["labeled"] == 5
    assert verdict["verdict_accuracy"] == 2 / 5  # 미검증은 오답
    assert verdict["fabrication_rate"] == 1 / 2
    assert verdict["unverified"] == 1
    assert verdict["citation_accuracy"] == 1 / 2
    assert verdict["confusion"]["없음"]["부분 매칭"] == 1
    assert verdict["confusion"]["매칭"]["미검증"] == 1


def test_verdict_metrics_unmeasurable_is_none_not_zero():
    case = _verdict_case([("매칭", [])], [("매칭", ["chunk-001"])])
    verdict = evaluate([case])["verdict"]
    assert verdict["verdict_accuracy"] == 1.0
    assert verdict["fabrication_rate"] is None  # 정답 '없음'이 없다
    assert verdict["citation_accuracy"] is None  # 정답 인용이 없다


def test_unlabeled_gold_is_skipped_and_typo_raises():
    case = _verdict_case([("매칭", [])], [("매칭", [])])
    del case["gold_requirements"][0]["verdict"]
    assert evaluate([case])["verdict"]["labeled"] == 0
    case["gold_requirements"][0]["verdict"] = "매칭 "
    with pytest.raises(ValueError):
        evaluate([case])


def test_verdict_rules_match_labeling_doc_and_prompt():
    from pathlib import Path

    from app.agents import MATCHER_PROMPT, VERDICT_RULES

    doc = (Path(__file__).resolve().parent.parent / "docs" / "판정-기준.md").read_text(
        encoding="utf-8"
    )
    for rule in VERDICT_RULES:
        assert rule in doc
        assert rule in MATCHER_PROMPT
