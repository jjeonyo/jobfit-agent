"""평가 지표 계산. 예측과 정답 라벨을 비교해 README용 수치를 낸다.

사용법:
  python -m eval.run --pred eval/preds.json --gold eval/cases/gold.json
두 파일 모두 {"cases": [{"id": ..., "gold_requirement_ids": [...],
"predicted_requirement_ids": [...], "matches": [{"requirement_id": ...,
"citations": [...]}]}]} 형식이다.
"""

from __future__ import annotations

import argparse
import json


def extraction_accuracy(gold_ids: list[str], predicted_ids: list[str]) -> float:
    """요건 추출 정확도 = 정답 요건 중 예측에 포함된 비율."""
    if not gold_ids:
        return 0.0
    gold = set(gold_ids)
    predicted = set(predicted_ids)
    return len(gold & predicted) / len(gold)


def citation_rate(matches: list[dict]) -> float:
    """근거 인용률 = '매칭/부분 매칭' 판정 중 인용이 붙은 비율. 목표 1.0."""
    judged = [m for m in matches if m.get("verdict") in ("매칭", "부분 매칭")]
    if not judged:
        return 1.0
    cited = [m for m in judged if m.get("citations")]
    return len(cited) / len(judged)


def evaluate(cases: list[dict]) -> dict:
    accuracies = [
        extraction_accuracy(c["gold_requirement_ids"], c["predicted_requirement_ids"])
        for c in cases
    ]
    rates = [citation_rate(c["matches"]) for c in cases]
    return {
        "cases": len(cases),
        "mean_extraction_accuracy": sum(accuracies) / len(accuracies) if accuracies else 0.0,
        "mean_citation_rate": sum(rates) / len(rates) if rates else 1.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True)
    parser.add_argument("--gold", required=True)
    args = parser.parse_args()
    pred = json.loads(open(args.pred, encoding="utf-8").read())
    gold = json.loads(open(args.gold, encoding="utf-8").read())
    gold_by_id = {c["id"]: c for c in gold["cases"]}
    merged = []
    for case in pred["cases"]:
        gold_case = gold_by_id.get(case["id"], {})
        merged.append(
            {
                "gold_requirement_ids": gold_case.get("gold_requirement_ids", []),
                "predicted_requirement_ids": case.get("predicted_requirement_ids", []),
                "matches": case.get("matches", []),
            }
        )
    print(json.dumps(evaluate(merged), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
