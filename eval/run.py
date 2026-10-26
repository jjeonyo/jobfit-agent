"""평가 지표 계산. 예측과 정답 라벨을 비교해 README용 수치를 낸다.

사용법:
  python -m eval.run --pred eval/preds.json --gold eval/cases/gold.json [--threshold 0.5]

정답 파일: {"cases": [{"id": ..., "gold_requirements": [{"id", "category", "text"}]}]}
예측 파일: {"cases": [{"id": ..., "requirements": [{"id", "category", "text"}],
           "matches": [{"requirement_id", "verdict", "citations"}]}]}
예측 케이스의 requirements·matches는 /analyze 응답(Report)을 그대로 쓴다.

요건 id는 Parser가 순서대로 붙이므로 정답 id와 대응하지 않는다. 그래서 예측 요건과
정답 요건은 id가 아니라 요건 문장의 유사도로 1:1 대응시킨다 (align_requirements).
"""

from __future__ import annotations

import argparse
import json
import re

DEFAULT_THRESHOLD = 0.5


def _bigrams(text: str) -> set[str]:
    normalized = re.sub(r"[^0-9a-z가-힣]", "", text.lower())
    if len(normalized) < 2:
        return {normalized} if normalized else set()
    return {normalized[i : i + 2] for i in range(len(normalized) - 1)}


def text_similarity(a: str, b: str) -> float:
    """문자 bigram Dice 계수 (0~1). 띄어쓰기·조사·문장부호 차이에 덜 민감하다.

    한계: 동의어("커뮤니케이션"/"의사소통")는 낮게, 핵심어 하나만 다른 문장
    ("AWS 운영 경험"/"GCP 운영 경험")은 높게 나온다. 케이스별 대응 결과를 검토한다.
    """
    x, y = _bigrams(a), _bigrams(b)
    if not x or not y:
        return 0.0
    return 2 * len(x & y) / (len(x) + len(y))


def align_requirements(
    gold: list[dict], predicted: list[dict], threshold: float = DEFAULT_THRESHOLD
) -> list[dict]:
    """정답 요건과 예측 요건을 문장 유사도로 1:1 대응시킨다.

    유사도가 높은 쌍부터 대응시키고, 이미 대응된 요건은 다시 쓰지 않는다.
    threshold 미만인 쌍은 대응시키지 않는다. 반환은 정답 요건 순서다:
    [{"gold_id": ..., "predicted_id": ..., "score": ...}]
    """
    pairs = sorted(
        (
            (text_similarity(g["text"], p["text"]), gi, pi)
            for gi, g in enumerate(gold)
            for pi, p in enumerate(predicted)
        ),
        key=lambda item: (-item[0], item[1], item[2]),
    )
    used_gold: set[int] = set()
    used_predicted: set[int] = set()
    aligned: list[tuple[int, dict]] = []
    for score, gi, pi in pairs:
        if score < threshold:
            break
        if gi in used_gold or pi in used_predicted:
            continue
        used_gold.add(gi)
        used_predicted.add(pi)
        pair = {"gold_id": gold[gi]["id"], "predicted_id": predicted[pi]["id"], "score": round(score, 3)}
        aligned.append((gi, pair))
    return [pair for _, pair in sorted(aligned, key=lambda item: item[0])]


def extraction_accuracy(
    gold: list[dict], predicted: list[dict], threshold: float = DEFAULT_THRESHOLD
) -> float:
    """요건 추출 정확도 = 정답 요건 중 내용이 대응하는 예측 요건이 있는 비율."""
    if not gold:
        return 0.0
    return len(align_requirements(gold, predicted, threshold)) / len(gold)


def citation_rate(matches: list[dict]) -> float:
    """근거 인용률 = '매칭/부분 매칭' 판정 중 인용이 붙은 비율. 목표 1.0."""
    judged = [m for m in matches if m.get("verdict") in ("매칭", "부분 매칭")]
    if not judged:
        return 1.0
    cited = [m for m in judged if m.get("citations")]
    return len(cited) / len(judged)


def evaluate(cases: list[dict], threshold: float = DEFAULT_THRESHOLD) -> dict:
    """케이스별 지표와 평균을 낸다. 케이스별 대응 결과(alignment)는 육안 검토용이다."""
    per_case = []
    for case in cases:
        gold = case["gold_requirements"]
        predicted = case["requirements"]
        alignment = align_requirements(gold, predicted, threshold)
        aligned_gold = {pair["gold_id"] for pair in alignment}
        per_case.append(
            {
                "id": case.get("id"),
                "extraction_accuracy": len(alignment) / len(gold) if gold else 0.0,
                "citation_rate": citation_rate(case["matches"]),
                "alignment": alignment,
                "unmatched_gold_ids": [g["id"] for g in gold if g["id"] not in aligned_gold],
            }
        )
    accuracies = [c["extraction_accuracy"] for c in per_case]
    rates = [c["citation_rate"] for c in per_case]
    return {
        "cases": len(cases),
        "threshold": threshold,
        "mean_extraction_accuracy": sum(accuracies) / len(accuracies) if accuracies else 0.0,
        "mean_citation_rate": sum(rates) / len(rates) if rates else 1.0,
        "per_case": per_case,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = parser.parse_args()
    pred = json.loads(open(args.pred, encoding="utf-8").read())
    gold = json.loads(open(args.gold, encoding="utf-8").read())
    gold_by_id = {c["id"]: c for c in gold["cases"]}
    merged = []
    for case in pred["cases"]:
        gold_case = gold_by_id.get(case["id"], {})
        merged.append(
            {
                "id": case["id"],
                "gold_requirements": gold_case.get("gold_requirements", []),
                "requirements": case.get("requirements", []),
                "matches": case.get("matches", []),
            }
        )
    print(json.dumps(evaluate(merged, args.threshold), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
