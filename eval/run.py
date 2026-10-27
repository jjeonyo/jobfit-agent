"""평가 지표 계산. 예측과 정답 라벨을 비교해 README용 수치를 낸다.

사용법:
  python -m eval.predict --gold eval/cases/gold.json --out eval/preds.json  # 예측 생성
  python -m eval.run --pred eval/preds.json --gold eval/cases/gold.json [--threshold 0.5]

정답 파일: {"cases": [{"id": ..., "posting_text": ...,
           "gold_requirements": [{"id", "category", "text", "verdict", "citations"}]}]}
  verdict는 매칭/부분 매칭/없음 (docs/판정-기준.md). verdict·citations는 선택이며,
  verdict가 없는 요건은 판정 지표에서 빠진다.
예측 파일: {"cases": [{"id": ..., "requirements": [...], "matches": [...], "unverified": [...],
           "blocked_claims", "llm_calls", "elapsed_sec"}]}
  eval/predict.py가 /analyze 응답(Report)과 같은 형식으로 만든다.

요건 id는 Parser가 순서대로 붙이므로 정답 id와 대응하지 않는다. 그래서 예측 요건과
정답 요건은 id가 아니라 요건 문장의 유사도로 1:1 대응시킨다 (align_requirements).
"""

from __future__ import annotations

import argparse
import json
import re

DEFAULT_THRESHOLD = 0.5
VERDICTS = ("매칭", "부분 매칭", "없음")
CLAIM_VERDICTS = ("매칭", "부분 매칭")
UNVERIFIED = "미검증"  # 검증 원장이 막아 리포트에 판정이 없는 요건


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
    judged = [m for m in matches if m.get("verdict") in CLAIM_VERDICTS]
    if not judged:
        return 1.0
    cited = [m for m in judged if m.get("citations")]
    return len(cited) / len(judged)


def _ratio(numerator: int, denominator: int) -> float | None:
    """분모가 0이면 None (측정 불가). 0.0과 구분한다."""
    return numerator / denominator if denominator else None


def verdict_pairs(case: dict, alignment: list[dict]) -> list[dict]:
    """대응한 요건마다 정답 판정과 예측 판정을 짝짓는다.

    - 정답 판정(verdict)을 적지 않은 정답 요건은 뺀다. 오타 등 알 수 없는 값은 에러다.
    - 예측 판정이 없는 요건(unverified이거나 matches에 없음)은 '미검증'으로 둔다.
      미검증은 정답이 될 수 없으므로 판정 정확도에서 오답으로 계산된다.
    """
    gold_by_id = {g["id"]: g for g in case.get("gold_requirements", [])}
    match_by_id = {m["requirement_id"]: m for m in case.get("matches", [])}
    unverified = set(case.get("unverified", []))
    pairs = []
    for pair in alignment:
        gold = gold_by_id[pair["gold_id"]]
        gold_verdict = gold.get("verdict")
        if gold_verdict is None:
            continue
        if gold_verdict not in VERDICTS:
            raise ValueError(
                f"{case.get('id')} {gold['id']}: 알 수 없는 정답 판정 {gold_verdict!r}"
            )
        match = match_by_id.get(pair["predicted_id"])
        if match is None or pair["predicted_id"] in unverified:
            predicted_verdict, predicted_citations = UNVERIFIED, []
        else:
            predicted_verdict, predicted_citations = match["verdict"], match.get("citations", [])
        pairs.append(
            {
                "gold_id": pair["gold_id"],
                "predicted_id": pair["predicted_id"],
                "gold": gold_verdict,
                "predicted": predicted_verdict,
                "gold_citations": gold.get("citations", []),
                "predicted_citations": predicted_citations,
            }
        )
    return pairs


def verdict_metrics(pairs: list[dict]) -> dict:
    """판정 지표.

    - verdict_accuracy: 정답 판정과 예측 판정이 같은 비율 (미검증은 오답).
    - fabrication_rate: 정답이 '없음'인 요건을 '매칭/부분 매칭'으로 판정한 비율. 목표 0.
    - unverified: 예측 판정이 '미검증'인 건수.
    - citation_accuracy: 정답·예측이 모두 '매칭/부분 매칭'이고 정답 인용이 있는 요건 중,
      예측 인용이 정답 인용과 하나 이상 겹치는 비율. 인용이 주장을 실제로 뒷받침하는지 본다.
    - confusion: 정답 판정(행) × 예측 판정(열) 건수.
    """
    confusion = {gold: {pred: 0 for pred in (*VERDICTS, UNVERIFIED)} for gold in VERDICTS}
    for pair in pairs:
        confusion[pair["gold"]][pair["predicted"]] += 1
    correct = sum(confusion[verdict][verdict] for verdict in VERDICTS)
    fabricated = sum(confusion["없음"][verdict] for verdict in CLAIM_VERDICTS)
    both_claims = [
        p
        for p in pairs
        if p["gold"] in CLAIM_VERDICTS and p["predicted"] in CLAIM_VERDICTS and p["gold_citations"]
    ]
    citation_hits = [
        p for p in both_claims if set(p["gold_citations"]) & set(p["predicted_citations"])
    ]
    return {
        "labeled": len(pairs),
        "verdict_accuracy": _ratio(correct, len(pairs)),
        "fabrication_rate": _ratio(fabricated, sum(confusion["없음"].values())),
        "unverified": sum(confusion[gold][UNVERIFIED] for gold in VERDICTS),
        "citation_accuracy": _ratio(len(citation_hits), len(both_claims)),
        "confusion": confusion,
    }


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def evaluate(cases: list[dict], threshold: float = DEFAULT_THRESHOLD) -> dict:
    """케이스별 지표와 전체 지표를 낸다.

    요건 추출 정확도·근거 인용률은 케이스 평균, 판정 지표는 전체 요건을 합쳐 계산한다.
    케이스별 대응 결과(alignment)와 판정 오류(verdict_errors)는 육안 검토용이다.
    """
    per_case = []
    all_pairs: list[dict] = []
    for case in cases:
        gold = case.get("gold_requirements", [])
        predicted = case.get("requirements", [])
        alignment = align_requirements(gold, predicted, threshold)
        aligned_gold = {pair["gold_id"] for pair in alignment}
        pairs = verdict_pairs(case, alignment)
        all_pairs += pairs
        per_case.append(
            {
                "id": case.get("id"),
                "extraction_accuracy": len(alignment) / len(gold) if gold else 0.0,
                "citation_rate": citation_rate(case.get("matches", [])),
                "alignment": alignment,
                "unmatched_gold_ids": [g["id"] for g in gold if g["id"] not in aligned_gold],
                "verdict_errors": [p for p in pairs if p["gold"] != p["predicted"]],
            }
        )
    accuracies = [c["extraction_accuracy"] for c in per_case]
    rates = [c["citation_rate"] for c in per_case]
    return {
        "cases": len(cases),
        "threshold": threshold,
        "mean_extraction_accuracy": sum(accuracies) / len(accuracies) if accuracies else 0.0,
        "mean_citation_rate": sum(rates) / len(rates) if rates else 1.0,
        "verdict": verdict_metrics(all_pairs),
        "pipeline": {
            "error_case_ids": [c.get("id") for c in cases if c.get("error")],
            "mean_llm_calls": _mean([c["llm_calls"] for c in cases if "llm_calls" in c]),
            "mean_elapsed_sec": _mean([c["elapsed_sec"] for c in cases if "elapsed_sec" in c]),
            "total_blocked_claims": sum(c.get("blocked_claims", 0) for c in cases),
        },
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
    merged = [
        {**case, "gold_requirements": gold_by_id.get(case["id"], {}).get("gold_requirements", [])}
        for case in pred["cases"]
    ]
    print(json.dumps(evaluate(merged, args.threshold), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
