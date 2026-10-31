"""Google Sheets에서 내려받은 CSV로 평가셋 정답 파일(gold.json)을 만든다.

사용법:
  python -m eval.import_labels --postings postings.csv --gold gold.csv --out eval/cases/gold.json

- postings.csv: n8n이 채우는 postings 탭 (case_id, posting_text …)
- gold.csv: 사람이 채우는 gold 탭 (case_id, gold_id, category, text, verdict, citations)
- gold 행이 하나도 없는 공고(아직 라벨을 달지 않은 공고)는 빼고 만든다.
- docs/판정-기준.md를 어긴 행은 모아서 한 번에 에러로 알린다. 파일은 쓰지 않는다.
  예: '매칭/부분 매칭'인데 인용이 없음, '없음'인데 인용이 있음, 알 수 없는 판정.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys

from eval.run import CLAIM_VERDICTS, VERDICTS

POSTING_HEADERS = ("case_id", "posting_text")
GOLD_HEADERS = ("case_id", "gold_id", "category", "text", "verdict", "citations")
CHUNK_ID = re.compile(r"chunk-\d{3}")


def read_csv(path: str) -> list[dict]:
    # 엑셀에서 다시 저장한 CSV는 BOM이 붙으므로 utf-8-sig로 읽는다.
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _check_headers(rows: list[dict], required: tuple[str, ...], name: str) -> list[str]:
    if not rows:
        return []
    missing = [h for h in required if h not in rows[0]]
    return [f"{name}: 머리글 없음: {', '.join(missing)}"] if missing else []


def build_gold(postings: list[dict], gold_rows: list[dict]) -> dict:
    """CSV 행으로 {"cases": [...]}를 만든다. 규칙 위반이 있으면 ValueError (전체 목록)."""
    errors = _check_headers(postings, POSTING_HEADERS, "postings")
    errors += _check_headers(gold_rows, GOLD_HEADERS, "gold")
    if errors:
        raise ValueError("\n".join(errors))

    posting_text = {}
    for row in postings:
        case_id = (row.get("case_id") or "").strip()
        if case_id:
            posting_text[case_id] = row.get("posting_text") or ""

    requirements: dict[str, list[dict]] = {}
    seen: set[tuple[str, str]] = set()
    for line, row in enumerate(gold_rows, start=2):  # 1행은 머리글
        cells = {key: (row.get(key) or "").strip() for key in GOLD_HEADERS}
        if not any(cells.values()):
            continue  # 빈 행
        where = f"gold {line}행"
        case_id, gold_id = cells["case_id"], cells["gold_id"]
        if case_id not in posting_text:
            errors.append(f"{where}: postings에 없는 case_id {case_id!r}")
            continue
        if not gold_id or not cells["text"]:
            errors.append(f"{where}: gold_id와 text는 비울 수 없음")
            continue
        if (case_id, gold_id) in seen:
            errors.append(f"{where}: 중복된 gold_id {gold_id!r} ({case_id})")
            continue
        seen.add((case_id, gold_id))

        citations = [c for c in re.split(r"[,\s]+", cells["citations"]) if c]
        bad = [c for c in citations if not CHUNK_ID.fullmatch(c)]
        if bad:
            errors.append(f"{where}: 청크 id 형식이 아님: {', '.join(bad)}")
        verdict = cells["verdict"]
        if verdict and verdict not in VERDICTS:
            errors.append(f"{where}: 알 수 없는 판정 {verdict!r} (매칭/부분 매칭/없음)")
        elif verdict in CLAIM_VERDICTS and not citations:
            errors.append(f"{where}: '{verdict}' 판정은 인용이 필요함")
        elif verdict == "없음" and citations:
            errors.append(f"{where}: '없음'은 인용을 비워야 함")

        requirement = {"id": gold_id, "category": cells["category"], "text": cells["text"]}
        if verdict:
            requirement["verdict"] = verdict
            requirement["citations"] = citations
        requirements.setdefault(case_id, []).append(requirement)

    if errors:
        raise ValueError("\n".join(errors))
    cases = [
        {"id": case_id, "posting_text": text, "gold_requirements": requirements[case_id]}
        for case_id, text in posting_text.items()
        if case_id in requirements
    ]
    return {"cases": cases}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--postings", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    postings = read_csv(args.postings)
    try:
        gold = build_gold(postings, read_csv(args.gold))
    except ValueError as exc:
        sys.exit(f"정답 라벨 오류:\n{exc}")
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(gold, f, ensure_ascii=False, indent=2)
    labeled = len(gold["cases"])
    total = sum(1 for row in postings if (row.get("case_id") or "").strip())
    print(f"{labeled}건을 썼습니다. 라벨 없는 공고 {total - labeled}건은 뺐습니다.", file=sys.stderr)


if __name__ == "__main__":
    main()
