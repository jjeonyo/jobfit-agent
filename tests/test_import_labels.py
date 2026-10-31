"""시트 CSV → gold.json 변환 테스트."""

import json

import pytest

from eval.import_labels import build_gold, main, read_csv

POSTINGS_CSV = """case_id,analyzed_at,posting_text,requirements
20261031-130506-a1b2c3,2026-10-31T13:05:06+09:00,"백엔드 개발자 채용
자격요건: Python",2
20261031-140000-d4e5f6,2026-10-31T14:00:00+09:00,라벨 안 단 공고,1
"""

GOLD_CSV = """case_id,gold_id,category,text,verdict,citations
20261031-130506-a1b2c3,g-01,필수,Python 백엔드 개발 경험,매칭,"chunk-001, chunk-003"
20261031-130506-a1b2c3,g-02,우대,Kubernetes 운영 경험,없음,
20261031-130506-a1b2c3,g-03,우대,Kafka 운영 경험,,
,,,,,
"""


def write(tmp_path, name, text, bom=False):
    path = tmp_path / name
    path.write_text(("﻿" if bom else "") + text, encoding="utf-8")
    return str(path)


def test_build_gold_from_sheet_csv(tmp_path):
    postings = read_csv(write(tmp_path, "postings.csv", POSTINGS_CSV))
    gold_rows = read_csv(write(tmp_path, "gold.csv", GOLD_CSV, bom=True))
    gold = build_gold(postings, gold_rows)

    # 라벨이 없는 공고는 빠진다.
    [case] = gold["cases"]
    assert case["id"] == "20261031-130506-a1b2c3"
    assert case["posting_text"] == "백엔드 개발자 채용\n자격요건: Python"
    g1, g2, g3 = case["gold_requirements"]
    assert g1 == {
        "id": "g-01",
        "category": "필수",
        "text": "Python 백엔드 개발 경험",
        "verdict": "매칭",
        "citations": ["chunk-001", "chunk-003"],
    }
    assert g2["verdict"] == "없음" and g2["citations"] == []
    # 판정을 아직 안 정한 요건은 verdict 없이 넣는다 (eval.run이 판정 지표에서 뺀다).
    assert "verdict" not in g3


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        ("20261031-130506-a1b2c3,g-01,필수,요건,아마도,", "알 수 없는 판정"),
        ("20261031-130506-a1b2c3,g-01,필수,요건,매칭,", "인용이 필요함"),
        ("20261031-130506-a1b2c3,g-01,필수,요건,없음,chunk-001", "인용을 비워야 함"),
        ("20261031-130506-a1b2c3,g-01,필수,요건,매칭,chunk-1", "청크 id 형식"),
        ("없는-case,g-01,필수,요건,매칭,chunk-001", "postings에 없는 case_id"),
        ("20261031-130506-a1b2c3,g-01,필수,,매칭,chunk-001", "비울 수 없음"),
    ],
)
def test_rule_violations_are_reported(row, expected):
    postings = [{"case_id": "20261031-130506-a1b2c3", "posting_text": "공고"}]
    header = "case_id,gold_id,category,text,verdict,citations".split(",")
    gold_rows = [dict(zip(header, row.split(",")))]
    with pytest.raises(ValueError, match=expected):
        build_gold(postings, gold_rows)


def test_all_violations_reported_together():
    postings = [{"case_id": "c1", "posting_text": "공고"}]
    gold_rows = [
        {"case_id": "c1", "gold_id": "g-01", "category": "", "text": "a", "verdict": "매칭", "citations": ""},
        {"case_id": "c1", "gold_id": "g-01", "category": "", "text": "b", "verdict": "없음", "citations": ""},
    ]
    with pytest.raises(ValueError) as excinfo:
        build_gold(postings, gold_rows)
    message = str(excinfo.value)
    assert "gold 2행" in message and "gold 3행: 중복된 gold_id" in message


def test_missing_header_is_reported():
    with pytest.raises(ValueError, match="머리글 없음: verdict"):
        build_gold(
            [{"case_id": "c1", "posting_text": "공고"}],
            [{"case_id": "c1", "gold_id": "g", "category": "", "text": "t", "citations": ""}],
        )


def test_cli_writes_gold_json(tmp_path, monkeypatch):
    out = tmp_path / "gold.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "import_labels",
            "--postings",
            write(tmp_path, "postings.csv", POSTINGS_CSV),
            "--gold",
            write(tmp_path, "gold.csv", GOLD_CSV),
            "--out",
            str(out),
        ],
    )
    main()
    assert len(json.loads(out.read_text(encoding="utf-8"))["cases"]) == 1
