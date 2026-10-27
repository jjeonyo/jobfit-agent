"""평가셋 예측 생성. 정답 파일의 공고마다 파이프라인을 실행해 eval/run.py 입력을 만든다.

사용법:
  python -m eval.predict --gold eval/cases/gold.json --out eval/preds.json

실제 LLM을 호출하므로 OPENAI_API_KEY가 필요하고 비용이 든다.
케이스 하나가 LLM 오류로 실패하면 error를 기록하고 다음 케이스로 넘어간다.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import openai

from app import agents
from app.graph import build_graph, initial_state
from app.profile import load_profile


def predict_case(graph, case: dict, profile_chunks: list[dict]) -> dict:
    """공고 1건을 분석해 Report 필드 + 응답 시간(elapsed_sec)을 반환한다."""
    started = time.perf_counter()
    try:
        state = graph.invoke(initial_state(case["posting_text"], profile_chunks))
    except (agents.LLMOutputError, openai.OpenAIError) as exc:
        return {"id": case["id"], "error": str(exc), "requirements": [], "matches": []}
    report = agents.build_report(state)
    return {
        "id": case["id"],
        **report.model_dump(),
        "elapsed_sec": round(time.perf_counter() - started, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY가 설정되지 않았습니다. 예측은 실제 LLM 호출이 필요합니다.")
    gold = json.loads(open(args.gold, encoding="utf-8").read())
    graph = build_graph()
    profile_chunks = load_profile()
    cases = []
    for case in gold["cases"]:
        result = predict_case(graph, case, profile_chunks)
        status = f"error: {result['error']}" if "error" in result else f"{result['elapsed_sec']}s"
        print(f"{case['id']}: {status}", file=sys.stderr)
        cases.append(result)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"cases": cases}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
