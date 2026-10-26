"""LangGraph 노드 4종 + 검증 원장.

공통 규칙:
- LLM이 없으면(OPENAI_API_KEY 미설정) 분석을 지어내지 않고 즉시 에러를 낸다.
- 모든 노드 출력은 pydantic 스키마로 검증한다. 검증 실패 시 1회 재시도하고,
  그래도 실패하면 해당 항목을 unverified로 넘긴다.
- 없는 경험은 '없음'으로 판정하고, 절대 실무 경험으로 둔갑시키지 않는다.
"""

from __future__ import annotations

import json
import os

from app.rag import hybrid_retrieve
from app.schemas import GapPlan, Match, Report, Requirement
from app.state import AgentState

MAX_VERIFY_ATTEMPTS = 3

PARSER_PROMPT = """다음 채용공고 텍스트에서 요건을 추출해 JSON 배열로만 출력하라.
각 항목: {{"id": "req-01", "category": "필수|우대|근무조건", "text": "요건 내용"}}.
요건이 아닌 문구(회사 소개, 복리후생 자랑)는 제외하라. 설명 문장을 덧붙이지 마라.
공고:
{posting_text}"""

MATCHER_PROMPT = """다음 요건에 대해, 이력서 원문 근거 청크만을 근거로 판정하라.
판정은 매칭 / 부분 매칭 / 없음 중 하나다. 이력서에 없는 경험은 '없음'으로 판정하고,
절대 실무 경험이 있는 것처럼 쓰지 마라. 출력은 JSON 배열로만 하라.
각 항목: {{"requirement_id": "...", "verdict": "...", "citations": ["chunk-001"], "rationale": "..."}}.
근거가 없으면 citations를 빈 배열로 하고 verdict를 '없음'으로 하라.
요건:
{requirements_json}
근거 청크:
{chunks_text}"""

GAP_PROMPT = """매칭되지 않았거나 부분 매칭인 요건에 대해, 학습 계획과 자소서 강조 각도를
제안하라. 없는 경험을 실무 경험으로 포장하지 말고 '학습 중'으로만 표기하라.
출력은 JSON 배열로만 하라. 각 항목: {{"requirement_id": "...", "plan": "..."}}.
대상 요건:
{gaps_json}"""


def _get_llm():
    """OPENAI_API_KEY가 없으면 지어낸 분석 대신 명확한 에러를 낸다."""
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY가 설정되지 않았습니다. .env에 키를 넣은 뒤 다시 실행하세요."
        )
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model="gpt-4o-mini", temperature=0)


def _call_json(llm, prompt: str) -> list:
    """LLM에 JSON 배열을 요청하고, 파싱 실패 시 1회 재시도한다."""
    response = llm.invoke(prompt)
    text = response.content.strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        response = llm.invoke(prompt + "\nJSON 배열만 출력하라.")
        parsed = json.loads(response.content.strip())
    if not isinstance(parsed, list):
        raise ValueError("LLM 출력이 JSON 배열이 아니다")
    return parsed


def parse_requirements(state: AgentState) -> dict:
    llm = _get_llm()
    items = _call_json(llm, PARSER_PROMPT.format(posting_text=state["posting_text"]))
    requirements = [Requirement(**item) for item in items]
    return {
        "requirements": requirements,
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


def match_profile(state: AgentState) -> dict:
    llm = _get_llm()
    chunks = state["profile_chunks"]
    # 요건마다 근거 청크를 검색해 프롬프트에 넣는다 (컨텍스트 절약).
    chunks_text_lines = []
    for req in state["requirements"]:
        for chunk in hybrid_retrieve(req.text, chunks, top_k=5):
            chunks_text_lines.append(f'{chunk["id"]}: {chunk["text"]}')
    seen: dict[str, str] = {}
    for line in chunks_text_lines:
        chunk_id, _, text = line.partition(": ")
        seen[chunk_id] = text
    chunks_text = "\n".join(f"{cid}: {text}" for cid, text in seen.items())
    requirements_json = json.dumps(
        [req.model_dump() for req in state["requirements"]], ensure_ascii=False
    )
    items = _call_json(
        llm,
        MATCHER_PROMPT.format(requirements_json=requirements_json, chunks_text=chunks_text),
    )
    matches = [Match(**item) for item in items]
    return {"matches": matches, "llm_calls": state.get("llm_calls", 0) + 1}


def plan_gaps(state: AgentState) -> dict:
    unmatched = [
        match
        for match in state["matches"]
        if match.verdict in ("부분 매칭", "없음")
    ]
    if not unmatched:
        return {"gaps": []}
    llm = _get_llm()
    gaps_json = json.dumps([m.model_dump() for m in unmatched], ensure_ascii=False)
    items = _call_json(llm, GAP_PROMPT.format(gaps_json=gaps_json))
    gaps = [GapPlan(**item) for item in items]
    return {"gaps": gaps, "llm_calls": state.get("llm_calls", 0) + 1}


def verify(state: AgentState) -> dict:
    """검증 원장: 인용 없는 매칭 주장은 차단하고, 재생성 또는 unverified로 처리한다.

    - citations가 비어 있는 '매칭/부분 매칭' 주장은 근거 없는 주장이다.
    - attempts < 3이면 matcher로 되돌려 재생성을 요청하고,
      3회 초과 시 해당 요건을 unverified로 표시하고 매칭에서 제외한다.
    """
    attempts = state.get("attempts", 0)
    bad_ids = [
        match.requirement_id
        for match in state["matches"]
        if match.verdict in ("매칭", "부분 매칭") and not match.citations
    ]
    if not bad_ids:
        return {"fix_requests": [], "route": "done"}
    if attempts < MAX_VERIFY_ATTEMPTS:
        return {
            "fix_requests": bad_ids,
            "attempts": attempts + 1,
            "route": "retry",
        }
    kept = [m for m in state["matches"] if m.requirement_id not in bad_ids]
    unverified = list(state.get("unverified", [])) + bad_ids
    return {
        "matches": kept,
        "unverified": unverified,
        "fix_requests": [],
        "route": "done",
    }


def build_report(state: AgentState) -> Report:
    return Report(
        requirements=state.get("requirements", []),
        matches=state.get("matches", []),
        gaps=state.get("gaps", []),
        unverified=state.get("unverified", []),
        llm_calls=state.get("llm_calls", 0),
    )
