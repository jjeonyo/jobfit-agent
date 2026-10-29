"""LangGraph 노드 4종 + 검증 원장.

공통 규칙:
- LLM이 없으면(OPENAI_API_KEY 미설정) 분석을 지어내지 않고 즉시 에러를 낸다.
- LLM 출력이 JSON 배열이 아니면 1회 재시도하고, 그래도 실패하면 LLMOutputError를 낸다.
- 모든 노드 출력은 pydantic 스키마로 검증한다. 검증에 실패한 항목은 버리고,
  판정이 빠진 요건은 검증 원장이 재생성을 요청한 뒤 unverified로 넘긴다.
- 없는 경험은 '없음'으로 판정하고, 절대 실무 경험으로 둔갑시키지 않는다.
"""

from __future__ import annotations

import json
import logging
import os
import re

from pydantic import BaseModel, ValidationError

from app.rag import hybrid_retrieve
from app.schemas import GapPlan, Match, Report, Requirement
from app.state import AgentState
from app.vectorstore import get_vector_store

logger = logging.getLogger(__name__)

MAX_VERIFY_ATTEMPTS = 3
CLAIM_VERDICTS = ("매칭", "부분 매칭")
# 분석 1건 동안 누적하는 측정값. 상태와 Report에 같은 이름으로 들어간다.
USAGE_KEYS = ("llm_calls", "input_tokens", "output_tokens")
# 근거 문장 최소 길이. 단어 하나("Python")만 인용해 통과하는 것을 막는다.
MIN_QUOTE_CHARS = 10
# 이 표기가 든 문장은 경험의 근거가 될 수 없다 (data/profile.md의 '학습 중' 항목, VERDICT_RULES).
NON_EXPERIENCE_MARKERS = ("학습 중",)

PARSER_PROMPT = """다음 채용공고 텍스트에서 요건을 추출해 JSON 배열로만 출력하라.
각 항목: {{"id": "req-01", "category": "필수|우대|근무조건", "text": "요건 내용"}}.
요건이 아닌 문구(회사 소개, 복리후생 자랑)는 제외하라. 설명 문장을 덧붙이지 마라.
공고:
{posting_text}"""

# 판정 기준. 평가셋 정답 라벨(docs/판정-기준.md)과 같은 문장을 쓴다.
# 기준을 바꾸면 문서도 같이 고친다 (tests/test_logic.py가 일치를 확인한다).
VERDICT_RULES = [
    "매칭: 이력서 원문에 요건을 직접 충족하는 경험·기술이 적혀 있다.",
    "부분 매칭: 관련 경험은 원문에 있으나 요건의 범위·수준·기간 중 일부만 충족한다.",
    "없음: 원문에 근거가 없다.",
    "매칭과 부분 매칭 사이에서 판단이 갈리면 부분 매칭으로 한다.",
    "부분 매칭과 없음 사이에서 판단이 갈리면, 원문에 인용할 문장이 있을 때만 부분 매칭이고 없으면 없음이다.",
    "인용할 문장이 없으면 없음이다. 추론은 근거가 아니다.",
    "이력서에 '학습 중'으로 표기된 기술은 경험이 아니므로 경험 요건에 대해 없음으로 판정한다.",
    "팀 작업은 원문에 적힌 담당 범위만 근거로 쓰고, 단독 경험으로 쓰지 않는다.",
    "근무지·고용 형태처럼 이력서로 판단할 수 없는 조건은 근거가 없으므로 없음으로 판정한다.",
]

MATCHER_PROMPT = (
    """다음 요건에 대해, 이력서 원문 근거 청크만을 근거로 판정하라.
판정은 매칭 / 부분 매칭 / 없음 중 하나다. 이력서에 없는 경험은 '없음'으로 판정하고,
절대 실무 경험이 있는 것처럼 쓰지 마라.
판정 기준:
"""
    + "\n".join(f"- {rule}" for rule in VERDICT_RULES)
    + """
출력은 JSON 배열로만 하라.
각 항목: {{"requirement_id": "...", "verdict": "...", "citations": ["chunk-001"],
"quotes": [{{"chunk_id": "chunk-001", "text": "..."}}], "rationale": "..."}}.
'매칭/부분 매칭'이면 quotes에 판정 근거 문장을 근거 청크에서 글자 그대로 복사해 넣어라.
고치거나 요약하지 말고, 한 청크 안의 연속된 """
    + str(MIN_QUOTE_CHARS)
    + """자 이상 문장을 복사하라. 인용한 청크마다 quote가 하나 이상 있어야 한다.
근거가 없으면 citations와 quotes를 빈 배열로 하고 verdict를 '없음'으로 하라.
요건:
{requirements_json}
근거 청크:
{chunks_text}"""
)

MATCHER_RETRY_NOTE = """
재검증 요청: 위 요건들의 이전 판정은 아래 사유로 차단됐다. 근거 청크에 직접적인 근거가
있으면 그 청크 id를 정확히 인용하고 근거 문장을 글자 그대로 복사하라. 없으면 verdict를
'없음', citations와 quotes를 빈 배열로 하라.
차단 사유:"""

GAP_PROMPT = """매칭되지 않았거나 부분 매칭인 요건에 대해, 학습 계획과 자소서 강조 각도를
제안하라. 없는 경험을 실무 경험으로 포장하지 말고 '학습 중'으로만 표기하라.
출력은 JSON 배열로만 하라. 각 항목: {{"requirement_id": "...", "plan": "..."}}.
대상 요건:
{gaps_json}"""


class LLMOutputError(ValueError):
    """LLM 출력이 재시도 후에도 JSON 배열이 아닐 때. 리포트를 만들지 않는다."""


def _get_llm():
    """OPENAI_API_KEY가 없으면 지어낸 분석 대신 명확한 에러를 낸다."""
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY가 설정되지 않았습니다. .env에 키를 넣은 뒤 다시 실행하세요."
        )
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model="gpt-4o-mini", temperature=0)


def _parse_json_array(text: str) -> list:
    """JSON 배열을 파싱한다. ```json 코드펜스로 감싼 출력도 받는다."""
    text = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    parsed = json.loads(text)
    if not isinstance(parsed, list):
        raise ValueError("LLM 출력이 JSON 배열이 아니다")
    return parsed


def _call_json(llm, prompt: str) -> tuple[list, dict]:
    """LLM에 JSON 배열을 요청하고, 파싱 실패 시 1회 재시도한다.

    반환: (항목 목록, 사용량). 사용량은 USAGE_KEYS별 값으로, 재시도 호출까지 합친
    실제 LLM 호출 횟수와 응답의 usage_metadata에 적힌 입력·출력 토큰 수다.
    """
    usage = dict.fromkeys(USAGE_KEYS, 0)

    def ask(text: str) -> str:
        response = llm.invoke(text)
        usage["llm_calls"] += 1
        tokens = getattr(response, "usage_metadata", None)
        if tokens:
            usage["input_tokens"] += tokens.get("input_tokens", 0)
            usage["output_tokens"] += tokens.get("output_tokens", 0)
        else:
            logger.warning("LLM 응답에 토큰 사용량(usage_metadata)이 없어 0으로 집계한다")
        return response.content

    content = ask(prompt)
    try:
        return _parse_json_array(content), usage
    except ValueError:
        pass
    content = ask(prompt + "\nJSON 배열만 출력하라.")
    try:
        return _parse_json_array(content), usage
    except ValueError as exc:
        raise LLMOutputError("LLM 출력을 2회 연속 JSON 배열로 파싱하지 못했습니다") from exc


def _add_usage(state: AgentState, usage: dict) -> dict:
    """상태의 누적 사용량에 이번 호출의 사용량을 더한 갱신분."""
    return {key: state.get(key, 0) + usage[key] for key in USAGE_KEYS}


def _validate(model: type[BaseModel], items: list) -> list:
    """스키마 검증에 실패한 항목은 버린다 (리포트에 넣지 않는다)."""
    valid = []
    for item in items:
        try:
            valid.append(model.model_validate(item))
        except ValidationError as exc:
            logger.warning("%s 스키마 검증 실패로 항목 제외: %s", model.__name__, exc)
    return valid


def parse_requirements(state: AgentState) -> dict:
    llm = _get_llm()
    items, usage = _call_json(llm, PARSER_PROMPT.format(posting_text=state["posting_text"]))
    requirements: dict[str, Requirement] = {}
    for req in _validate(Requirement, items):
        requirements.setdefault(req.id, req)
    return {"requirements": list(requirements.values()), **_add_usage(state, usage)}


def match_profile(state: AgentState) -> dict:
    """요건별 근거 청크를 검색해 판정한다.

    재시도(verifier의 fix_requests가 있을 때)에는 차단된 요건만 다시 판정하고,
    이미 검증을 통과한 판정은 그대로 둔다.
    """
    requirements = state["requirements"]
    fix_ids = set(state.get("fix_requests", []))
    targets = [req for req in requirements if not fix_ids or req.id in fix_ids]
    if not targets:
        return {"matches": []}
    llm = _get_llm()
    chunks = state["profile_chunks"]
    vector_store = get_vector_store(chunks)
    # 요건마다 근거 청크를 검색해 프롬프트에 넣는다 (컨텍스트 절약).
    evidence: dict[str, str] = {}
    for req in targets:
        for chunk in hybrid_retrieve(req.text, chunks, vector_store, top_k=5):
            evidence[chunk["id"]] = chunk["text"]
    chunks_text = "\n".join(f"{cid}: {text}" for cid, text in evidence.items())
    requirements_json = json.dumps([req.model_dump() for req in targets], ensure_ascii=False)
    prompt = MATCHER_PROMPT.format(
        requirements_json=requirements_json,
        chunks_text=chunks_text or "(검색된 근거 청크 없음)",
    )
    if fix_ids:
        reasons = state.get("fix_reasons", {})
        prompt += MATCHER_RETRY_NOTE + "".join(
            f"\n- {req.id}: {reasons.get(req.id, '판정 누락')}" for req in targets
        )
    items, usage = _call_json(llm, prompt)

    target_ids = {req.id for req in targets}
    by_id = {
        m.requirement_id: m
        for m in state.get("matches", [])
        if m.requirement_id not in target_ids
    }
    for match in _validate(Match, items):
        if match.requirement_id in target_ids:
            by_id.setdefault(match.requirement_id, match)
    matches = [by_id[req.id] for req in requirements if req.id in by_id]
    return {"matches": matches, **_add_usage(state, usage)}


def plan_gaps(state: AgentState) -> dict:
    """검증을 통과한 판정 중 '부분 매칭/없음'에 대해서만 준비 계획을 만든다."""
    requirements = {req.id: req for req in state.get("requirements", [])}
    unmatched = [
        match
        for match in state["matches"]
        if match.verdict in ("부분 매칭", "없음") and match.requirement_id in requirements
    ]
    if not unmatched:
        return {"gaps": []}
    llm = _get_llm()
    gaps_json = json.dumps(
        [
            {
                "requirement_id": match.requirement_id,
                "category": requirements[match.requirement_id].category,
                "requirement": requirements[match.requirement_id].text,
                "verdict": match.verdict,
                "rationale": match.rationale,
            }
            for match in unmatched
        ],
        ensure_ascii=False,
    )
    items, usage = _call_json(llm, GAP_PROMPT.format(gaps_json=gaps_json))
    unmatched_ids = {match.requirement_id for match in unmatched}
    gaps = [gap for gap in _validate(GapPlan, items) if gap.requirement_id in unmatched_ids]
    return {"gaps": gaps, **_add_usage(state, usage)}


def _normalize(text: str) -> str:
    """공백을 한 칸으로 접는다. 청크도 같은 방식으로 접혀 있다 (app/chunking.py)."""
    return re.sub(r"\s+", " ", text).strip()


def _short(text: str, limit: int = 40) -> str:
    return text if len(text) <= limit else text[:limit] + "…"


# 청크는 줄바꿈이 공백으로 접혀 있으므로, 목록 기호(- * >)나 제목(#)을 항목의 시작으로 본다.
_ITEM_START = re.compile(r"(?:^|\s)(?:[-*>]|#{1,6})\s")


def _non_experience_marker(
    text: str, quote: str, lo: int = 0, hi: int | None = None
) -> str | None:
    """text[lo:hi]에 있는 근거 문장이 든 목록 항목에 비경험 표기('학습 중')가 있으면 그 표기를 반환한다.

    항목은 근거 문장 직전의 목록 기호부터 근거 문장 끝까지로 본다. 그래서
    "- 학습 중 (…): Airflow, 클라우드, CI/CD"에서 "Airflow, 클라우드, CI/CD"만 인용해도 잡힌다.
    항목 시작은 text 전체에서 찾으므로, text가 전체 본문이면 청크 경계를 넘는 항목도 잡힌다.
    근거 문장이 여러 번 나오면, 모든 위치가 비경험 항목일 때만 반환한다.
    """
    hi = len(text) if hi is None else hi
    found = None
    start = text.find(quote, lo, hi)
    while start != -1:
        item_starts = [m.end() for m in _ITEM_START.finditer(text, 0, start)]
        item = text[(item_starts[-1] if item_starts else 0) : start + len(quote)]
        found = next((marker for marker in NON_EXPERIENCE_MARKERS if marker in item), None)
        if found is None:
            return None
        start = text.find(quote, start + 1, hi)
    return found


class ProfileText:
    """검증 원장이 보는 이력서 본문.

    청크 id별 본문으로 근거 문장이 원문에 있는지 확인한다. 청크에 start가 있으면
    (app/chunking.py) 전체 본문을 복원해, 청크 경계를 넘는 목록 항목의 비경험 표기도 확인한다.
    """

    def __init__(self, chunks: list[dict]) -> None:
        self._chunks = {chunk["id"]: chunk for chunk in chunks}
        self._normalized = {chunk["id"]: _normalize(chunk["text"]) for chunk in chunks}
        self._full_text: str | None = None
        if chunks and all("start" in chunk for chunk in chunks):
            full = ""
            for chunk in sorted(chunks, key=lambda c: c["start"]):
                full = full[: chunk["start"]] + chunk["text"]
            self._full_text = full

    def __contains__(self, chunk_id: str) -> bool:
        return chunk_id in self._chunks

    def contains_quote(self, chunk_id: str, quote: str) -> bool:
        return quote in self._normalized[chunk_id]

    def non_experience_marker(self, chunk_id: str, quote: str) -> str | None:
        chunk = self._chunks[chunk_id]
        # 전체 본문을 복원할 수 없거나 원본 청크에서 문장을 못 찾으면(공백이 접히지 않은 입력)
        # 청크 안에서만 확인한다. 못 찾았다고 통과시키지 않기 위해서다.
        if self._full_text is None or quote not in chunk["text"]:
            return _non_experience_marker(self._normalized[chunk_id], quote)
        start = chunk["start"]
        return _non_experience_marker(self._full_text, quote, start, start + len(chunk["text"]))


def block_reason(match: Match, profile: ProfileText) -> str | None:
    """'매칭/부분 매칭' 주장이 근거를 갖추지 못했으면 차단 사유를, 갖췄으면 None을 반환한다.

    근거를 갖췄다는 것은 다음을 모두 만족한다는 뜻이다.
    - 인용(citations)이 있고, 모두 현재 이력서에 있는 청크다.
    - 인용한 청크마다 근거 문장(quotes)이 하나 이상 있다.
    - 근거 문장은 MIN_QUOTE_CHARS자 이상이고, 그 청크 원문에 글자 그대로 있다.
    - 근거 문장이 든 목록 항목에 '학습 중' 같은 비경험 표기가 없다.
    """
    if match.verdict not in CLAIM_VERDICTS:
        return None
    if not match.citations:
        return "인용 없음"
    unknown = [cid for cid in match.citations if cid not in profile]
    if unknown:
        return f"이력서에 없는 청크 인용: {', '.join(unknown)}"
    quoted: set[str] = set()
    for quote in match.quotes:
        text = _normalize(quote.text)
        if quote.chunk_id not in match.citations:
            return f"citations에 없는 청크의 근거 문장: {quote.chunk_id}"
        if len(text) < MIN_QUOTE_CHARS:
            return f"근거 문장이 {MIN_QUOTE_CHARS}자 미만: {text!r}"
        if not profile.contains_quote(quote.chunk_id, text):
            return f"근거 문장이 {quote.chunk_id} 원문에 없음: {_short(text)!r}"
        marker = profile.non_experience_marker(quote.chunk_id, text)
        if marker is not None:
            return f"'{marker}' 항목의 문장은 경험 근거가 아님: {_short(text)!r}"
        quoted.add(quote.chunk_id)
    no_quote = [cid for cid in match.citations if cid not in quoted]
    if no_quote:
        return f"근거 문장 없는 인용: {', '.join(no_quote)}"
    return None


def verify(state: AgentState) -> dict:
    """검증 원장: 근거 없는 매칭 주장은 차단하고, 재생성 또는 unverified로 처리한다.

    - '매칭/부분 매칭' 주장 중 block_reason이 사유를 내는 것은 근거 없는 주장이다
      (blocked_claims로 집계).
    - 판정이 누락된 요건도 재생성 대상이다.
    - attempts < 3이면 차단 사유와 함께 matcher로 되돌려 재생성을 요청하고,
      3회 재생성 후에도 남으면 해당 요건을 unverified로 표시하고 매칭에서 제외한다.
    """
    attempts = state.get("attempts", 0)
    matches = state["matches"]
    profile = ProfileText(state.get("profile_chunks", []))

    reasons: dict[str, str] = {}
    for match in matches:
        reason = block_reason(match, profile)
        if reason is not None:
            reasons[match.requirement_id] = reason
    blocked_count = len(reasons)
    judged_ids = {m.requirement_id for m in matches}
    for req in state.get("requirements", []):
        if req.id not in judged_ids:
            reasons[req.id] = "판정 누락"
    bad_ids = list(reasons)
    blocked_claims = state.get("blocked_claims", 0) + blocked_count

    if not bad_ids:
        return {
            "fix_requests": [],
            "fix_reasons": {},
            "route": "done",
            "blocked_claims": blocked_claims,
        }
    if attempts < MAX_VERIFY_ATTEMPTS:
        return {
            "fix_requests": bad_ids,
            "fix_reasons": reasons,
            "attempts": attempts + 1,
            "route": "retry",
            "blocked_claims": blocked_claims,
        }
    bad = set(bad_ids)
    kept = [m for m in matches if m.requirement_id not in bad]
    unverified = list(state.get("unverified", []))
    unverified += [rid for rid in bad_ids if rid not in unverified]
    return {
        "matches": kept,
        "unverified": unverified,
        "fix_requests": [],
        "fix_reasons": {},
        "route": "done",
        "blocked_claims": blocked_claims,
    }


def build_report(state: AgentState) -> Report:
    return Report(
        requirements=state.get("requirements", []),
        matches=state.get("matches", []),
        gaps=state.get("gaps", []),
        unverified=state.get("unverified", []),
        blocked_claims=state.get("blocked_claims", 0),
        **{key: state.get(key, 0) for key in USAGE_KEYS},
    )
