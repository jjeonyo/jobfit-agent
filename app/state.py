"""LangGraph 상태 정의. 모든 노드는 이 상태의 일부만 갱신한다."""

from __future__ import annotations

from typing import TypedDict

from app.schemas import GapPlan, Match, Requirement


class AgentState(TypedDict, total=False):
    posting_text: str
    profile_chunks: list[dict]
    requirements: list[Requirement]
    matches: list[Match]
    gaps: list[GapPlan]
    unverified: list[str]
    fix_requests: list[str]  # verifier가 재생성을 요청한 요건 id
    attempts: int  # verifier 재생성 시도 횟수 (최대 3)
    blocked_claims: int  # verifier가 차단한 근거 없는 주장 누적 건수
    route: str  # verifier 판정: "retry"면 matcher로 회귀, "done"이면 gap으로 진행
    llm_calls: int
