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
    llm_calls: int
