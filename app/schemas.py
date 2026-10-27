"""잡핏 에이전트 — 데이터 스키마.

모든 에이전트 출력은 이 모델로 검증된다. 검증에 실패한 출력은
리포트에 포함되지 않고, 검증 원장(verifier)이 차단한다.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Requirement(BaseModel):
    """공고에서 추출한 요건 한 건."""

    id: str
    category: str  # 필수 / 우대 / 근무조건
    text: str


class Match(BaseModel):
    """요건과 이력서 원문 근거의 대조 결과."""

    requirement_id: str
    verdict: Literal["매칭", "부분 매칭", "없음"]
    citations: list[str] = Field(default_factory=list)  # 근거 청크 id 목록
    rationale: str = ""


class GapPlan(BaseModel):
    """매칭되지 않은 요건에 대한 준비 계획. 없는 경험을 실무 경험으로 쓰지 않는다."""

    requirement_id: str
    plan: str


class Report(BaseModel):
    """최종 리포트. unverified에는 근거가 없어 표시하지 못한 항목이 들어간다."""

    requirements: list[Requirement] = Field(default_factory=list)
    matches: list[Match] = Field(default_factory=list)
    gaps: list[GapPlan] = Field(default_factory=list)
    unverified: list[str] = Field(default_factory=list)
    blocked_claims: int = 0  # 측정용: 검증 원장이 차단한 근거 없는 주장 건수 (재시도 포함 누적)
    llm_calls: int = 0  # 측정용: 분석 1건당 LLM 호출 횟수
