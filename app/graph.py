"""LangGraph 그래프: 파서 → 대조 → 검증 → 갭 → 리포트.

검증 원장이 재생성을 요청하면 대조 단계로 되돌아가고(최대 3회),
그 뒤에도 근거가 없으면 해당 항목은 unverified로 표시된다.
갭 플래너는 검증을 통과한 판정만 받는다. 검증 전에 돌리면 차단될 주장의
rationale이 학습 계획 문장으로 새어 나가 검증 원장을 우회하게 된다.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app import agents
from app.state import AgentState


def _route_after_verify(state: AgentState) -> str:
    return "retry" if state.get("route") == "retry" else "done"


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("parser", agents.parse_requirements)
    graph.add_node("matcher", agents.match_profile)
    graph.add_node("verifier", agents.verify)
    graph.add_node("gap", agents.plan_gaps)
    graph.add_edge(START, "parser")
    graph.add_edge("parser", "matcher")
    graph.add_edge("matcher", "verifier")
    graph.add_conditional_edges(
        "verifier",
        _route_after_verify,
        {"retry": "matcher", "done": "gap"},
    )
    graph.add_edge("gap", END)
    return graph.compile()
