"""LangGraph 그래프: 파서 → 대조 → 갭 → 검증 → 리포트.

검증 원장이 재생성을 요청하면 대조 단계로 되돌아가고(최대 3회),
그 뒤에도 근거가 없으면 해당 항목은 unverified로 표시된다.
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
    graph.add_node("gap", agents.plan_gaps)
    graph.add_node("verifier", agents.verify)
    graph.add_edge(START, "parser")
    graph.add_edge("parser", "matcher")
    graph.add_edge("matcher", "gap")
    graph.add_edge("gap", "verifier")
    graph.add_conditional_edges(
        "verifier",
        _route_after_verify,
        {"retry": "matcher", "done": END},
    )
    return graph.compile()
