from langgraph.graph import StateGraph, END
from graph_state import ComplianceState
from graph_nodes import (
    planner_node,
    law_node,
    policy_node,
    web_node,
    compliance_node,
    critic_node
)

def build_graph():
    graph = StateGraph(ComplianceState)

    graph.add_node("planner", planner_node)
    graph.add_node("law", law_node)
    graph.add_node("policy", policy_node)
    graph.add_node("web", web_node)
    graph.add_node("compliance", compliance_node)
    graph.add_node("critic", critic_node)

    graph.set_entry_point("planner")

    graph.add_edge("planner", "law")
    graph.add_edge("law", "policy")
    graph.add_edge("policy", "web")
    graph.add_edge("web", "compliance")
    graph.add_edge("compliance", "critic")

    def route_after_critic(state: ComplianceState):
        if state["critic_decision"] == "RETRY":
            return "law"
        return END

    graph.add_conditional_edges("critic", route_after_critic)

    return graph.compile()
