from agents.planner import planner_decide
from retrievers import retrieve_laws, retrieve_company_policy
from agents.web_search_agent import web_search
from compliance_agent import analyze_compliance
from graph_state import ComplianceState
from agents.critic_agent import critic_review

# -----------------------------
# Planner Agent
# -----------------------------
def planner_node(state: ComplianceState):
    plan = planner_decide(state["query"])
    state["plan"] = plan
    state["agent_trace"] = ["🧠 Planner Agent"]
    return state

# -----------------------------
# Law Retriever Agent
# -----------------------------
def law_node(state: ComplianceState):
    if state["plan"]["use_law_retriever"]:
        state["law_chunks"] = retrieve_laws(state["query"], top_k=3)
        state["agent_trace"].append("📚 Law Retriever Agent")
    else:
        state["law_chunks"] = []
    return state

# -----------------------------
# Policy Retriever Agent
# -----------------------------
def policy_node(state: ComplianceState):
    if state["plan"]["use_policy_retriever"]:
        state["policy_chunks"] = retrieve_company_policy(state["query"], top_k=3)
        state["agent_trace"].append("📄 Policy Retriever Agent")
    else:
        state["policy_chunks"] = []
    return state

# -----------------------------
# Web Search Agent
# -----------------------------
def web_node(state: ComplianceState):
    if state["plan"]["use_web_search"]:
        state["web_chunks"] = web_search(state["query"])
        state["agent_trace"].append("🌐 Web Search Agent")
    else:
        state["web_chunks"] = None
    return state

# -----------------------------
# Compliance Analysis Agent
# -----------------------------
def compliance_node(state: ComplianceState):
    output = analyze_compliance(
        query=state["query"],
        law_chunks=state["law_chunks"],
        policy_chunks=state["policy_chunks"],
        web_chunks=state["web_chunks"]
    )

    state["analysis"] = output["analysis"]
    state["law_sources"] = output["law_sources"]
    state["policy_sources"] = output["policy_sources"]

    state["agent_trace"].append("⚖️ Compliance Analysis Agent")
    return state

# -----------------------------
# Critic Agent
# -----------------------------
def critic_node(state: ComplianceState):
    review = critic_review(
        query=state["query"],
        analysis=state["analysis"]
    )

    state["critic_decision"] = review["decision"]
    state["critic_feedback"] = review["feedback"]

    state["agent_trace"].append("🧪 Critic Agent")
    return state
