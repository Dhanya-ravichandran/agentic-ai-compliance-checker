from typing import TypedDict, List, Dict, Any

class ComplianceState(TypedDict):
    query: str
    plan: Dict[str, bool]

    law_chunks: List[Dict[str, Any]]
    policy_chunks: List[Dict[str, Any]]
    web_chunks: Any

    analysis: str

    critic_decision: str
    critic_feedback: str

    agent_trace: List[str]   # 👈 NEW
