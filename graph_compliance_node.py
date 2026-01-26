from groq import Groq
import os
from graph_state import ComplianceState

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def compliance_node(state: ComplianceState) -> ComplianceState:
    laws = state["law_chunks"]
    policies = state["policy_chunks"]
    query = state["query"]

    law_text = "\n".join([l["metadata"]["text"] for l in laws])
    policy_text = "\n".join([p["metadata"]["text"] for p in policies])

    prompt = f"""
You are a regulatory compliance analysis agent.

Compare legal requirements with company policy.

LEGAL REQUIREMENTS:
{law_text}

COMPANY POLICY:
{policy_text}

QUESTION:
{query}

OUTPUT:
Compliance Status + Reasoning
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )

    return {
        **state,
        "compliance_result": response.choices[0].message.content
    }
