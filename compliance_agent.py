from groq import Groq
import os

llm = Groq(api_key=os.getenv("GROQ_API_KEY"))

def analyze_compliance(
    query: str,
    law_chunks: list,
    policy_chunks: list,
    web_chunks: str | None = None
) -> dict:
    """
    Returns analysis + sources (NO logic change)
    """
    law_text = "\n".join(
        c["metadata"].get("text", "") for c in law_chunks
        )
    
    policy_text = "\n".join(
        c["metadata"].get("text", "") for c in policy_chunks
        )

    law_sources = [
        f"{c['metadata']['source']} (Page {c['metadata']['page']})"
        for c in law_chunks
    ]

    policy_sources = [
        f"{c['metadata']['source']} (Page {c['metadata']['page']})"
        for c in policy_chunks
    ]

    web_text = web_chunks if web_chunks else "No external updates used."

    prompt = f"""
You are a regulatory compliance expert.

LEGAL REQUIREMENTS:
{law_text}

COMPANY POLICY:
{policy_text}

LATEST WEB UPDATES:
{web_text}

QUESTION:
{query}

Provide:
1. Compliance Status
2. Reasoning
3. Explicit gaps
"""

    response = llm.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )

    return {
        "analysis": response.choices[0].message.content,
        "law_sources": law_sources,
        "policy_sources": policy_sources
    }
