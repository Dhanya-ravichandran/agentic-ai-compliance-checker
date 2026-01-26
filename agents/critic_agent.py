from groq import Groq
import os

llm = Groq(api_key=os.getenv("GROQ_API_KEY"))

def critic_review(query: str, analysis: str) -> dict:
    """
    Reviews compliance analysis quality.
    """

    prompt = f"""
You are a senior regulatory auditor.

Review the following compliance analysis.

QUESTION:
{query}

ANALYSIS:
{analysis}

Evaluate:
1. Are legal claims supported?
2. Are sources implicitly grounded?
3. Is reasoning sufficient?

Respond in JSON with:
- decision: PASS or RETRY
- feedback: short explanation
"""

    response = llm.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )

    text = response.choices[0].message.content

    if "RETRY" in text.upper():
        return {"decision": "RETRY", "feedback": text}

    return {"decision": "PASS", "feedback": text}
