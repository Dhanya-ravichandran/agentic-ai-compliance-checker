import os
from tavily import TavilyClient
from groq import Groq

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
llm = Groq(api_key=os.getenv("GROQ_API_KEY"))

def web_search_agent(query: str) -> str:
    search_results = tavily.search(
        query=query,
        search_depth="basic",
        max_results=5
    )

    context = "\n".join(
        [r["content"] for r in search_results["results"]]
    )

    prompt = f"""
You are a legal research assistant.

Summarize the latest regulatory information relevant to:
"{query}"

WEB CONTEXT:
{context}

Provide a concise legal summary.
"""

    response = llm.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return response.choices[0].message.content


# THIS is what LangGraph imports
def web_search(query: str) -> str:
    return web_search_agent(query)
