# planner_agent.py
def planner_decide(query: str):
    query_lower = query.lower()

    needs_web = any(
        keyword in query_lower
        for keyword in ["latest", "recent", "updated", "2024", "2025"]
    )

    plan = {
        "use_law_retriever": True,
        "use_policy_retriever": True,
        "use_web_search": needs_web
    }

    return plan

if __name__ == "__main__":
    q1 = "Does the company comply with AI risk management?"
    q2 = "What are the latest EU AI Act risk management updates?"

    print(planner_decide(q1))
    print(planner_decide(q2))
