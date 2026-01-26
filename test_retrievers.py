from retrievers import retrieve_laws, retrieve_company_policy

query = "AI risk management requirements"

print("\n--- LAW RETRIEVER RESULTS ---\n")
law_results = retrieve_laws(query, top_k=3)

for i, r in enumerate(law_results, 1):
    print(f"Law Result {i}")
    print("Source:", r["metadata"]["source"])
    print("Page:", r["metadata"]["page"])
    print(r["metadata"]["text"][:300])
    print("-" * 50)


print("\n--- COMPANY POLICY RETRIEVER RESULTS ---\n")
policy_results = retrieve_company_policy(query, top_k=3)

for i, r in enumerate(policy_results, 1):
    print(f"Policy Result {i}")
    print("Source:", r["metadata"]["source"])
    print("Page:", r["metadata"]["page"])
    print(r["metadata"]["text"][:300])
    print("-" * 50)
