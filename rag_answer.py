import os
from groq import Groq
from retrieval import retrieve

# Load Groq API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables")

def generate_answer(query):
    # Retrieve relevant chunks
    matches = retrieve(query, top_k=5)

    # Build grounded context
    context = ""
    for m in matches:
        context += (
            f"Source: {m['metadata']['source']} | Page: {m['metadata']['page']}\n"
            f"{m['metadata']['text']}\n\n"
        )

    prompt = f"""
You are a regulatory compliance assistant.

Answer the question using ONLY the context below.
If the context is insufficient, say so clearly.
Cite sources using (Source, Page).

Context:
{context}

Question:
{query}

Answer:
"""

    client = Groq(api_key=GROQ_API_KEY)

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    query = "Does the company comply with AI risk management requirements?"
    answer = generate_answer(query)
    print("\nGenerated Answer:\n")
    print(answer)

