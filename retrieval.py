import os
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in environment variables")

INDEX_NAME = "legal-lens-index"

def retrieve(query, top_k=5):
    # Initialize Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)

    # Load embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Convert query to embedding
    query_embedding = model.encode(query).tolist()

    # Query Pinecone
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )

    return results["matches"]


if __name__ == "__main__":
    query = "How does the company process personal data and ensure GDPR compliance?"
    matches = retrieve(query)

    print("\nTop retrieved chunks:\n")
    for i, match in enumerate(matches, 1):
        print(f"Result {i}")
        print(f"Score: {match['score']}")
        print(f"Source: {match['metadata']['source']}")
        print(f"Page: {match['metadata']['page']}")
        print(match['metadata']['text'][:400])
        print("-" * 60)
