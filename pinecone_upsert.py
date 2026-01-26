import os
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "legal-lens-index"
BATCH_SIZE = 100

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

model = SentenceTransformer("all-MiniLM-L6-v2")


def upsert_to_pinecone(chunks: list, namespace: str):
    vectors = []

    for i, chunk in enumerate(chunks):
        embedding = model.encode(chunk["text"]).tolist()
        vectors.append({
            "id": f"chunk-{i}",
            "values": embedding,
            "metadata": {
                "text": chunk["text"],          # ✅ ADD THIS
                "source": chunk["metadata"]["source"],
                "page": chunk["metadata"]["page"],
                "type": chunk["metadata"].get("type")
                }
            })
        
    for i in range(0, len(vectors), BATCH_SIZE):
        index.upsert(
            vectors=vectors[i:i + BATCH_SIZE],
            namespace=namespace
        )
