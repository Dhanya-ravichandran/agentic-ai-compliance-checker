import os
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "legal-lens-index"

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

model = SentenceTransformer("all-MiniLM-L6-v2")


def embed_query(query: str):
    return model.encode(query).tolist()


def retrieve_laws(query, top_k=5):
    query_vector = embed_query(query)

    response = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
        namespace="law"   # ✅ ADD THIS
    )
    return response["matches"]


def retrieve_company_policy(query, top_k=5):
    query_vector = embed_query(query)

    response = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
        namespace="policy"   # ✅ ADD THIS
    )
    return response["matches"]
