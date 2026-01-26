from sentence_transformers import SentenceTransformer
from chunk import chunk_documents
from document_loader import load_pdfs

def generate_embeddings():
    # Load documents
    docs = load_pdfs()
    chunks = chunk_documents(docs)

    # Load embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(texts, show_progress_bar=True)

    print(f"Total embeddings created: {len(embeddings)}")
    print(f"Embedding vector size: {len(embeddings[0])}")

    return embeddings, chunks


if __name__ == "__main__":
    generate_embeddings()