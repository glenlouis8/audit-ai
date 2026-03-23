import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from audit_ai.config import BASE_DIR, COLLECTION_NAME, EMBEDDING_MODEL, QDRANT_URL, QDRANT_API_KEY, GOOGLE_API_KEY, CHUNK_SIZE, CHUNK_OVERLAP

PDF_FILE_NAME = os.path.join(BASE_DIR, "data", "nist_framework.pdf")


def ingest_docs():
    print(f"📄 Loading PDF: {PDF_FILE_NAME}...")
    loader = PyPDFLoader(PDF_FILE_NAME)
    documents = loader.load()

    print("✂️  Splitting text...")
    # Chunk size of 1000 characters with 200-character overlap balances retrieval
    # granularity against context window usage. The overlap preserves continuity
    # across chunk boundaries, which matters for multi-sentence policy clauses.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, separators=["\n\n", "\n", " ", ""]
    )
    splits = text_splitter.split_documents(documents)
    print(f"   Created {len(splits)} chunks.")

    print(f"🧠 Initializing Google Gemini Embeddings...")
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=GOOGLE_API_KEY)

    print("☁️  Connecting to Qdrant Cloud...")

    # force_recreate=True ensures the collection is rebuilt from scratch on each run.
    # This prevents dimension mismatches if the embedding model is ever changed,
    # and guarantees the index always reflects the current document exactly.
    QdrantVectorStore.from_documents(
        splits,
        embeddings,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name=COLLECTION_NAME,
        prefer_grpc=True,
        force_recreate=True,
    )
    print("✅ Ingestion Complete! New Google vectors stored.")


if __name__ == "__main__":
    ingest_docs()
