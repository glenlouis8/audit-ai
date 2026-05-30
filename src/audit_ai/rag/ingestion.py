import os
import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import PayloadSchemaType
from audit_ai.config import (
    BASE_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    QDRANT_URL,
    QDRANT_API_KEY,
    GOOGLE_API_KEY,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

DATA_DIR = os.path.join(BASE_DIR, "data")


def ingest_docs():
    pdf_paths = glob.glob(os.path.join(DATA_DIR, "*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDFs found in {DATA_DIR}")

    documents = []
    for path in pdf_paths:
        fname = os.path.basename(path)
        print(f"📄 Loading PDF: {fname}...")
        loader = PyPDFLoader(path)
        docs = loader.load()
        for doc in docs:
            doc.metadata["filename"] = fname
        documents.extend(docs)

    print(f"   Loaded {len(documents)} pages from {len(pdf_paths)} PDFs.")
    print("✂️  Splitting text...")
    # Chunk size of 1000 characters with 200-character overlap balances retrieval
    # granularity against context window usage. The overlap preserves continuity
    # across chunk boundaries, which matters for multi-sentence policy clauses.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    splits = text_splitter.split_documents(documents)
    print(f"   Created {len(splits)} chunks.")

    print(f"🧠 Initializing Google Gemini Embeddings...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL, google_api_key=GOOGLE_API_KEY
    )

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
        prefer_grpc=False,
        force_recreate=True,
    )

    print("🗂️  Creating payload index on metadata.filename...")
    qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    qdrant_client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="metadata.filename",
        field_schema=PayloadSchemaType.KEYWORD,
    )
    print("✅ Ingestion Complete! Vectors stored and index created.")


if __name__ == "__main__":
    ingest_docs()
