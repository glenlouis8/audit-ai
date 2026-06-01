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
    CHUNK_SIZE_800_53,
    CHUNK_OVERLAP_800_53,
    MIN_CHUNK_LENGTH,
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

    default_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    # 800-53 interleaves control tables with prose — smaller chunks keep table rows
    # from sharing a chunk with adjacent control text, improving retrieval precision.
    splitter_800_53 = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE_800_53,
        chunk_overlap=CHUNK_OVERLAP_800_53,
        separators=["\n\n", "\n", " ", ""],
    )

    splits = []
    for doc in documents:
        fname = doc.metadata.get("filename", "")
        splitter = splitter_800_53 if fname == "NIST.SP.800-53r5.pdf" else default_splitter
        splits.extend(splitter.split_documents([doc]))

    before = len(splits)
    splits = [s for s in splits if len(s.page_content.strip()) >= MIN_CHUNK_LENGTH]
    print(f"   Created {len(splits)} chunks ({before - len(splits)} short chunks dropped).")

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
