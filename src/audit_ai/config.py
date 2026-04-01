import os
from dotenv import load_dotenv

load_dotenv()

# All external credentials are loaded from the environment to keep secrets out of source control.
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Model names are centralised here so that swapping models only requires a single change.
# Both the LLM and eval judge use the same model family to keep latency and cost consistent.
EMBEDDING_MODEL = "models/gemini-embedding-001"
LLM_MODEL = "gemini-2.5-flash-lite"
EVAL_JUDGE_MODEL = "gemini-2.5-flash-lite"
COLLECTION_NAME = "compliance_audit"

# --- RAG tuning knobs ---
# Number of documents fetched from Qdrant per query. Higher values improve recall but increase
# grading cost (one LLM call per document).
RETRIEVAL_K = 4

# Maximum number of query-rewrite retries before falling back to partial-context generation.
MAX_RETRIES = 3

# PDF ingestion chunking parameters. Overlap prevents important sentences being split across chunks.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Number of recent chat history messages (user + assistant interleaved) passed to the router.
# 6 items = 3 full turns. Keeps context window small while preserving short-term memory.
HISTORY_WINDOW = 6

# Resolve the project root relative to this file so paths work regardless of the working directory.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Fail fast at startup rather than surfacing cryptic errors deep inside a request handler.
if not QDRANT_URL or not QDRANT_API_KEY or not GOOGLE_API_KEY:
    raise ValueError("Missing critical API Keys in .env file")
