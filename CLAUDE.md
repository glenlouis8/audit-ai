# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AuditAI is an agentic RAG (Retrieval-Augmented Generation) backend that autonomously audits organizations against the NIST Cybersecurity Framework 2.0. It uses a **Corrective RAG (CRAG)** pattern implemented in LangGraph with streaming responses via FastAPI.

## Folder Structure

```
audit-ai-backend/
├── src/audit_ai/
│   ├── api/
│   │   └── main.py         # FastAPI app — /chat SSE streaming endpoint & /health
│   ├── rag/
│   │   ├── engine.py       # LangGraph CRAG graph — all RAG logic lives here
│   │   └── ingestion.py    # One-off script: loads PDF → chunks → embeds → Qdrant
│   └── config.py           # Centralized API keys & model names (loaded from .env)
├── evals/
│   ├── collector.py        # Runs all test.csv questions through RAG → rag_results.json
│   ├── evaluator.py        # RAGAS metrics runner (first 10 records) → ragas_results.csv + ragas_report.md
│   └── test.csv            # 10 ground-truth NIST Q&A pairs
├── data/
│   └── nist_framework.pdf  # Source document ingested into Qdrant
├── pyproject.toml          # Dependencies & project metadata (uv / hatchling)
├── Dockerfile              # Multi-stage production build (python:3.12-slim)
├── docker-compose.yml      # Local dev orchestration (mounts . → /app)
└── render.yaml             # Render.com deployment config
```

## Commands

```bash
# Install dependencies (uses uv, not pip)
uv sync

# Run the API server (dev)
uv run uvicorn audit_ai.api.main:app --reload

# Ingest/re-index the NIST PDF into Qdrant (destructive — overwrites existing vectors)
uv run python src/audit_ai/rag/ingestion.py

# Evaluation pipeline
uv run python evals/collector.py    # Runs 50 test questions through RAG → rag_results.json
uv run python evals/evaluator.py    # RAGAS evaluation (first 10 records) → ragas_results.csv + ragas_report.md

# Docker (production-like)
docker-compose up --build
```

There is no pytest setup. Evaluation is done via the RAGAS pipeline in `evals/`.

## Architecture

### LangGraph CRAG Flow (`src/audit_ai/rag/engine.py`)

The core is a stateful LangGraph graph with retry logic. Routing happens **outside** the compiled graph in the API layer before calling it:

```
User Query → route_query() [API layer, not a graph node]
                 ↓ "chat"     → run_chat_logic() (1-3 sentence response, no sources)
                 ↓ "search"   → LangGraph: RETRIEVE → GRADE_DOCUMENTS
                                                           ↓
                                           Grade=YES → GENERATE → Response + sources
                                           Grade=NO + retries<3 → TRANSFORM_QUERY → RETRIEVE (loop)
                                           retries>=3 → GENERATE (partial context)
```

**Key state fields in `GraphState`:** `question`, `search_query`, `generation`, `documents`, `grade`, `retry_count`, `history`

### API Streaming (`src/audit_ai/api/main.py`)

`POST /chat` streams NDJSON (newline-delimited JSON) token-by-token:
```json
{"type": "token", "content": "The"}
{"type": "sources", "content": [{"file": "NIST CSF 2.0", "page": 42, "text": "..."}]}
```

Sources are suppressed if the generation contains refusal phrases (e.g., "I don't know").

### External Services

- **Google Gemini API** — generation (`gemini-2.0-flash-lite`), embeddings (`gemini-embedding-001`), RAGAS eval judge (`gemini-2.5-flash-lite`)
- **Qdrant Cloud** — managed vector DB; collection name is `compliance_audit`; `ingestion.py` uses `force_recreate=True`

### Configuration (`src/audit_ai/config.py`)

All API keys and model names are centralized here, loaded from `.env`. Required env vars: `GOOGLE_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY`.

### Key Architectural Notes

- **`PYTHONPATH=src`** must be set when running outside Docker (Docker sets this automatically). `uv run` from the project root handles this via pyproject.toml.
- Ingestion chunks PDFs at 1000 chars with 200-char overlap; retrieves `k=10` documents per query.
- `process_query()` in `rag/engine.py` is the synchronous entry point used by the eval pipeline; it uses a `ThreadPoolExecutor` + `asyncio.run()` to avoid "event loop already running" errors. The async streaming path goes through `engine.app.astream_events()` in `api/main.py`.
- Deployment target is Render.com (free tier, Oregon region) — see `render.yaml`.
