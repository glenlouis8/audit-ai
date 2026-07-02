# AuditAI — Interview Recap

## What is this project?

Agentic RAG backend that audits organizations against **NIST Cybersecurity Framework 2.0**.
Uses a **Corrective RAG (CRAG)** pattern in LangGraph with token-level streaming via FastAPI.

Stack: Python, FastAPI, LangGraph, Qdrant Cloud, Google Gemini API, Next.js frontend.

---

## `ingestion.py` — One-time pipeline to build the knowledge base

**Three stages:**

1. **Load** — `PyPDFLoader` reads NIST CSF 2.0 PDF → `Document` objects with `{page_content, metadata}`
2. **Chunk** — `RecursiveCharacterTextSplitter` breaks into 1000-char chunks, 200-char overlap
   - Overlap preserves sentence continuity across chunk boundaries (important for multi-clause policy text)
3. **Embed + Store** — `gemini-embedding-001` → 3072-dim vectors → upserted to Qdrant Cloud (`compliance_audit` collection)
   - `force_recreate=True` wipes collection first — no stale vectors if model or PDF ever changes

Run once. The API never writes to Qdrant.

**Talking points:**
- Chunking tradeoffs: larger chunks = more context, smaller = better retrieval precision
- `force_recreate` vs incremental upserts — current approach is simple/safe but doesn't scale to large doc sets
- Why overlap matters: policy clauses span sentences; hard boundary would cut a clause mid-meaning

---

## `engine.py` — The brain (five sections)

### 1. `GraphState` (TypedDict)
Shared state flowing through every graph node.

| Field | Purpose |
|---|---|
| `question` | Original unmodified user question |
| `search_query` | Possibly rewritten query sent to vector store |
| `documents` | Chunks retrieved from Qdrant |
| `grade` | `"yes"` / `"no"` from relevance grader |
| `generation` | Final answer from generate node |
| `retry_count` | Number of query-rewrite attempts |
| `history` | Prior conversation turns |

### 2. Initialization
Module-level singletons: LLM (`temperature=0` — compliance needs determinism), embeddings, Qdrant client.

**Vector store is lazy** via `_get_vector_store()` — defers construction until first request.
- Why: `QdrantVectorStore.__init__` calls `get_collection` immediately. If collection doesn't exist at boot, server crashes. Lazy init prevents this.

### 3. Semantic Cache
Second Qdrant collection (`semantic_cache`). On every search query:
1. Embed the question
2. Cosine search against cached questions
3. Score ≥ 0.93 → return stored answer, skip entire RAG graph
4. After successful generation → write to cache

Cuts repeat-query cost to near zero.

### 4. CRAG Graph Nodes

| Node | What it does |
|---|---|
| `retrieve` | Vector similarity search, k=4 chunks |
| `grade_documents` | **Parallel** LLM calls per chunk — "yes/no relevant?" (`asyncio.gather`) |
| `transform_query` | Rewrites question into better search terms (NIST keywords, removes filler) |
| `generate` | Final answer, context-only, streams via `RunnableConfig` |

**Flow:**
```
retrieve → grade_documents → [decide_to_generate]
                                  ↓ yes → generate → END
                                  ↓ no + retries < 3 → transform_query → retrieve (loop)
                                  ↓ retries ≥ 3 → generate anyway (partial context fallback)
```

**Why parallel grading matters:** k=4 docs. Serial = 4× LLM latency. Parallel = 1× LLM latency.

**Why retry cap:** Without it, no-match queries loop forever. After 3 rewrites, generate from whatever was retrieved.

### 5. Public Interface

- `route_query()` — runs **before** the graph. Keyword fast-path (greetings skip LLM), then LLM classifier for ambiguous. Defaults to `"chat"` when unsure (avoids wasted Qdrant calls).
- `run_chat_logic()` — lightweight response for greetings/identity. No vector store.
- `process_query()` — sync wrapper for eval pipeline. `ThreadPoolExecutor` + `asyncio.run()` avoids "event loop already running" in Jupyter/async test runners.

---

## `main.py` — FastAPI + Streaming

**`POST /chat` flow:**

1. **Route** — chat vs. search
2. **Cache check** — return cached answer if hit
3. **RAG graph** — `astream_events()` with 120s timeout
   - Filter: only forward tokens from `generate` node (`langgraph_node == "generate"`)
   - Grader + query-rewriter LLM chunks are internal signals — never shown to user
4. **Source filtering** — suppress sources if answer contains refusal phrases ("missing from database", "cannot answer", etc.) — avoids misleading citations on no-answer responses

Response format: NDJSON (newline-delimited JSON)
```json
{"type": "token", "content": "The"}
{"type": "sources", "content": [{"file": "NIST CSF 2.0", "page": 42, "text": "..."}]}
```

**`GET /health`** — checks Qdrant reachable, returns 503 if not.

---

## Evals Pipeline — RAGAS

### Two-step process:

**Step 1 — `collector.py`**
- Reads `test.csv` (question + ground truth)
- Runs each through `process_query()` (full RAG engine)
- Saves `{question, answer, contexts, ground_truth}` → `rag_results.json`
- 1s sleep between calls — Gemini free-tier rate limit guard

**Step 2 — `evaluator.py`**
- Loads `rag_results.json`, evaluates first 10 records (token budget)
- Uses same Gemini model as app (`gemini-2.5-flash-lite`) as judge
- Outputs `ragas_results.csv` + `ragas_report.md`

### Four RAGAS metrics:

| Metric | Question it answers | How measured |
|---|---|---|
| **Faithfulness** | Does answer only claim things the chunks support? | LLM checks each claim vs. retrieved context |
| **Answer Relevancy** | Does answer address the question? | Embedding similarity: reverse-generated question → original |
| **Context Precision** | Are relevant chunks ranked first? | Ground truth vs. chunk ordering |
| **Context Recall** | Did retrieval find everything needed? | Ground truth coverage by retrieved chunks |

### Last results (Feb 2026, 10 questions):

| Metric | Score | Notes |
|---|---|---|
| Faithfulness | **1.00** | Zero hallucination — most important for compliance tool |
| Context Recall | **0.90** | Retrieved almost all needed content |
| Context Precision | **0.78** | Relevant chunks found but not always ranked optimally |
| Answer Relevancy | **0.76** | Lower because system quotes verbatim (verbose), not tight summaries |

**Why Faithfulness=1.0 matters most:** Compliance auditing requires every claim to be traceable to source. Hallucination = liability.

**Why Answer Relevancy is lower than Faithfulness:** RAGAS measures relevancy by embedding similarity between the answer and a "reverse-generated" question. Verbatim block-quote answers are technically correct but score lower on embedding similarity than concise paraphrases.

---

## Key architectural decisions to know cold

| Decision | Why |
|---|---|
| `temperature=0` on LLM | Compliance answers must be deterministic |
| Lazy vector store init | Prevents boot crash when collection missing |
| `prefer_grpc=False` in ingestion | Qdrant Cloud only serves HTTP, not gRPC |
| Parallel grading (`asyncio.gather`) | k×LLM latency → 1×LLM latency |
| Retry cap at 3 | Prevents infinite loop on no-match queries |
| Route before graph | Avoids Qdrant calls for greetings/off-topic |
| Semantic cache (0.93 threshold) | Near-zero cost on repeated compliance queries |
| `force_recreate=True` in ingestion | Prevents dim mismatch if embedding model changes |
| `MAX_QUERY_LENGTH=5000` | Input sanitation at API boundary |
| Source suppression on refusal | No misleading citations on "I don't know" responses |
