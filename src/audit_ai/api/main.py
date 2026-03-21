import os
import json
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from audit_ai.rag.engine import app as audit_graph, route_query, chat_chain, _format_history

app = FastAPI(
    title="AuditAI Agent API",
    description="A Streaming Agentic RAG API for NIST Compliance",
    version="2.0",
)

# Wildcard CORS is intentional for this deployment: the frontend is served from a
# separate origin and the API does not expose any authenticated user data.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    query: str
    history: Optional[List[Dict[str, str]]] = []  # [{"role": "user"/"assistant", "content": "..."}]


async def run_agent_stream(query: str, history: list = None):
    """
    Async generator that streams the agent's response as NDJSON tokens.

    The router runs first to decide whether the query needs full RAG retrieval or
    can be answered by the lightweight chat chain. This avoids a round-trip to
    Qdrant for messages that don't require document lookup.

    Sources are withheld when the model's answer contains a known refusal phrase,
    preventing the UI from displaying irrelevant citations alongside a "no answer" response.
    """
    history = history or []

    # --- 1. ROUTING ---
    # Classify intent before touching the vector store to avoid unnecessary retrieval cost.
    intent = route_query(query, history)

    if intent == "chat":
        async for chunk in chat_chain.astream({"query": query, "history": _format_history(history)}):
            if chunk:
                payload = json.dumps({"type": "token", "content": chunk})
                yield f"{payload}\n"
        payload = json.dumps({"type": "sources", "content": []})
        yield f"{payload}\n"
        return

    # --- 2. RAG GRAPH ---
    captured_sources = []
    full_answer_accumulator = ""

    try:
        async for event in audit_graph.astream_events(
            {"question": query, "history": history}, version="v1"
        ):
            kind = event["event"]
            data = event.get("data", {})

            # Capture the source documents when the retrieve node completes so they
            # can be attached to the final response after generation finishes.
            if kind == "on_chain_end" and event.get("name") == "retrieve":
                if "output" in data and data["output"]:
                    docs = data["output"].get("documents", [])
                    captured_sources = [
                        {
                            "file": d.metadata.get("source_file", "NIST CSF 2.0"),
                            "page": d.metadata.get("page", 0),
                            "text": d.page_content[:200] + "...",
                        }
                        for d in docs
                    ]

            # Stream only tokens produced by the generate node. Other nodes (grader,
            # query rewriter) also emit LLM chunks, but those are internal signals
            # ('yes'/'no', rewritten queries) that should never be shown to the user.
            if "chunk" in data:
                if event.get("metadata", {}).get("langgraph_node") != "generate":
                    continue

                chunk = data["chunk"]
                content = ""
                if hasattr(chunk, "content"):
                    content = chunk.content
                elif isinstance(chunk, dict) and "content" in chunk:
                    content = chunk["content"]

                if content:
                    full_answer_accumulator += content
                    payload = json.dumps({"type": "token", "content": content})
                    yield f"{payload}\n"

    except Exception as e:
        print(f"Graph Error: {e}")
        err_payload = json.dumps(
            {"type": "token", "content": f"\n[System Error: {str(e)}]"}
        )
        yield f"{err_payload}\n"

    # --- 3. SOURCE FILTERING ---
    # The system prompt instructs the model to use specific phrases when it cannot
    # answer from the provided context. Detecting those phrases here lets us suppress
    # sources on the client side, avoiding the misleading appearance of citations
    # alongside a "not found" response.
    refusal_phrases = [
        "missing from the database",
        "does not mention",
        "cannot answer",
        "no information",
        "context does not contain",
        "not mentioned in the provided documents",
    ]

    is_refusal = any(
        phrase in full_answer_accumulator.lower() for phrase in refusal_phrases
    )

    if is_refusal:
        payload = json.dumps({"type": "sources", "content": []})
    else:
        payload = json.dumps({"type": "sources", "content": captured_sources})

    yield f"{payload}\n"


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    # Cache-Control and X-Accel-Buffering headers are required to prevent Nginx
    # and CDN layers from buffering the stream before it reaches the client.
    return StreamingResponse(
        run_agent_stream(request.query, request.history),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate", "X-Accel-Buffering": "no"},
    )


@app.get("/health")
def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    if not os.getenv("GOOGLE_API_KEY"):
        print("❌ Error: GOOGLE_API_KEY is missing!")
    else:
        print("🚀 Starting AuditAI Agent...")
        uvicorn.run(app, host="0.0.0.0", port=8000)
