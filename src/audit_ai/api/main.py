import os
import json
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import the graph AND the router logic
from audit_ai.rag.engine import app as audit_graph, route_query, run_chat_logic

app = FastAPI(
    title="AuditAI Agent API",
    description="A Streaming Agentic RAG API for NIST Compliance",
    version="2.0",
)

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
    Robust Generator: Streams text and conditionally filters sources if the AI doesn't know the answer.
    """
    history = history or []

    # --- 1. ROUTER (Fast Path) ---
    intent = route_query(query, history)

    if intent == "chat":
        response = run_chat_logic(query, history)
        answer_text = response["answer"]
        tokens = answer_text.split(" ")
        for token in tokens:
            payload = json.dumps({"type": "token", "content": token + " "})
            yield f"{payload}\n"
        # Chat intent implies no sources
        payload = json.dumps({"type": "sources", "content": []})
        yield f"{payload}\n"
        return

    # --- 2. GRAPH (Search Path) ---
    captured_sources = []
    full_answer_accumulator = ""  # Track the full answer to check for "I don't know"

    try:
        # Stream events from the graph
        async for event in audit_graph.astream_events(
            {"question": query, "history": history}, version="v1"
        ):
            kind = event["event"]
            data = event.get("data", {})

            # A. Capture Sources (When retrieval finishes)
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

            # B. Capture Tokens
            if "chunk" in data:
                # Only stream tokens from the 'generate' graph node to avoid leaking grader output ("yes"/"no")
                if event.get("metadata", {}).get("langgraph_node") != "generate":
                    continue

                chunk = data["chunk"]
                content = ""
                if hasattr(chunk, "content"):
                    content = chunk.content
                elif isinstance(chunk, dict) and "content" in chunk:
                    content = chunk["content"]

                if content:
                    full_answer_accumulator += content  # Accumulate text
                    payload = json.dumps({"type": "token", "content": content})
                    yield f"{payload}\n"

    except Exception as e:
        print(f"Graph Error: {e}")
        err_payload = json.dumps(
            {"type": "token", "content": f"\n[System Error: {str(e)}]"}
        )
        yield f"{err_payload}\n"

    # --- 3. SMART SOURCE FILTERING ---
    # We defined standard refusal phrases in the prompt. If the AI says them, we hide sources.
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
        # If the AI admitted it doesn't know, send ZERO sources.
        payload = json.dumps({"type": "sources", "content": []})
    else:
        # Otherwise, send the retrieved sources.
        payload = json.dumps({"type": "sources", "content": captured_sources})

    yield f"{payload}\n"


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    return StreamingResponse(
        run_agent_stream(request.query, request.history), media_type="application/x-ndjson"
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
