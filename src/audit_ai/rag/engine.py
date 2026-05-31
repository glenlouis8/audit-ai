import os
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Literal, TypedDict
from uuid import uuid4

from audit_ai.config import (
    GOOGLE_API_KEY, QDRANT_URL, QDRANT_API_KEY,
    LLM_MODEL, EMBEDDING_MODEL, COLLECTION_NAME,
    RETRIEVAL_K, MAX_RETRIES, HISTORY_WINDOW,
    CACHE_COLLECTION, CACHE_SIMILARITY_THRESHOLD, CACHE_EMBEDDING_DIM,
)

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from langchain_core.runnables import RunnableConfig

from langgraph.graph import StateGraph, END

# =============================================================================
# 1. STATE DEFINITION
# =============================================================================

class GraphState(TypedDict):
    """
    The shared context that flows through every node in the CRAG graph.

    Using TypedDict (rather than a plain dict) gives static type checking and
    makes the contract between nodes explicit — each node declares exactly which
    fields it reads and writes.
    """
    question: str               # The user's original, unmodified question
    search_query: str           # The query actually sent to the vector store (may be rewritten)
    generation: str             # The final answer produced by the generate node
    documents: List[Document]   # The chunks retrieved from Qdrant
    grade: str                  # Relevance verdict from the grader: 'yes' or 'no'
    retry_count: int            # Number of query-rewrite attempts so far
    history: List[Dict[str, str]]  # Prior turns in the conversation


def _format_history(history: List[Dict[str, str]]):
    """
    Converts the plain-dict history format used by the API into typed LangChain
    message objects required by ChatPromptTemplate's MessagesPlaceholder.
    """
    messages = []
    for msg in history:
        if msg.get("role") == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg.get("role") == "assistant":
            messages.append(AIMessage(content=msg["content"]))
    return messages


# =============================================================================
# 2. INITIALIZATION
# =============================================================================

# temperature=0 is intentional: compliance answers must be deterministic and
# grounded in the retrieved text, not creatively embellished.
llm = ChatGoogleGenerativeAI(
    model=LLM_MODEL,
    temperature=0,
    google_api_key=GOOGLE_API_KEY,
)

embeddings = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODEL,
    google_api_key=GOOGLE_API_KEY
)

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

_vector_store: QdrantVectorStore | None = None

def _get_vector_store() -> QdrantVectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = QdrantVectorStore(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding=embeddings,
        )
    return _vector_store


# =============================================================================
# 3. SEMANTIC CACHE
# =============================================================================

def _ensure_cache_collection():
    try:
        client.get_collection(CACHE_COLLECTION)
    except Exception:
        client.create_collection(
            collection_name=CACHE_COLLECTION,
            vectors_config=VectorParams(size=CACHE_EMBEDDING_DIM, distance=Distance.COSINE),
        )
        print(f"---CACHE COLLECTION CREATED: {CACHE_COLLECTION}---")

_ensure_cache_collection()


async def check_cache(query: str) -> dict | None:
    try:
        query_vec = embeddings.embed_query(query)
        results = client.query_points(
            collection_name=CACHE_COLLECTION,
            query=query_vec,
            limit=1,
            score_threshold=CACHE_SIMILARITY_THRESHOLD,
        ).points
        if results:
            print(f"---CACHE HIT (score={results[0].score:.3f})---")
            return results[0].payload
    except Exception as e:
        print(f"Cache check failed: {e}")
    return None


async def store_cache(query: str, answer: str, sources: list):
    try:
        query_vec = embeddings.embed_query(query)
        client.upsert(
            collection_name=CACHE_COLLECTION,
            points=[PointStruct(
                id=str(uuid4()),
                vector=query_vec,
                payload={"answer": answer, "sources": sources},
            )],
        )
        print("---CACHE STORED---")
    except Exception as e:
        print(f"Cache store failed: {e}")


# =============================================================================
# 4. GRAPH NODES
# =============================================================================

_FRAMEWORK_FILES = [
    "nist_framework.pdf",
    "NIST.SP.800-53r5.pdf",
    "ISO_IEC-270012022-ed.3.pdf",
    "trust-services-criteria.pdf",
]

_FRAMEWORK_KEYWORDS = {
    "nist_framework.pdf": [
        "csf", "cybersecurity framework", "nist csf", "nist framework",
        "govern function", "identify function", "protect function", "detect function",
        "respond function", "recover function", "framework profile", "framework tier",
        "organizational profile", "community profile", "csf 2.0", "csf2",
        "risk management strategy", "cybersecurity risk management",
    ],
    "NIST.SP.800-53r5.pdf": [
        "800-53", "sp 800", "nist sp", "ac-", "au-", "ca-", "cm-", "cp-",
        "ia-", "ir-", "ma-", "mp-", "pe-", "pl-", "pm-", "ps-", "pt-",
        "ra-", "sa-", "sc-", "si-", "sr-", "control family", "safeguarding measures",
        "security control", "privacy control", "audit and accountability",
        "access control family", "incident response family", "physical and environmental",
        "risk assessment family", "system and communications", "configuration management",
        "identification and authentication", "contingency planning",
    ],
    "ISO_IEC-270012022-ed.3.pdf": [
        "iso 27001", "iso/iec 27001", "27001", "isms", "information security management system",
        "annex a", "clause 4", "clause 5", "clause 6", "clause 7", "clause 8",
        "clause 9", "clause 10", "risk treatment", "statement of applicability",
        "certification", "certification maintenance", "audit programme", "management review", "nonconformity",
        "continual improvement", "internal audit", "information security policy",
        "leadership", "top management", "information security risk",
        "risk assessment process", "risk treatment process",
    ],
    "trust-services-criteria.pdf": [
        "soc 2", "trust services", "tsc", "common criteria", "cc1", "cc2", "cc3",
        "cc4", "cc5", "cc6", "cc7", "cc8", "cc9", "availability criteria",
        "processing integrity", "confidentiality criteria", "privacy criteria", "aicpa",
    ],
}


def _detect_frameworks(query: str) -> List[str]:
    query_lower = query.lower()
    matched = [
        fname for fname, keywords in _FRAMEWORK_KEYWORDS.items()
        if any(kw in query_lower for kw in keywords)
    ]
    return matched if matched else _FRAMEWORK_FILES


def _search_framework(query: str, filename: str, k: int) -> List:
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    vs = _get_vector_store()
    qdrant_filter = Filter(
        must=[FieldCondition(key="metadata.filename", match=MatchValue(value=filename))]
    )
    return vs.similarity_search(query, k=k, filter=qdrant_filter)


def retrieve(state: GraphState):
    print("---RETRIEVE NODE---")
    query = state.get("search_query") or state["question"]

    target_frameworks = _detect_frameworks(query)
    k_per_framework = max(RETRIEVAL_K, round(12 / len(target_frameworks)))
    print(f"---TARGETING {len(target_frameworks)} framework(s): {[f.split('.')[0] for f in target_frameworks]} (k={k_per_framework} each)---")

    documents = []
    with ThreadPoolExecutor(max_workers=len(target_frameworks)) as executor:
        futures = {executor.submit(_search_framework, query, fname, k_per_framework): fname for fname in target_frameworks}
        for future in as_completed(futures):
            try:
                documents.extend(future.result())
            except Exception as e:
                print(f"---RETRIEVE ERROR ({futures[future]}): {e}---")

    print(f"---RETRIEVED {len(documents)} chunks---")
    return {"documents": documents, "question": state["question"]}


async def grade_documents(state: GraphState):
    """
    Assesses whether any retrieved document is relevant to the question.

    All documents are graded in parallel via asyncio.gather — worst-case latency
    drops from k*LLM_latency to 1*LLM_latency regardless of RETRIEVAL_K.
    """
    print("---GRADE DOCUMENTS NODE---")
    question = state["question"]
    documents = state["documents"]

    prompt = ChatPromptTemplate.from_template(
        "You are a grader assessing relevance of a retrieved document to a user question. \n"
        "Here is the retrieved document: \n\n {context} \n\n"
        "Here is the user question: {question} \n"
        "If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n"
        "Be generous — if the document is even partially relevant or from the same framework domain as the question, grade it yes. \n"
        "Return ONLY the word 'yes' or 'no'."
    )

    chain = prompt | llm | StrOutputParser()

    grades = await asyncio.gather(
        *[chain.ainvoke({"question": question, "context": doc.page_content}) for doc in documents]
    )

    relevant_docs = [doc for doc, g in zip(documents, grades) if "yes" in g.lower()]

    # For cross-framework queries, the grader can eliminate an entire framework's chunks.
    # Guarantee at least one chunk per framework survives by keeping the first chunk
    # from each framework that was retrieved but fully graded out.
    if relevant_docs:
        present_frameworks = {os.path.basename(d.metadata.get("source", "")) for d in relevant_docs}
        for doc in documents:
            fname = os.path.basename(doc.metadata.get("source", ""))
            if fname and fname not in present_frameworks:
                relevant_docs.append(doc)
                present_frameworks.add(fname)

    score = "yes" if relevant_docs else "no"
    print(f"---RESULT: {len(relevant_docs)}/{len(documents)} chunks kept---")

    update = {"grade": score}
    if relevant_docs:
        update["documents"] = relevant_docs
    return update


def transform_query(state: GraphState):
    """
    Rewrites the user's question into a more effective vector search query.

    Natural-language questions often contain pronouns, filler words, or phrasing
    that doesn't match the dense technical vocabulary in the NIST document.
    Rewriting towards domain-specific terms significantly improves retrieval recall.
    """
    print("---TRANSFORM QUERY NODE---")
    question = state["question"]

    prompt = ChatPromptTemplate.from_template(
        "You are generating a specialized vector search query from a user question. \n"
        "The previous search for the question '{question}' failed to yield relevant results. \n"
        "Please re-phrase the question to focus on key searchable terms relevant to the original intent. \n"
        "If the question appears to be about cybersecurity compliance (NIST CSF, NIST SP 800-53, ISO 27001, SOC 2), include core framework keywords. \n"
        "Return ONLY the new query text."
    )

    chain = prompt | llm | StrOutputParser()
    better_query = chain.invoke({"question": question})
    current_retries = state.get("retry_count", 0)
    print(f"---REWRITTEN QUERY: {better_query}---")
    return {"search_query": better_query, "retry_count": current_retries + 1}


async def generate(state: GraphState, config: RunnableConfig):
    """
    Produces the final answer strictly from the retrieved context.

    The system prompt enforces source attribution and prohibits the model from
    drawing on its pretrained knowledge. This ensures answers are always traceable
    back to a specific document, which is a core requirement for compliance auditing.
    The RunnableConfig is forwarded so that LangGraph can intercept token-level
    streaming events for the SSE response in the API layer.
    """
    print("---GENERATE NODE---")
    question = state["question"]
    documents = state["documents"]
    history = state.get("history") or []

    _filename_to_framework = {
        "nist_framework.pdf": "NIST CSF 2.0",
        "NIST.SP.800-53r5.pdf": "NIST SP 800-53",
        "ISO_IEC-270012022-ed.3.pdf": "ISO 27001:2022",
        "trust-services-criteria.pdf": "SOC 2 TSC",
    }
    context_text = "\n\n".join(
        [
            f"[Source: {_filename_to_framework.get(os.path.basename(doc.metadata.get('source', '')), 'Unknown')}, Page {doc.metadata.get('page', '?')}]\n{doc.page_content}"
            for doc in documents
        ]
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system",
            "You are a strict Compliance Auditor AI. "
            "Answer the user's question using ONLY the context provided below. "
            "Do NOT describe what you are doing, do NOT explain your reasoning process, do NOT say 'the user is asking' or 'I need to find'. Start your answer directly. "
            "Quote the relevant text from the source directly and verbatim — do not paraphrase or summarize. "
            "Use block quotes (>) for exact excerpts, then briefly note the source name and page if available. "
            "If multiple passages are relevant, quote each one. "
            "If the documents conflict, point out the difference. "
            "If the context does not contain the answer, state: 'The provided context does not contain information about [topic].'\n\n"
            "Context:\n{context}"
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])

    rag_chain = prompt | llm | StrOutputParser()

    response = await rag_chain.ainvoke(
        {"context": context_text, "question": question, "history": _format_history(history)},
        config=config,
    )

    return {"generation": response}


# =============================================================================
# 4. GRAPH ASSEMBLY
# =============================================================================

workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("transform_query", transform_query)

workflow.set_entry_point("retrieve")

workflow.add_edge("retrieve", "grade_documents")
workflow.add_edge("transform_query", "retrieve")


def decide_to_generate(state: GraphState):
    """
    Routing function for the conditional edge after grade_documents.

    The retry cap at 3 prevents infinite loops when no relevant documents exist
    in the knowledge base. After 3 failed rewrites, we generate from whatever
    partial context is available rather than hanging indefinitely.
    """
    grade = state.get("grade")
    retries = state.get("retry_count", 0)

    if grade == "yes":
        return "generate"
    elif retries >= MAX_RETRIES:
        return "generate"
    else:
        return "transform_query"


workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {"generate": "generate", "transform_query": "transform_query"},
)

workflow.add_edge("generate", END)

app = workflow.compile()


# =============================================================================
# 5. PUBLIC INTERFACE
# =============================================================================

_CHAT_KEYWORDS = {
    "hi", "hello", "hey", "thanks", "thank you", "bye", "goodbye",
    "who are you", "what are you", "what can you do", "help",
}


def route_query(user_query: str, history: List[Dict[str, str]] = None) -> Literal["chat", "search"]:
    """
    Classifies the user's intent before invoking the full CRAG graph.

    Routing is done outside the compiled graph so that casual messages (greetings,
    off-topic questions) short-circuit immediately without touching the vector store.
    The last 3 conversation turns are included so the router can correctly handle
    follow-up questions that reference prior context (e.g., "can you elaborate?").
    When in doubt, the router defaults to 'chat' to avoid unnecessarily expensive
    retrieval on non-compliance queries.

    A keyword pre-filter runs first to skip the LLM call entirely for obvious greetings.
    """
    query_lower = user_query.lower().strip()

    # Fast path: obvious greetings/identity — whole-word match only to avoid
    # false positives like "hi" matching inside "leadership" or "this"
    import re
    if len(user_query) < 60 and any(
        re.search(r'\b' + re.escape(kw) + r'\b', query_lower) for kw in _CHAT_KEYWORDS
    ):
        return "chat"

    # Fast path: explicit compliance terms — skip LLM router, go straight to search
    _SEARCH_KEYWORDS = {
        "nist", "iso 27001", "soc 2", "800-53", "csf", "isms", "tsc", "aicpa",
        "function", "control", "framework", "compliance", "audit", "policy",
        "govern", "identify", "protect", "detect", "respond", "recover",
        "encrypt", "cryptograph", "access control", "incident", "risk",
        "annex", "clause", "criteria", "certification", "safeguard",
        "leadership", "requirement", "management system",
    }
    if any(kw in query_lower for kw in _SEARCH_KEYWORDS):
        return "search"

    history = history or []
    history_context = ""
    if history:
        recent = history[-HISTORY_WINDOW:]  # last 3 exchanges (user + assistant per turn)
        history_context = "Previous conversation:\n" + "\n".join(
            f"{m['role'].title()}: {m['content']}" for m in recent
        ) + "\n\n"

    prompt = ChatPromptTemplate.from_template(
        "You are a router. Classify user input into one of two categories: \n"
        "1. 'chat': Greetings, identity checks, unrelated/nonsense questions (dogs, painting, sports), or general help. \n"
        "2. 'search': Any question about cybersecurity compliance frameworks (NIST CSF 2.0, NIST SP 800-53, ISO 27001, SOC 2), including their requirements, controls, functions, clauses, criteria, policies, leadership obligations, risk management, audit processes, or certification. \n\n"
        "{history_context}"
        "Input: {query} \n"
        "If the question asks about requirements, controls, functions, criteria, clauses, or policies of any framework — return 'search'. \n"
        "Only return 'chat' if the input is clearly a greeting, identity check, or completely unrelated to cybersecurity compliance. \n"
        "When in doubt, return 'search'. \n"
        "Return ONLY one word: 'chat' or 'search'."
    )

    chain = prompt | llm | StrOutputParser()
    intent = chain.invoke({"query": user_query, "history_context": history_context}).strip().lower()

    if "chat" in intent:
        return "chat"
    return "search"


# The chat prompt is defined at module level so the chain is instantiated once
# and reused across requests rather than being rebuilt on every call.
_chat_prompt = ChatPromptTemplate.from_messages([
    ("system",
        "You are **AuditAI**, a professional compliance auditor specializing in cybersecurity frameworks: NIST CSF 2.0, NIST SP 800-53, ISO 27001:2022, and SOC 2.\n\n"
        "Rules:\n"
        "1. Respond naturally to greetings and identity questions.\n"
        "2. If the user asks about your capabilities, mention that you can audit against NIST CSF 2.0, NIST SP 800-53, ISO 27001, and SOC 2 using your Agentic RAG engine.\n"
        "3. For ANY question that is not a greeting or about your identity/capabilities, respond with exactly: "
        "'I can only assist with cybersecurity compliance questions (NIST CSF 2.0, NIST SP 800-53, ISO 27001, SOC 2). Please ask me something related to those frameworks.'\n"
        "4. Keep responses concise (1-3 sentences)."
    ),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{query}"),
])

chat_chain = _chat_prompt | llm | StrOutputParser()


def run_chat_logic(user_query: str, history: List[Dict[str, str]] = None):
    history = history or []
    answer = chat_chain.invoke({"query": user_query, "history": _format_history(history)})
    return {"answer": answer}


def process_query(user_query: str, history: List[Dict[str, str]] = None):
    """
    Synchronous entry point for non-streaming execution (used by the eval pipeline).

    The graph's async invoke is wrapped in a dedicated thread via ThreadPoolExecutor
    so that asyncio.run() can safely create a fresh event loop. This avoids the
    "event loop already running" error that occurs when calling async code from
    within an already-running event loop (e.g. Jupyter, async test runners).
    """
    history = history or []
    intent = route_query(user_query, history)

    if intent == "chat":
        return run_chat_logic(user_query, history)

    inputs = {"question": user_query, "history": history}
    try:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(asyncio.run, app.ainvoke(inputs))
            final_state = future.result()
        return {
            "answer": final_state["generation"],
            "context": final_state["documents"],
        }
    except Exception as e:
        print(f"Graph Error: {e}")
        return {"answer": "Error processing request.", "context": []}
