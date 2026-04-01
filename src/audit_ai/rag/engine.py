import os
import asyncio
from typing import Dict, List, Literal, TypedDict

from audit_ai.config import (
    GOOGLE_API_KEY, QDRANT_URL, QDRANT_API_KEY,
    LLM_MODEL, EMBEDDING_MODEL, COLLECTION_NAME,
    RETRIEVAL_K, MAX_RETRIES, HISTORY_WINDOW,
)

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
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

vector_store = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings,
)


# =============================================================================
# 3. GRAPH NODES
# =============================================================================

def retrieve(state: GraphState):
    """
    Queries the vector store for the most relevant document chunks.

    If the query was previously rewritten by transform_query, that improved
    version is used instead of the original question to maximise recall.
    k=10 was chosen to give the grader enough coverage without inflating
    the context window passed to the generator.
    """
    print("---RETRIEVE NODE---")
    query = state.get("search_query") or state["question"]
    documents = vector_store.similarity_search(query, k=RETRIEVAL_K)
    return {"documents": documents, "question": state["question"]}


def grade_documents(state: GraphState):
    """
    Assesses whether any retrieved document is relevant to the question.

    A short-circuit strategy is used: as soon as one relevant document is found,
    grading stops. This avoids unnecessary LLM calls when the first chunk already
    confirms a good retrieval.
    """
    print("---GRADE DOCUMENTS NODE---")
    question = state["question"]
    documents = state["documents"]

    prompt = ChatPromptTemplate.from_template(
        "You are a grader assessing relevance of a retrieved document to a user question. \n"
        "Here is the retrieved document: \n\n {context} \n\n"
        "Here is the user question: {question} \n"
        "If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n"
        "Return ONLY the word 'yes' or 'no'."
    )

    chain = prompt | llm | StrOutputParser()

    score = "no"
    for doc in documents:
        grade = chain.invoke({"question": question, "context": doc.page_content})
        if "yes" in grade.lower():
            score = "yes"
            break

    print(f"---RESULT: Documents relevant? {score.upper()}---")
    return {"grade": score}


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
        "If the question appears to be about NIST or cybersecurity policies, include core framework keywords. \n"
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

    context_text = "\n\n".join(
        [
            f"[Source: NIST CSF 2.0, Page {doc.metadata.get('page', '?')}]\n{doc.page_content}"
            for doc in documents
        ]
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system",
            "You are a strict Compliance Auditor AI. "
            "Answer the user's question using ONLY the context provided below. "
            "Quote the relevant text from the source directly and verbatim — do not paraphrase or summarize. "
            "Use block quotes (>) for exact excerpts, then briefly note the source name and page if available. "
            "If multiple passages are relevant, quote each one. "
            "If the documents conflict, point out the difference. "
            "If the context is empty, simply state that the specific information is missing from the database.\n\n"
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

def route_query(user_query: str, history: List[Dict[str, str]] = None) -> Literal["chat", "search"]:
    """
    Classifies the user's intent before invoking the full CRAG graph.

    Routing is done outside the compiled graph so that casual messages (greetings,
    off-topic questions) short-circuit immediately without touching the vector store.
    The last 3 conversation turns are included so the router can correctly handle
    follow-up questions that reference prior context (e.g., "can you elaborate?").
    When in doubt, the router defaults to 'chat' to avoid unnecessarily expensive
    retrieval on non-compliance queries.
    """
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
        "2. 'search': Specific, serious questions about NIST CSF 2.0, organizational policies, or compliance audits. \n\n"
        "{history_context}"
        "Input: {query} \n"
        "If you are even slightly unsure if it is a compliance query, return 'chat'. \n"
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
        "You are **AuditAI**, a professional auditor specializing in the **NIST Cybersecurity Framework (CSF) 2.0**.\n\n"
        "Rules:\n"
        "1. Respond naturally to greetings and identity questions.\n"
        "2. If the user asks about your capabilities, mention that you can perform deep-dive audits against organizational policies using your Agentic RAG engine.\n"
        "3. For ANY question that is not a greeting or about your identity/capabilities, respond with exactly: "
        "'I can only assist with NIST CSF 2.0 compliance and cybersecurity audit questions. Please ask me something related to that.'\n"
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
