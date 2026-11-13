"""The workers (nodes) and signposts (routers) of the research assistant.

There is NO LangGraph import here on purpose: these are plain functions that
read the shared state (the whiteboard) and return what they want to update.
That makes them easy to read and easy to test. graph.py connects them.
"""
import operator
import os
import re
from typing import Annotated, TypedDict

import llm
import tools

MAX_DISTANCE = float(os.getenv("RELEVANCE_MAX_DISTANCE", "0.6"))

NOT_FOUND_MESSAGE = (
    "I could not find this in your documents or on the web.\n"
    "You can: (1) rephrase the question, (2) add a relevant file to docs/ and run "
    "`python ingest.py`, or (3) check your internet connection."
)

ROUTER_PROMPT = """Decide where to look FIRST for this question. Reply with ONE word only.

DOCS = general or background knowledge that is likely in the user's own notes and documents
WEB  = recent or time-sensitive facts (news, prices, "latest", "today", current events)

Question: {question}
Answer (DOCS or WEB):"""


class State(TypedDict, total=False):
    """The shared whiteboard. Every node reads and writes it."""
    question: str
    route: str                 # "docs" or "web" (the router's choice)
    web_tried: bool            # have we already searched the web?
    doc_hits: list[dict]       # relevant chunks from ChromaDB
    web_hits: list[dict]       # results from the web search
    answer: str
    sources: list[str]         # the citations shown to the user
    found: bool                # did we manage to answer?
    trace: Annotated[list[str], operator.add]   # a diary of steps; new lines are ADDED


# ---------------------------------------------------------------- workers (nodes)

def router_node(state: State) -> dict:
    try:
        reply = llm.ask(ROUTER_PROMPT.format(question=state["question"])).upper()
    except Exception:
        return {"route": "docs", "trace": ["router: model unavailable, defaulting to docs"]}
    route = "web" if ("WEB" in reply and "DOCS" not in reply) else "docs"
    return {"route": route, "trace": [f"router: chose {route}"]}


def retriever_node(state: State) -> dict:
    try:
        hits = tools.retrieve(state["question"], max_distance=MAX_DISTANCE)
    except Exception as e:
        return {"doc_hits": [], "trace": [f"retriever: error ({type(e).__name__})"]}
    if hits:
        note = f"retriever: {len(hits)} relevant chunk(s), best distance {hits[0]['distance']}"
    else:
        note = "retriever: nothing relevant in your documents"
    return {"doc_hits": hits, "trace": [note]}


def web_search_node(state: State) -> dict:
    try:
        hits = tools.web_search(state["question"])
    except Exception as e:  # no internet, rate limit, ... -> go to the fallback path
        return {"web_hits": [], "web_tried": True,
                "trace": [f"web_search: failed ({type(e).__name__})"]}
    return {"web_hits": hits, "web_tried": True,
            "trace": [f"web_search: {len(hits)} result(s)"]}


def writer_node(state: State) -> dict:
    hits = state.get("doc_hits", []) + state.get("web_hits", [])
    if not hits:
        return {"found": False, "trace": ["writer: no sources to write from"]}

    numbered = "\n\n".join(f"[{i}] ({h['label']})\n{h['text']}" for i, h in enumerate(hits, 1))
    prompt = (
        "You are a careful research assistant. Answer the question using ONLY the "
        "numbered sources below.\n"
        "Rules:\n"
        "- After each fact, cite the source number like [1] or [2].\n"
        "- Keep it short: 5 sentences at most.\n"
        "- If the sources do not contain the answer, reply with exactly: NOT_FOUND\n\n"
        f"Sources:\n{numbered}\n\nQuestion: {state['question']}\nAnswer:"
    )
    reply = llm.ask(prompt)

    if "NOT_FOUND" in reply:
        return {"found": False, "trace": ["writer: sources did not contain the answer"]}

    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", reply) if 1 <= int(n) <= len(hits)})
    if cited:
        sources = [f"[{i}] {hits[i - 1]['label']}" for i in cited]
        note = f"writer: answered, cited {cited}"
    else:  # the model forgot to cite: be honest about it
        sources = [f"[{i}] {h['label']}" for i, h in enumerate(hits, 1)]
        reply += "\n\n(Warning: the model did not cite specific sources. All consulted sources are listed.)"
        note = "writer: answered, but NO citations were given"
    return {"answer": reply, "sources": sources, "found": True, "trace": [note]}


def fallback_node(state: State) -> dict:
    return {"answer": NOT_FOUND_MESSAGE, "sources": [], "found": False,
            "trace": ["fallback: could not find an answer"]}


# ---------------------------------------------------------------- signposts (routers)

def route_after_router(state: State) -> str:
    return "web_search" if state.get("route") == "web" else "retriever"


def route_after_retriever(state: State) -> str:
    return "writer" if state.get("doc_hits") else "web_search"


def route_after_web(state: State) -> str:
    return "writer" if state.get("web_hits") else "fallback"


def route_after_writer(state: State) -> str:
    if state.get("found"):
        return "end"
    # Not answered: try the web once, then give up politely.
    return "fallback" if state.get("web_tried") else "web_search"
