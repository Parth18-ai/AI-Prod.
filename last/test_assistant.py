"""Tests with a FAKE model and FAKE tools: no Ollama, internet or ChromaDB needed.

Part 1 tests each worker/signpost alone. Part 2 tests the whole graph
(skipped automatically if langgraph is not installed).
"""
import pytest

import nodes

DOC = {"text": "RAG means retrieval-augmented generation.",
       "label": "docs/rag_basics.md (chunk 1)", "kind": "doc", "distance": 0.31}
WEB = {"text": "Some web text about the topic.",
       "label": "Example page - https://example.com", "kind": "web"}


def fake_llm(monkeypatch, replies):
    """Make llm.ask return the given replies one by one."""
    queue = list(replies)
    monkeypatch.setattr(nodes.llm, "ask", lambda *a, **k: queue.pop(0))


# ------------------------------------------------------------ part 1: single pieces

def test_router_picks_web(monkeypatch):
    fake_llm(monkeypatch, ["WEB"])
    assert nodes.router_node({"question": "latest AI news today"})["route"] == "web"


def test_router_defaults_to_docs_on_odd_reply(monkeypatch):
    fake_llm(monkeypatch, ["hmm, not sure"])
    assert nodes.router_node({"question": "what is RAG?"})["route"] == "docs"


def test_router_survives_model_error(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("ollama down")
    monkeypatch.setattr(nodes.llm, "ask", boom)
    assert nodes.router_node({"question": "x"})["route"] == "docs"


def test_signposts():
    assert nodes.route_after_router({"route": "web"}) == "web_search"
    assert nodes.route_after_router({"route": "docs"}) == "retriever"
    assert nodes.route_after_retriever({"doc_hits": [DOC]}) == "writer"
    assert nodes.route_after_retriever({"doc_hits": []}) == "web_search"
    assert nodes.route_after_web({"web_hits": [WEB]}) == "writer"
    assert nodes.route_after_web({"web_hits": []}) == "fallback"
    assert nodes.route_after_writer({"found": True}) == "end"
    assert nodes.route_after_writer({"found": False}) == "web_search"
    assert nodes.route_after_writer({"found": False, "web_tried": True}) == "fallback"


def test_web_search_failure_is_caught(monkeypatch):
    def boom(*a, **k):
        raise ConnectionError("offline")
    monkeypatch.setattr(nodes.tools, "web_search", boom)
    out = nodes.web_search_node({"question": "x"})
    assert out["web_hits"] == [] and out["web_tried"] is True


def test_writer_keeps_only_cited_sources(monkeypatch):
    fake_llm(monkeypatch, ["RAG retrieves text first [1]. Nothing about [7]."])
    out = nodes.writer_node({"question": "what is RAG?", "doc_hits": [DOC], "web_hits": [WEB]})
    assert out["found"] is True
    assert out["sources"] == ["[1] docs/rag_basics.md (chunk 1)"]   # [7] does not exist, ignored


def test_writer_warns_when_no_citation(monkeypatch):
    fake_llm(monkeypatch, ["RAG retrieves text first."])
    out = nodes.writer_node({"question": "q", "doc_hits": [DOC]})
    assert "did not cite" in out["answer"]
    assert len(out["sources"]) == 1


def test_writer_not_found(monkeypatch):
    fake_llm(monkeypatch, ["NOT_FOUND"])
    assert nodes.writer_node({"question": "q", "doc_hits": [DOC]})["found"] is False


# ------------------------------------------------------------ part 2: the whole graph

@pytest.fixture
def app():
    pytest.importorskip("langgraph")
    import graph
    return graph.build_graph()


def run(app, question="q"):
    return app.invoke({"question": question, "trace": []})


def test_docs_path(app, monkeypatch):
    fake_llm(monkeypatch, ["DOCS", "RAG means retrieval-augmented generation [1]."])
    monkeypatch.setattr(nodes.tools, "retrieve", lambda *a, **k: [DOC])
    result = run(app)
    assert result["found"] is True
    assert result["sources"] == ["[1] docs/rag_basics.md (chunk 1)"]
    assert not any("web_search" in t for t in result["trace"])   # web never needed


def test_web_path_when_docs_have_nothing(app, monkeypatch):
    fake_llm(monkeypatch, ["DOCS", "The web says so [1]."])
    monkeypatch.setattr(nodes.tools, "retrieve", lambda *a, **k: [])
    monkeypatch.setattr(nodes.tools, "web_search", lambda *a, **k: [WEB])
    result = run(app)
    assert result["found"] is True
    assert result["sources"][0].startswith("[1] Example page")


def test_router_can_send_straight_to_web(app, monkeypatch):
    fake_llm(monkeypatch, ["WEB", "Fresh news [1]."])
    monkeypatch.setattr(nodes.tools, "web_search", lambda *a, **k: [WEB])
    result = run(app)
    assert result["found"] is True
    assert not any(t.startswith("retriever") for t in result["trace"])


def test_fallback_when_nothing_anywhere(app, monkeypatch):
    fake_llm(monkeypatch, ["DOCS"])
    monkeypatch.setattr(nodes.tools, "retrieve", lambda *a, **k: [])
    monkeypatch.setattr(nodes.tools, "web_search", lambda *a, **k: [])
    result = run(app)
    assert result["found"] is False
    assert result["answer"] == nodes.NOT_FOUND_MESSAGE
    assert result["sources"] == []


def test_fallback_when_web_crashes(app, monkeypatch):
    def boom(*a, **k):
        raise ConnectionError("offline")
    fake_llm(monkeypatch, ["DOCS"])
    monkeypatch.setattr(nodes.tools, "retrieve", lambda *a, **k: [])
    monkeypatch.setattr(nodes.tools, "web_search", boom)
    assert run(app)["answer"] == nodes.NOT_FOUND_MESSAGE


def test_writer_not_found_tries_web_once_then_falls_back(app, monkeypatch):
    # docs found, but the writer says NOT_FOUND -> web -> writer says NOT_FOUND -> fallback
    fake_llm(monkeypatch, ["DOCS", "NOT_FOUND", "NOT_FOUND"])
    monkeypatch.setattr(nodes.tools, "retrieve", lambda *a, **k: [DOC])
    monkeypatch.setattr(nodes.tools, "web_search", lambda *a, **k: [WEB])
    result = run(app)
    assert result["answer"] == nodes.NOT_FOUND_MESSAGE
    assert sum(t.startswith("web_search") for t in result["trace"]) == 1   # no endless loop
