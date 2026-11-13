"""The two 'tools' the graph can use: your documents (ChromaDB) and the web."""
CHROMA_DIR = "chroma_db"
COLLECTION = "knowledge"


def get_collection():
    import chromadb

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    # cosine = "how similar in meaning"; distance 0 is identical, bigger is less similar
    return client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})


def reset_collection():
    import chromadb

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass  # nothing to delete


def retrieve(question: str, k: int = 4, max_distance: float = 0.6) -> list[dict]:
    """Find the chunks in ChromaDB closest to the question. Keep only the relevant ones."""
    col = get_collection()
    if col.count() == 0:
        return []
    res = col.query(query_texts=[question], n_results=min(k, col.count()))
    hits = []
    for text, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        if dist <= max_distance:
            hits.append({
                "text": text,
                "label": f"{meta['source']} ({meta['where']})",
                "kind": "doc",
                "distance": round(dist, 3),
            })
    return hits


def web_search(query: str, k: int = 4) -> list[dict]:
    """Search the web with DuckDuckGo (no API key needed)."""
    try:
        from ddgs import DDGS
    except ImportError:  # older package name
        from duckduckgo_search import DDGS

    hits = []
    for r in DDGS().text(query, max_results=k) or []:
        text = (r.get("body") or "").strip()
        if text:
            hits.append({
                "text": text,
                "label": f"{r.get('title', '(no title)')} - {r.get('href', '')}",
                "kind": "web",
            })
    return hits
