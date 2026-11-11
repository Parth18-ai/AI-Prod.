"""
PDF Q&A CLI (RAG)

Pipeline:
  PDF -> text per page -> chunks -> embeddings -> ChromaDB -> question -> top chunks -> LLM answer + citations

Usage:
  python rag_cli.py path/to/file.pdf
  python rag_cli.py path/to/file.pdf --max-distance 0.5 --top-k 4
"""

import argparse
import hashlib
import sys
from pathlib import Path

import chromadb
import pymupdf as fitz
import ollama

EMBED_MODEL = "nomic-embed-text"   # run: ollama pull nomic-embed-text
CHAT_MODEL = "llama3.2"            # run: ollama pull llama3.2
CHUNK_SIZE = 800                   # characters per chunk
CHUNK_OVERLAP = 150                # characters shared between neighbouring chunks
TOP_K = 4                          # how many chunks to retrieve
MAX_DISTANCE = 0.6                 # cosine distance; lower = more similar. Tune this!
DB_PATH = "./chroma_db"
NOT_FOUND_TOKEN = "NOT_FOUND"


# ---------- 1. Load PDF ----------
def load_pdf(path: Path) -> list[tuple[int, str]]:
    """Return a list of (page_number, text). Page numbers start at 1."""
    if not path.exists():
        sys.exit(f"Error: file not found: {path}")
    try:
        doc = fitz.open(path)
    except Exception as e:
        sys.exit(f"Error: could not open PDF ({e})")

    if doc.needs_pass:
        sys.exit("Error: this PDF is password protected.")

    pages = []
    for i, page in enumerate(doc, start=1):
        text = page.get_text().strip()
        if text:
            pages.append((i, text))

    if not pages:
        sys.exit(
            "Error: no text found. This PDF may be scanned images "
            "(it would need OCR first)."
        )
    return pages


# ---------- 2. Chunk ----------
def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks, trying to cut at spaces."""
    text = " ".join(text.split())  # collapse newlines/extra spaces
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            space = text.rfind(" ", start, end)
            if space > start + size // 2:  # only cut at a space if it's not too early
                end = space
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)  # always move forward
    return chunks


# ---------- 3. Embed ----------
def embed(texts: list[str]) -> list[list[float]]:
    try:
        response = ollama.embed(model=EMBED_MODEL, input=texts)
        return response["embeddings"]
    except Exception as e:
        sys.exit(
            f"Error talking to Ollama ({e}).\n"
            f"Is Ollama running? Did you run: ollama pull {EMBED_MODEL} ?"
        )


# ---------- 4. Store in ChromaDB ----------
def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def build_index(path: Path):
    """Create (or reuse) a Chroma collection for this PDF."""
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(
        name=f"pdf_{file_hash(path)}",
        metadata={"hnsw:space": "cosine"},
    )

    if collection.count() > 0:
        print(f"Using existing index ({collection.count()} chunks).")
        return collection

    print("Reading PDF...")
    pages = load_pdf(path)

    ids, docs, metas = [], [], []
    for page_num, text in pages:
        for i, chunk in enumerate(chunk_text(text)):
            ids.append(f"p{page_num}_c{i}")
            docs.append(chunk)
            metas.append({"source": path.name, "page": page_num})

    print(f"Embedding {len(docs)} chunks (this can take a minute)...")
    batch = 32
    for i in range(0, len(docs), batch):
        collection.add(
            ids=ids[i:i + batch],
            documents=docs[i:i + batch],
            metadatas=metas[i:i + batch],
            embeddings=embed(docs[i:i + batch]),
        )
    print("Index ready.\n")
    return collection


# ---------- 5. Ask ----------
def retrieve(collection, question: str, top_k: int, max_distance: float):
    """Return only chunks that are close enough to the question."""
    q_emb = embed([question])[0]
    result = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    hits = []
    for doc, meta, dist in zip(
        result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        if dist <= max_distance:
            hits.append({"text": doc, "meta": meta, "distance": dist})
    return hits


def answer(question: str, hits: list[dict]) -> str:
    context = "\n\n".join(
        f"[{i}] (page {h['meta']['page']}) {h['text']}" for i, h in enumerate(hits, start=1)
    )
    system = (
        "You answer questions using ONLY the numbered context passages provided. "
        "Cite the passages you use like [1] or [2] after each claim. "
        f"If the context does not contain the answer, reply with exactly: {NOT_FOUND_TOKEN}"
    )
    user = f"Context:\n{context}\n\nQuestion: {question}"
    try:
        response = ollama.chat(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            options={"temperature": 0},
        )
        return response["message"]["content"].strip()
    except Exception as e:
        sys.exit(f"Error talking to Ollama ({e}). Did you run: ollama pull {CHAT_MODEL} ?")


def print_sources(hits: list[dict]) -> None:
    print("\nSources:")
    for i, h in enumerate(hits, start=1):
        snippet = h["text"][:100].replace("\n", " ")
        print(f"  [{i}] {h['meta']['source']}, page {h['meta']['page']} "
              f"(distance {h['distance']:.2f}): \"{snippet}...\"")


# ---------- 6. CLI loop ----------
def main():
    parser = argparse.ArgumentParser(description="Ask questions about a PDF.")
    parser.add_argument("pdf", type=Path, help="path to the PDF file")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--max-distance", type=float, default=MAX_DISTANCE,
                        help="chunks farther than this are ignored (lower = stricter)")
    args = parser.parse_args()

    collection = build_index(args.pdf)
    print(f"Ask questions about {args.pdf.name}. Type 'quit' to exit.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not question:
            continue
        if question.lower() in {"quit", "exit"}:
            print("Bye!")
            break

        hits = retrieve(collection, question, args.top_k, args.max_distance)

        # Edge case 1: nothing relevant retrieved -> don't even call the LLM
        if not hits:
            print("\nBot: I couldn't find anything about that in this PDF.\n")
            continue

        reply = answer(question, hits)

        # Edge case 2: the LLM itself says the context doesn't contain the answer
        if NOT_FOUND_TOKEN in reply:
            print("\nBot: The PDF doesn't seem to answer that question.\n")
            continue

        print(f"\nBot: {reply}")
        print_sources(hits)
        print()


if __name__ == "__main__":
    main()
