"""Put your documents into ChromaDB.

    python ingest.py            # read everything in docs/ (.md, .txt, .pdf)
    python ingest.py --reset    # wipe the database first, then re-read docs/

Re-running is safe: the same chunk is updated, not duplicated.
"""
import sys
from pathlib import Path

import tools

DOCS_DIR = Path("docs")
CHUNK_SIZE = 500   # characters per chunk
OVERLAP = 100      # shared characters between neighbours, so ideas aren't cut in half


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> list[str]:
    text = " ".join(text.split())
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        if start + size >= len(text):
            break
        start += size - overlap
    return chunks


def read_pages(path: Path):
    """Yield (page_label, text). Markdown/text files are one 'page'."""
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        for n, page in enumerate(PdfReader(str(path)).pages, 1):
            yield f"page {n}", page.extract_text() or ""
    else:
        yield "", path.read_text(encoding="utf-8")


def main():
    if "--reset" in sys.argv:
        tools.reset_collection()
        print("Database wiped.")

    files = [p for p in sorted(DOCS_DIR.glob("*")) if p.suffix.lower() in {".md", ".txt", ".pdf"}]
    if not files:
        print("No .md, .txt or .pdf files found in docs/. Add some and run again.")
        return

    col = tools.get_collection()
    total = 0
    for path in files:
        ids, texts, metas = [], [], []
        for page_label, text in read_pages(path):
            for i, chunk in enumerate(chunk_text(text), 1):
                where = f"{page_label}, chunk {i}" if page_label else f"chunk {i}"
                ids.append(f"{path.name}-{page_label or 'p'}-{i}")
                texts.append(chunk)
                metas.append({"source": f"docs/{path.name}", "where": where})
        if ids:
            col.upsert(ids=ids, documents=texts, metadatas=metas)
        total += len(ids)
        print(f"  {path.name}: {len(ids)} chunks")
    print(f"Done. {total} chunks stored, {col.count()} in the database.")


if __name__ == "__main__":
    main()
