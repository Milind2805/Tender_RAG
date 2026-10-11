# ingest.py
# Usage:
#   python ingest.py                                  # same as before: 1000/150 -> chunks.json
#   python ingest.py --size 500 --overlap 100 --flatten --tag _500f
#   python ingest.py --size 500 --overlap 100 --flatten --tables --tag _500ft   # + header-aware table chunks
#   python ingest.py --size 500 --overlap 100 --tag _500   # -> chunks_500.json
import argparse
import json
import re
from pathlib import Path

import pymupdf

DATA_DIR = Path("data")
SKIP = {"esima12.pdf"}   # scanned; needs OCR later

ap = argparse.ArgumentParser()
ap.add_argument("--size", type=int, default=1000, help="chunk size in characters")
ap.add_argument("--overlap", type=int, default=150)
ap.add_argument("--tag", default="", help="suffix for output file, e.g. _500 -> chunks_500.json")
ap.add_argument("--flatten", action="store_true",
                help="join lines into running text (fixes tables whose labels and values are "
                     "split across many blank lines)")
ap.add_argument("--compact", action="store_true",
                help="remove blank lines and trailing spaces so chunks hold more real text")
ap.add_argument("--tables", action="store_true",
                help="also add chunks built from detected tables, as 'Column: value | Column: value' "
                     "rows, so numbers keep their column headers")
args = ap.parse_args()
CHUNK_CHARS, OVERLAP = args.size, args.overlap


def clean(text):
    if args.compact:
        text = re.sub(r"[ \t]*\n\s*", "\n", text)      # drop blank lines, trim line ends
    text = re.sub(r"[ \t]+", " ", text)
    if args.flatten:
        text = re.sub(r"\s*\n\s*", " ", text)      # every line break and blank line -> one space
        return re.sub(r" {2,}", " ", text).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text, size=CHUNK_CHARS, overlap=OVERLAP):
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            cut = max(text.rfind(" ", start, end), text.rfind("\n", start, end))   # don't cut words in half
            if cut > start + size // 2:
                end = cut
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def table_chunks(page, size=CHUNK_CHARS):
    """Return text chunks built from the tables on a page; each row keeps its column headers."""
    out = []
    try:
        found = page.find_tables()
    except Exception:
        return out
    for tab in found.tables:
        rows = [[(c or "").replace("\n", " ").strip() for c in r] for r in tab.extract()]
        rows = [r for r in rows if any(r)]
        if len(rows) < 2:
            continue
        ncols = max(len(r) for r in rows)
        lines = []
        if ncols <= 3:
            # Form-style table (label | : | value): the first row is NOT a header, so keep
            # every row as plain "label : value" text instead of attaching a wrong header.
            for r in rows:
                line = " ".join(c for c in r if c)
                if line:
                    lines.append(line)
        else:
            # Data table: the first row holds the column headers; label every cell with it.
            header = rows[0]
            for r in rows[1:]:
                pairs = []
                for j, cell in enumerate(r):
                    if not cell:
                        continue
                    label = header[j] if j < len(header) else ""
                    pairs.append(f"{label}: {cell}" if label else cell)
                if pairs:
                    lines.append(" | ".join(pairs))
        buf = ""
        for line in lines:
            if buf and len(buf) + len(line) > size:
                out.append(buf)
                buf = ""
            buf = f"{buf} || {line}" if buf else line
        if buf:
            out.append(buf)
    return out


chunks = []
for pdf in sorted(DATA_DIR.glob("*.pdf")):
    if pdf.name in SKIP:
        continue
    doc = pymupdf.open(pdf)
    for page_num, page in enumerate(doc, start=1):
        text = clean(page.get_text())
        for i, piece in enumerate(chunk_text(text)):
            chunks.append({
                "id": f"{pdf.stem}_p{page_num}_c{i}",
                "source": pdf.name,
                "page": page_num,
                "text": piece,
            })
        if args.tables:
            for i, piece in enumerate(table_chunks(page)):
                chunks.append({
                    "id": f"{pdf.stem}_p{page_num}_t{i}",
                    "source": pdf.name,
                    "page": page_num,
                    "text": "[TABLE] " + piece,
                })

out = f"chunks{args.tag}.json"
with open(out, "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False, indent=1)

print(f"Saved {len(chunks)} chunks to {out} (size={CHUNK_CHARS}, overlap={OVERLAP})")
for c in chunks[:3] + chunks[200:202]:
    print("\n---", c["id"], "---")
    print(c["text"][:300])
