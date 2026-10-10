# ingest.py
# Usage:
#   python ingest.py                                  # same as before: 1000/150 -> chunks.json
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
args = ap.parse_args()
CHUNK_CHARS, OVERLAP = args.size, args.overlap


def clean(text):
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text, size=CHUNK_CHARS, overlap=OVERLAP):
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            cut = text.rfind(" ", start, end)   # don't cut words in half
            if cut > start + size // 2:
                end = cut
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


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

out = f"chunks{args.tag}.json"
with open(out, "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False, indent=1)

print(f"Saved {len(chunks)} chunks to {out} (size={CHUNK_CHARS}, overlap={OVERLAP})")
for c in chunks[:3] + chunks[200:202]:
    print("\n---", c["id"], "---")
    print(c["text"][:300])
