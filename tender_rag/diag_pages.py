"""
Diagnose why retrieval misses: are the gold pages in chunks.json, and does the answer text sit on them?

    python diag_pages.py                      # check every answerable question
    python diag_pages.py --find 180 "pre-bid" # also list which pages contain these terms
    python diag_pages.py --page-offset 1      # if chunk pages are 0-based
"""
import argparse
import json
import os
from collections import defaultdict

from eval_retrieval import norm, needles  # reuse the same text normalisation

ap = argparse.ArgumentParser()
ap.add_argument("--doc", default="TenderDocument.pdf")
ap.add_argument("--find", nargs="*", default=[])
ap.add_argument("--page-offset", type=int, default=0)
args = ap.parse_args()

chunks = json.load(open("chunks.json", encoding="utf-8"))
items = json.load(open("eval.json", encoding="utf-8"))

page_text = defaultdict(list)
for c in chunks:
    if os.path.basename(str(c["source"])).lower() == args.doc.lower():
        page_text[int(c["page"]) + args.page_offset].append(c["text"])

pages = sorted(page_text)
print(f"{args.doc}: {len(pages)} pages have text; first/last = {pages[0]}/{pages[-1]}")
print("chunks per page:", {p: len(page_text[p]) for p in pages})

blob = {p: norm(" ".join(t)) for p, t in page_text.items()}

print("\n=== Gold pages vs. chunks.json ===")
for q in items:
    if not q["answerable"]:
        continue
    gp = q["gold_pages"]
    no_text = [p for p in gp if p not in page_text]
    ns = needles(q)
    on_gold = [n for n in ns if any(n in blob.get(p, "") for p in gp)]
    anywhere = {n: [p for p in pages if n in blob[p]] for n in ns}
    print(f"\n{q['id']} [{q['type']}] gold={gp}")
    if no_text:
        print(f"  pages with NO text in chunks.json: {no_text}  <- likely scanned, needs OCR")
    if ns:
        print(f"  needles on gold pages: {len(on_gold)}/{len(ns)}")
        for n, ps in anywhere.items():
            print(f"    '{n}' found on pages: {ps[:15]}{' ...' if len(ps) > 15 else ''}")

if args.find:
    print("\n=== Term search ===")
    for term in args.find:
        t = norm(term)
        ps = [p for p in pages if t in blob[p]]
        print(f"'{term}': pages {ps[:25]}{' ...' if len(ps) > 25 else ''}")
