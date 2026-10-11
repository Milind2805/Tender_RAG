"""
Show exactly which chunks retrieval returns for a question, and whether a term is among them.

    python show_retrieval.py "How long must the bid remain valid?" --find 180
    python show_retrieval.py "What are the account details for online EMD payment?" --find 53046410126 --k 10
    python show_retrieval.py "..." --tag default        # use the original chunks.json index
"""
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("question")
ap.add_argument("--find", default="", help="term to look for in the retrieved chunks")
ap.add_argument("--k", type=int, default=10)
ap.add_argument("--tag", default="_500ft", help="index suffix, or 'default' for chunks.json")
ap.add_argument("--mode", default="hybrid", choices=["dense", "bm25", "hybrid"])
args = ap.parse_args()
os.environ["RAG_TAG"] = "" if args.tag == "default" else args.tag

from eval_retrieval import norm
from query import chunks, search

hits = search(args.question, args.k, mode=args.mode)
term = norm(args.find) if args.find else ""
found_at = []
for rank, h in enumerate(hits, 1):
    has = bool(term) and term in norm(h["text"])
    if has:
        found_at.append(rank)
    flag = "  <== contains '" + args.find + "'" if has else ""
    print(f"\n#{rank}  p.{h['page']}  score={h['score']:.3f}{flag}")
    print("   " + h["text"][:220].replace("\n", " "))

if term:
    print("\n" + ("Term found at rank(s): " + str(found_at) if found_at else
                  f"'{args.find}' is NOT in the top {args.k} chunks."))
    where = sorted({c["page"] for c in chunks if term in norm(c["text"])})
    print(f"'{args.find}' exists on pages: {where[:20]}")
