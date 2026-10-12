# ask.py
# Question answering over the tender documents: hybrid retrieval (query.py) + Gemini.
#
# Defaults (best eval results): index _500ft, hybrid search, top_k 10, neighbours on, query rewriting on.
# Turn a feature off with $env:RAG_NEIGHBORS="0" or $env:RAG_MULTIQ="0".
# Index choice (default: the table-aware index built with --tag _500ft):
#   $env:RAG_INDEX="default"   ; python ask.py     -> original chunks.json / embeddings.npy
#   $env:RAG_INDEX="_500ft2"   ; python ask.py     -> another tagged index
#   $env:RAG_MODE="dense"      ; python ask.py     -> dense | bm25 | hybrid (default hybrid)
import logging
import os
import re
import warnings

from dotenv import load_dotenv

load_dotenv()

# hide the Gemini SDK's "automatic function calling" notice
logging.getLogger("google_genai").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*automatic function calling.*")

_INDEX = os.environ.get("RAG_INDEX", "_500ft")
os.environ["RAG_TAG"] = "" if _INDEX == "default" else _INDEX   # read by query.py on import

from google import genai
from google.genai import types

from query import search

MODEL = "gemini-3.5-flash-lite"   # check AI Studio for a current free Flash/Flash-Lite model name
MIN_SCORE = float(os.environ.get("RAG_MIN_SCORE", "0.60"))   # refuse when the best dense score is below this
TOP_K = int(os.environ.get("RAG_TOP_K", "10"))   # chunks sent to Gemini (eval: full@5 0.57, full@10 0.86)
MODE = os.environ.get("RAG_MODE", "hybrid")
NEIGHBORS = int(os.environ.get("RAG_NEIGHBORS", "1"))   # 1 = also send the next chunk of the 3 best hits
MULTIQ = int(os.environ.get("RAG_MULTIQ", "1"))         # 1 = also search with 2 rewordings of the question

client = genai.Client()           # reads GEMINI_API_KEY from the environment

INSTRUCTIONS = (
    "You answer questions about a government tender document using ONLY the excerpts below. "
    "Cite the source like [TenderDocument.pdf p.17]. "
    "Excerpts marked [TABLE] show table rows as 'Column: value' pairs; use those column labels. "
    "Rules for numbers: when a question asks for an amount, date or period and the excerpts contain "
    "several numbers, report only a number whose label is clearly attached to it in the text, and "
    "quote that label. If the labels are missing or ambiguous, list the candidate numbers with their "
    "pages and say the labels are unclear. If different excerpts give different values for the same "
    "item, report both values with their pages. Never guess which number is meant. "
    "If the excerpts do not contain the answer, reply exactly: I can't find this in the document."
)


def retrieve(q, doc=None):
    return search(q, TOP_K, doc, mode=MODE)


def expand(hits):
    """Add the following chunk (same document and page) of the 3 best hits: facts are often split."""
    if not NEIGHBORS:
        return hits
    from query import chunks as all_chunks
    seen = {h["idx"] for h in hits}
    out = list(hits)
    for h in hits[:3]:
        j = h["idx"] + 1
        if j < len(all_chunks) and j not in seen:
            c = all_chunks[j]
            if c["source"] == h["source"] and c["page"] == h["page"]:
                out.append({"idx": j, "source": c["source"], "page": c["page"],
                            "text": c["text"], "score": 0.0})
                seen.add(j)
    return out


def rewrite_queries(q):
    """Ask Gemini for 2 alternative search queries using the document's likely wording."""
    prompt = (
        "Rewrite this question about a government tender document as 2 different short search queries "
        "that use the wording a tender document or Bid Data Sheet table would likely use "
        "(for example 'Period of Validity of Bid'). Output only the 2 queries, one per line, "
        "with no numbering or extra text.\n\nQUESTION: " + q
    )
    try:
        resp = client.models.generate_content(
            model=MODEL, contents=prompt, config=types.GenerateContentConfig(temperature=0))
        lines = [re.sub(r"^[\s\-\*\d\.\)]+", "", l).strip() for l in resp.text.splitlines()]
        return [l for l in lines if l][:2]
    except Exception:
        return []


def add_rewrites(q, hits, doc=None):
    """Add up to 4 new chunks for each reworded query (vocabulary mismatch, e.g. 'remain valid' vs 'validity')."""
    if not MULTIQ:
        return hits
    seen = {h["idx"] for h in hits}
    out = list(hits)
    for alt in rewrite_queries(q):
        for h in search(alt, 4, doc, mode=MODE):
            if h["idx"] not in seen:
                out.append(h)
                seen.add(h["idx"])
    return out


def answer(q, doc=None):
    """Return (answer_text, hits). hits are dicts with source, page, text, score."""
    hits = retrieve(q, doc)
    if not hits or max(h["score"] for h in hits) < MIN_SCORE:
        return "I can't find this in the document.", hits
    hits = expand(hits)
    hits = add_rewrites(q, hits, doc)
    context = "\n\n".join(f"[{h['source']} p.{h['page']}]\n{h['text']}" for h in hits)
    prompt = f"{INSTRUCTIONS}\n\nEXCERPTS:\n{context}\n\nQUESTION: {q}"
    resp = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0),
    )
    return resp.text, hits


if __name__ == "__main__":
    print(f"[index='{_INDEX}', mode={MODE}, top_k={TOP_K}, neighbors={NEIGHBORS}, multiq={MULTIQ}, min_score={MIN_SCORE}]")
    while True:
        q = input("\nQuestion (blank to quit): ").strip()
        if not q:
            break
        try:
            text, hits = answer(q)
        except Exception as e:      # e.g. Gemini free-tier rate limit (429) or no network
            print(f"\nCould not get an answer ({str(e)[:90]}). Wait about 15 seconds and ask again.")
            continue
        print("\n" + text)
        print("\nRetrieved:", [(h["source"], h["page"], round(h["score"], 3)) for h in hits])
