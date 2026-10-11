# query.py
# Retrieval with three modes: "dense" (embeddings), "bm25" (keywords), "hybrid" (both, fused).
# Which index is used is set by the RAG_TAG environment variable ("" -> chunks.json /
# embeddings.npy, "_500" -> chunks_500.json / embeddings_500.npy).
import json
import math
import os
import re
from collections import Counter, defaultdict

import numpy as np

TAG = os.environ.get("RAG_TAG", "")
PREFIX = "Represent this sentence for searching relevant passages: "

chunks = json.load(open(f"chunks{TAG}.json", encoding="utf-8"))
_basenames = np.array([os.path.basename(str(c["source"])).lower() for c in chunks])

# ---------------------------------------------------------------- dense (lazy load)
_model = None
_emb = None


def _dense_scores(question):
    global _model, _emb
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        _emb = np.load(f"embeddings{TAG}.npy")
    qv = _model.encode([PREFIX + question], normalize_embeddings=True)[0]
    return _emb @ qv


# ---------------------------------------------------------------- BM25 (no extra packages)
_STOP = {"the", "a", "an", "of", "and", "or", "to", "in", "for", "is", "are", "be", "by", "on",
         "at", "as", "with", "what", "which", "that", "this", "shall", "if", "it", "from", "any",
         "does", "do", "how", "who", "when", "where"}


def tokenize(text):
    text = re.sub(r"(?<=\d),(?=\d)", "", text.lower())        # 4,31,005 -> 431005
    out = []
    for t in re.findall(r"[a-z0-9]+(?:\.[0-9]+)*", text):
        if t in _STOP:
            continue
        out.append(t)
        if "." in t:                                           # 4310.00 also matches 4310
            out.extend(p for p in t.split(".") if p)
    return out


_K1, _B = 1.5, 0.75
_inv = defaultdict(list)          # term -> [(chunk_index, term_frequency)]
_dl = np.zeros(len(chunks))
for _i, _c in enumerate(chunks):
    _tf = Counter(tokenize(_c["text"]))
    _dl[_i] = sum(_tf.values())
    for _t, _n in _tf.items():
        _inv[_t].append((_i, _n))
_avgdl = _dl.mean() if len(chunks) else 1.0


def _bm25_scores(question):
    scores = np.zeros(len(chunks))
    n = len(chunks)
    for t in set(tokenize(question)):
        postings = _inv.get(t)
        if not postings:
            continue
        idf = math.log(1 + (n - len(postings) + 0.5) / (len(postings) + 0.5))
        for i, tf in postings:
            scores[i] += idf * tf * (_K1 + 1) / (tf + _K1 * (1 - _B + _B * _dl[i] / _avgdl))
    return scores


# ---------------------------------------------------------------- search
def search(question, k=5, doc=None, mode="hybrid", pool=50, rrf_k=60):
    """Top-k chunks, best first: dicts with source, page, text, score, bm25, rrf.

    mode: "dense" | "bm25" | "hybrid" (reciprocal rank fusion of dense and bm25 rankings).
    score: dense cosine similarity of the chunk (BM25 raw score in bm25 mode); use it for
           the refusal threshold.
    doc:   optional file name; only chunks from that document are considered.
    """
    allowed = (_basenames == os.path.basename(doc).lower()) if doc else np.ones(len(chunks), bool)

    dense = _dense_scores(question) if mode in ("dense", "hybrid") else None
    bm = _bm25_scores(question) if mode in ("bm25", "hybrid") else None

    def ranking(scores, need_positive=False):
        s = np.where(allowed, scores, -np.inf)
        order = np.argsort(-s)[:pool]
        return [int(i) for i in order if np.isfinite(s[i]) and (not need_positive or s[i] > 0)]

    if mode == "dense":
        order = ranking(dense)[:k]
        fused = {i: float(dense[i]) for i in order}
    elif mode == "bm25":
        order = ranking(bm, need_positive=True)[:k]
        fused = {i: float(bm[i]) for i in order}
    else:
        fused = defaultdict(float)
        for lst in (ranking(dense), ranking(bm, need_positive=True)):
            for rank, i in enumerate(lst, start=1):
                fused[i] += 1.0 / (rrf_k + rank)
        order = sorted(fused, key=fused.get, reverse=True)[:k]

    return [
        {
            "idx": i,
            "source": chunks[i]["source"],
            "page": chunks[i]["page"],
            "text": chunks[i]["text"],
            "score": float(dense[i]) if dense is not None else float(bm[i]),
            "bm25": float(bm[i]) if bm is not None else None,
            "rrf": float(fused[i]),
        }
        for i in order
    ]


if __name__ == "__main__":
    mode = os.environ.get("RAG_MODE", "hybrid")
    print(f"index tag='{TAG}', mode={mode}")
    while True:
        q = input("\nQuestion (blank to quit): ").strip()
        if not q:
            break
        for rank, c in enumerate(search(q, 5, mode=mode), start=1):
            print(f"\n#{rank}  score={c['score']:.3f}  {c['source']} p.{c['page']}")
            print(c["text"][:250].replace("\n", " "))
