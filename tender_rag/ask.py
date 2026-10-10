# ask.py
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from google import genai

MODEL = "gemini-3.5-flash-lite"   # check AI Studio for a current free Flash/Flash-Lite model name
MIN_SCORE = 0.60                  # tentative; tune with your eval set
TOP_K = 5
PREFIX = "Represent this sentence for searching relevant passages: "
from dotenv import load_dotenv
load_dotenv()

chunks = json.load(open("chunks.json", encoding="utf-8"))
emb = np.load("embeddings.npy")
embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
client = genai.Client()           # reads GEMINI_API_KEY from the environment


def retrieve(q):
    qv = embedder.encode([PREFIX + q], normalize_embeddings=True)[0]
    scores = emb @ qv
    idx = np.argsort(-scores)[:TOP_K]
    return [(chunks[i], float(scores[i])) for i in idx]


def answer(q):
    hits = retrieve(q)
    if hits[0][1] < MIN_SCORE:
        return "I can't find this in the document.", hits
    context = "\n\n".join(
        f"[{c['source']} p.{c['page']}]\n{c['text']}" for c, _ in hits
    )
    prompt = (
        "You answer questions about a government tender document using ONLY "
        "the excerpts below. Cite the source like [TenderDocument.pdf p.17]. "
        "If the excerpts do not contain the answer, reply exactly: "
        "I can't find this in the document.\n\n"
        f"EXCERPTS:\n{context}\n\nQUESTION: {q}"
    )
    resp = client.models.generate_content(model=MODEL, contents=prompt)
    return resp.text, hits


while True:
    q = input("\nQuestion (blank to quit): ").strip()
    if not q:
        break
    text, hits = answer(q)
    print("\n" + text)
    print("\nRetrieved:", [(c["source"], c["page"], round(s, 3)) for c, s in hits])