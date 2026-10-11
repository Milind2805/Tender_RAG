# build_index.py
# Usage: python build_index.py            -> embeddings.npy   (from chunks.json)
#        python build_index.py --tag _500 -> embeddings_500.npy (from chunks_500.json)
import argparse
import json

import numpy as np
from sentence_transformers import SentenceTransformer

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="")
args = ap.parse_args()

chunks = json.load(open(f"chunks{args.tag}.json", encoding="utf-8"))
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

texts = [c["text"] for c in chunks]
emb = model.encode(texts, batch_size=32, normalize_embeddings=True,
                   show_progress_bar=True)
np.save(f"embeddings{args.tag}.npy", emb)
print("Saved", emb.shape, f"to embeddings{args.tag}.npy")
