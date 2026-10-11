"""
Retrieval evaluation for the tender RAG project.

    python eval_retrieval.py                                # compare dense, bm25, hybrid on chunks.json
    python eval_retrieval.py --tag _500                     # same, on chunks_500.json / embeddings_500.npy
    python eval_retrieval.py --modes hybrid --k 3 5 8       # one mode, full report
    python eval_retrieval.py --modes dense bm25 hybrid --verbose   # full report for every mode

Metrics for answerable questions:
    any@k    at least one gold page appears in the top-k chunks
    all@k    every gold page appears in the top-k chunks (strict)
    frac@k   fraction of gold pages covered
    MRR      reciprocal rank of the first chunk from a gold page
    evid@k   fraction of gold numbers/keywords found in the text of the top-k chunks
    full@k   share of questions where ALL gold numbers/keywords are in the top-k chunks text
             (the strictest and most useful number: the answer text really was retrieved)
For unanswerable questions: a top-1 score threshold sweep (refuse when score < t).
"""
import argparse
import json
import os
import re
import statistics
from collections import defaultdict
from pathlib import Path

EVAL_PATH = Path("eval.json")
PAGE_OFFSET = 0


# --------------------------------------------------------------------------------------
# Retrieval adapter (uses query.search; index chosen by RAG_TAG, set in main())
# --------------------------------------------------------------------------------------
def retrieve(question, k, doc=None, mode="hybrid"):
    from query import search

    return [
        {"page": int(r["page"]) + PAGE_OFFSET, "text": r["text"], "score": r["score"]}
        for r in search(question, k, doc, mode=mode)
    ]


# --------------------------------------------------------------------------------------
# Text normalisation so "4,31,005", "Rs. 431005" and "431005" all match
# --------------------------------------------------------------------------------------
def norm(s):
    s = s.lower()
    s = re.sub(r"(?<=\d),(?=\d)", "", s)       # drop thousands/lakh separators
    s = re.sub(r"[^a-z0-9.]+", " ", s)          # keep letters, digits, dots (dates, decimals)
    s = re.sub(r"\.(?!\d)|(?<!\d)\.", " ", s)   # drop dots that are not inside numbers
    return re.sub(r"\s+", " ", s).strip()


def needles(q):
    return [norm(x) for x in q.get("gold_numbers", []) + q.get("gold_keywords", []) if x]


def evidence_fraction(q, chunks):
    ns = needles(q)
    if not ns:
        return None
    blob = norm(" ".join(c["text"] for c in chunks))
    return sum(1 for n in ns if n in blob) / len(ns)


def page_metrics(gold_pages, chunks, ks):
    gold = set(gold_pages)
    out = {}
    for k in ks:
        got = {c["page"] for c in chunks[:k]}
        out[f"any@{k}"] = float(bool(gold & got))
        out[f"all@{k}"] = float(gold <= got)
        out[f"frac@{k}"] = len(gold & got) / len(gold)
    out["rr"] = 0.0
    for rank, c in enumerate(chunks, 1):
        if c["page"] in gold:
            out["rr"] = 1.0 / rank
            break
    return out


def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else float("nan")


# --------------------------------------------------------------------------------------
def run_mode(mode, items, ks):
    kmax = max(ks)
    per_q = []
    for q in items:
        chunks = retrieve(q["question"], kmax, q.get("doc"), mode)
        top1 = chunks[0]["score"] if chunks else float("-inf")
        rec = {"id": q["id"], "type": q["type"], "answerable": q["answerable"], "top1_score": top1}
        if q["answerable"]:
            rec.update(page_metrics(q["gold_pages"], chunks, ks))
            for k in ks:
                ev = evidence_fraction(q, chunks[:k])
                rec[f"evid@{k}"] = ev
                rec[f"full@{k}"] = None if ev is None else float(ev >= 0.999)
            rec["retrieved_pages"] = [c["page"] for c in chunks[:kmax]]
        per_q.append(rec)
    return per_q


def report(mode, per_q, items, ks):
    ans = [r for r in per_q if r["answerable"]]
    kref = 5 if 5 in ks else ks[-1]
    gold_by_id = {q["id"]: q["gold_pages"] for q in items if q["answerable"]}
    top1_ans = [r["top1_score"] for r in ans]
    top1_unans = [r["top1_score"] for r in per_q if not r["answerable"]]

    print(f"\n================ mode: {mode} ================")
    print(f"{'k':>3} {'any@k':>8} {'all@k':>8} {'frac@k':>8} {'evid@k':>8} {'full@k':>8}")
    for k in ks:
        print(f"{k:>3} {mean(r[f'any@{k}'] for r in ans):8.2f} {mean(r[f'all@{k}'] for r in ans):8.2f} "
              f"{mean(r[f'frac@{k}'] for r in ans):8.2f} {mean(r[f'evid@{k}'] for r in ans):8.2f} "
              f"{mean(r[f'full@{k}'] for r in ans):8.2f}")
    print(f"MRR: {mean(r['rr'] for r in ans):.3f}")

    by_type = defaultdict(list)
    for r in ans:
        by_type[r["type"]].append(r)
    print(f"\nBy question type (k={kref})")
    print(f"{'type':<24} {'n':>3} {'any':>6} {'evid':>6} {'full':>6}")
    for t, rs in sorted(by_type.items()):
        print(f"{t:<24} {len(rs):>3} {mean(r[f'any@{kref}'] for r in rs):6.2f} "
              f"{mean(r[f'evid@{kref}'] for r in rs):6.2f} {mean(r[f'full@{kref}'] for r in rs):6.2f}")

    print(f"\nAnswer text NOT retrieved at k={kref} (full@{kref} = 0)")
    bad = [r for r in ans if r[f"full@{kref}"] == 0.0]
    for r in bad:
        print(f"  {r['id']} [{r['type']}] gold={gold_by_id[r['id']]} got={r['retrieved_pages'][:kref]} "
              f"evid={r[f'evid@{kref}']:.2f}")
    if not bad:
        print("  none")

    if top1_unans and mode != "bm25":
        print("\nRefusal sweep (refuse when top-1 dense score < t)")
        print(f"  unanswerable top-1: {[round(s, 3) for s in top1_unans]}")
        print(f"  answerable   top-1: min={min(top1_ans):.3f} median={statistics.median(top1_ans):.3f} "
              f"max={max(top1_ans):.3f}")


def summary_row(mode, per_q, ks):
    ans = [r for r in per_q if r["answerable"]]
    k5 = 5 if 5 in ks else ks[-1]
    k10 = max(ks)
    return (mode, mean(r[f"any@{k5}"] for r in ans), mean(r[f"full@{k5}"] for r in ans),
            mean(r[f"full@{k10}"] for r in ans), mean(r["rr"] for r in ans),
            mean(r[f"evid@{k5}"] for r in ans), k5, k10)


def main():
    global PAGE_OFFSET
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, nargs="+", default=[1, 3, 5, 10])
    ap.add_argument("--modes", nargs="+", default=["dense", "bm25", "hybrid"],
                    choices=["dense", "bm25", "hybrid"])
    ap.add_argument("--tag", default="", help="index suffix, e.g. _500")
    ap.add_argument("--page-offset", type=int, default=0)
    ap.add_argument("--verbose", action="store_true", help="full report for every mode")
    args = ap.parse_args()
    PAGE_OFFSET = args.page_offset
    os.environ["RAG_TAG"] = args.tag          # must be set before query.py is imported
    ks = sorted(set(args.k))

    items = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    n_ans = sum(1 for q in items if q["answerable"])
    print(f"Index: chunks{args.tag}.json | questions: {len(items)} ({n_ans} answerable, "
          f"{len(items) - n_ans} unanswerable)")

    rows = []
    for mode in args.modes:
        per_q = run_mode(mode, items, ks)
        if args.verbose or len(args.modes) == 1:
            report(mode, per_q, items, ks)
        rows.append(summary_row(mode, per_q, ks))
        Path(f"eval_results_{mode}{args.tag}.json").write_text(
            json.dumps(per_q, indent=2, default=str), encoding="utf-8")

    k5, k10 = rows[0][6], rows[0][7]
    print(f"\n=== Comparison (index '{args.tag or 'default'}') ===")
    print(f"{'mode':<8} {'any@' + str(k5):>7} {'full@' + str(k5):>8} {'full@' + str(k10):>8} "
          f"{'MRR':>6} {'evid@' + str(k5):>8}")
    for mode, a, f5, f10, mrr, ev, _, _ in rows:
        print(f"{mode:<8} {a:7.2f} {f5:8.2f} {f10:8.2f} {mrr:6.3f} {ev:8.2f}")
    print("\nPer-question results saved to eval_results_<mode><tag>.json")
    print("Run with --modes <one mode> for the full report (misses, per-type, refusal sweep).")


if __name__ == "__main__":
    main()
