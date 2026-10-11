"""
End-to-end check: run every question in eval.json through ask.answer() and score the answers.

    python eval_answers.py                 # uses the index chosen by RAG_INDEX (default _500ft)
    python eval_answers.py --sleep 8       # wait longer between calls if Gemini rate-limits you

Automatic status per question:
    OK     answerable: every gold number/keyword appears in the answer; unanswerable: it refused
    CHECK  only some gold items appear (or none are defined), or an unanswerable question got an
           answer saying the details are missing: read the answer yourself
    WRONG  refused an answerable question, answered an unanswerable one, no gold item found,
           or the answer contains a number known to be wrong (BAD_NUMBERS below)
The automatic check is approximate. Read the CHECK rows, and skim the OK rows too.
Writes answers_report.csv.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import ask
from eval_retrieval import needles, norm

BAD_NUMBERS = {"q04": ["2000"]}      # q04: Rs. 2000 is the bid document fee, not the EMD
REFUSAL = norm("I can't find this in the document")
# phrases that mean "the document does not give this" without using the exact refusal sentence
SOFT_NEGATIVE = [norm(p) for p in (
    "not explicitly", "not specified", "not mentioned", "not provided", "not available",
    "not stated", "not filled", "does not contain", "do not contain", "would be mentioned",
)]


def call(q, doc, tries=3, wait=15):
    for attempt in range(tries):
        try:
            return ask.answer(q, doc)
        except Exception as e:                      # rate limit or network error
            print(f"   (error: {str(e)[:80]}; retrying in {wait}s)")
            time.sleep(wait)
    return "ERROR: no answer", []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sleep", type=float, default=5.0, help="seconds between Gemini calls")
    args = ap.parse_args()

    items = json.loads(Path("eval.json").read_text(encoding="utf-8"))
    rows, counts = [], {"OK": 0, "CHECK": 0, "WRONG": 0}
    print(f"index='{ask._INDEX}' mode={ask.MODE} top_k={ask.TOP_K} neighbors={ask.NEIGHBORS} multiq={ask.MULTIQ}\n")

    for q in items:
        text, hits = call(q["question"], q.get("doc"))
        a = norm(text)
        refused = a.startswith(REFUSAL)
        ns = needles(q)
        found = sum(1 for n in ns if n in a)
        bad = [b for b in BAD_NUMBERS.get(q["id"], []) if norm(b) in a]

        if not q["answerable"]:
            if refused:
                status = "OK"
            elif any(p in a for p in SOFT_NEGATIVE):
                status = "CHECK"          # says the details are missing: read it, usually acceptable
            else:
                status = "WRONG"
        elif refused or text.startswith("ERROR") or bad:
            status = "WRONG"
        elif ns and found == len(ns):
            status = "OK"
        elif ns and found == 0:
            status = "WRONG"
        else:
            status = "CHECK"

        counts[status] += 1
        pages = sorted({h["page"] for h in hits})
        print(f"{q['id']} {status:<5} found {found}/{len(ns)} pages={pages}")
        print("    " + text.replace("\n", " ")[:160])
        rows.append({
            "id": q["id"], "status": status, "question": q["question"],
            "needles_found": f"{found}/{len(ns)}", "answer": text,
            "gold_answer": q.get("gold_answer"), "retrieved_pages": pages,
        })
        time.sleep(args.sleep)

    n = len(items)
    print(f"\nOK {counts['OK']}/{n} | CHECK {counts['CHECK']}/{n} | WRONG {counts['WRONG']}/{n}")
    with open("answers_report.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("Saved answers_report.csv (open it in Excel to read the full answers).")


if __name__ == "__main__":
    main()
