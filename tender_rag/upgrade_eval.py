"""
Upgrade eval.json in place: adds machine-checkable fields, notes, and validates the schema.
Run once from D:\\tender_rag:   python upgrade_eval.py
A backup is written to eval.backup.json first. Safe to re-run (patches are idempotent).
"""
import json
import shutil
import sys
from pathlib import Path

EVAL = Path("eval.json")
BACKUP = Path("eval.backup.json")

# Extra fields per question id. Questions not listed (q06, q11, q12) are left as they are;
# add gold_numbers / gold_keywords for them by hand.
# gold_numbers  : numeric or ID facts that must appear in the retrieved text (as strings)
# gold_keywords : short phrases that must appear (lowercase is fine, matching is normalised)
# gold_items    : for list questions, one entry per required item
PATCHES = {
    "q01": {
        "gold_numbers": ["1314", "30.09.2026"],
        "gold_keywords": ["e-tendering"],
    },
    "q02": {
        "gold_keywords": ["nagar parishad chhapiheda"],
    },
    "q03": {
        "gold_numbers": ["431005"],
        "gold_keywords": ["03 month"],
        "notes": "Two facts in one question: score amount and completion period separately.",
    },
    "q04": {
        "gold_numbers": ["4310"],
        "verify": True,
        "notes": "Gold answer lists 4310.00 and 14821 with no labels. 4310 is 1% of 431005. "
                 "Check page 16: if 14821 is also an EMD figure, add it to gold_numbers and "
                 "label both in gold_answer; if it is something else, remove it.",
    },
    "q05": {
        "gold_items": [
            "registration number or proof of application (Annexure H)",
            "notarized affidavit (Annexure B)",
            "tender document purchase receipt",
        ],
        "gold_keywords": ["annexure h", "annexure b", "receipt"],
        "verify": True,
        "notes": "Gold string is cut off on screen after 'tender docu...'. Check it against the "
                 "PDF and confirm the list is complete (the 'pur. receipt' wording too).",
    },
    "q07": {
        "gold_keywords": ["forfeiture", "earnest money"],
    },
    "q08": {
        "gold_keywords": ["more than one bid", "disqualified"],
    },
    "q09": {
        "gold_keywords": ["two working days", "technical bid"],
    },
    "q10": {
        "gold_numbers": ["3 year", "1 year"],
        "gold_keywords": ["defect liability"],
        "notes": "Check gold_pages 50 and 57 are physical PDF page numbers, not printed ones.",
    },
    "q12": {
        "gold_numbers": ["180"],
    },
    "q13": {
        "gold_numbers": ["53046410126", "SBIN0010807"],
        "notes": "Scanned page: depends on OCR quality. Check the OCR text for page 16.",
    },
    "q14": {
        "gold_keywords": ["1.5", "bid capacity"],
        "notes": "Formula page may be scanned. Check OCR on page 29.",
    },
}

REQUIRED = ("id", "question", "doc", "gold_pages", "gold_answer", "type", "answerable")


def validate(items):
    problems = []
    seen = set()
    for i, q in enumerate(items):
        qid = q.get("id", f"<index {i}>")
        for key in REQUIRED:
            if key not in q:
                problems.append(f"{qid}: missing '{key}'")
        if qid in seen:
            problems.append(f"{qid}: duplicate id")
        seen.add(qid)
        if not isinstance(q.get("gold_pages", []), list):
            problems.append(f"{qid}: gold_pages must be a list")
        if q.get("answerable") is True:
            if not q.get("gold_pages"):
                problems.append(f"{qid}: answerable but gold_pages is empty")
            if not q.get("gold_answer"):
                problems.append(f"{qid}: answerable but gold_answer is empty")
        if q.get("answerable") is False:
            if q.get("gold_pages"):
                problems.append(f"{qid}: unanswerable but gold_pages is not empty")
            if q.get("gold_answer") is not None:
                problems.append(f"{qid}: unanswerable but gold_answer is not null")
    return problems


def main():
    if not EVAL.exists():
        sys.exit("eval.json not found. Run this from D:\\tender_rag.")
    try:
        items = json.loads(EVAL.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        sys.exit(f"eval.json is not valid JSON: line {e.lineno}, col {e.colno}: {e.msg}")

    shutil.copy(EVAL, BACKUP)

    for q in items:
        q.update(PATCHES.get(q["id"], {}))

    problems = validate(items)
    EVAL.write_text(json.dumps(items, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    n_ans = sum(1 for q in items if q["answerable"])
    print(f"Wrote {len(items)} questions ({n_ans} answerable, {len(items) - n_ans} unanswerable).")
    print(f"Backup: {BACKUP}")
    if problems:
        print("\nProblems to fix by hand:")
        for p in problems:
            print("  -", p)
    flagged = [q["id"] for q in items if q.get("verify")]
    if flagged:
        print("\nVerify against the PDF:", ", ".join(flagged))


if __name__ == "__main__":
    main()
