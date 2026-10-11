"""
Preview what PyMuPDF's table detection finds, so you can check it before re-indexing.

    python diag_tables.py                 # pages 4 and 16 of TenderDocument.pdf
    python diag_tables.py 4 16 29         # other pages
"""
import sys

import pymupdf

PDF = "data/TenderDocument.pdf"
pages = [int(a) for a in sys.argv[1:]] or [4, 16]

doc = pymupdf.open(PDF)
for n in pages:
    page = doc[n - 1]
    tabs = page.find_tables().tables
    print(f"\n=== page {n}: {len(tabs)} table(s) found ===")
    for t_idx, tab in enumerate(tabs):
        rows = tab.extract()
        print(f"-- table {t_idx}: {len(rows)} rows x {len(rows[0]) if rows else 0} cols")
        for r in rows[:6]:
            print("   ", [(c or "").replace("\n", " ")[:28] for c in r])
        if len(rows) > 6:
            print(f"    ... {len(rows) - 6} more rows")
