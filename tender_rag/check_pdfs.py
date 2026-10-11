# check_pdfs.py
import fitz  # PyMuPDF
from pathlib import Path

data_dir = Path("data")  # put your PDFs in tender_rag/data/
for pdf in sorted(data_dir.glob("*.pdf")):
    doc = fitz.open(pdf)
    chars = [len(page.get_text().strip()) for page in doc]
    scanned = sum(1 for c in chars if c < 50)
    print(f"{pdf.name}: {len(doc)} pages, "
          f"{scanned} pages with almost no text "
          f"(avg {sum(chars)//max(len(chars),1)} chars/page)")