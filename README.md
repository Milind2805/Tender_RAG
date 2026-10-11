# Tender RAG

Ask questions about construction tender documents and get answers with page citations. Built on real e-procurement tender PDFs (including scanned pages), using free tools only: local embeddings and the Gemini free tier.

The focus of this project is **evaluation**: I wrote a question set first, then measured how each change to retrieval and prompting moved the answers, instead of judging by a few demo questions.

## Example

```
Q: How long must the bid remain valid after the bid submission closing date?
A: 180 days (Bid Data Sheet, TenderDocument.pdf p.16)
```

## How it works

1. **Ingest** (`ingest.py`): extract text per page; table-aware chunking at about 500 characters; OCR/fallback for scanned pages [state what you used]
2. **Index:** local embeddings ([model name]) plus keyword search, stored in [format]
3. **Retrieve** (`query.py`): hybrid search (keyword + embedding), top 10 chunks, plus 1 neighbouring chunk on each side for context
4. **Query rewriting:** the question is rewritten into extra query variants and results are merged
5. **Answer** (`ask.py`): Gemini (free tier) answers only from the retrieved text and cites the file and page; when nothing relevant is retrieved it should say so

## Evaluation

I wrote 16 questions on `TenderDocument.pdf` (`eval.json`) covering simple facts, numbers with units, lists, conditions, answers spread over two sections, rephrased questions, a lookup on a scanned page, and 2 questions the document cannot answer. Answers are auto-scored against gold keywords and pages (`eval_answers.py`); anything the scorer cannot decide is marked CHECK.

| Configuration | OK | CHECK | WRONG |
|---|---|---|---|
| Hybrid search, top-5 | 8 | 3 | 5 |
| top-10 | 10 | 5 | 1 |
| top-10 + neighbouring chunks | 12 | 3 | 1 |
| top-10 + neighbours + query rewriting | **13** | 3 | **0** |

What each change fixed:
- **Table-aware chunking:** [one line on what it fixed, e.g. numbers in tables being split from their labels]
- **More chunks and neighbours:** answers that sit across a chunk boundary
- **Query rewriting:** questions phrased differently from the tender's wording

## Limitations (read before trusting the numbers)

- **Small, shared set.** The 16 questions were used both to pick settings and to report results, so the final score is optimistic. A fresh set on a different tender is still to do.
- **One main document.** Results are on a single tender; other formats may behave differently.
- **Scanned pages** were the main retrieval failures in my first retrieval eval; OCR quality limits those answers. [update with current status]
- **Cross-references.** Tenders often say "as specified in the Bid Data Sheet", so the answer lives on a different page from the clause that asks the question. The system sometimes lists the pointer as a separate answer instead of resolving it. [update if fixed]
- **Conflicting values.** The tender states two different EMD amounts on different pages (p.4 and p.16); the system should show both with citations, not pick one.
- **Auto-scoring** is keyword-based, so CHECK rows were read by hand.

## Run it

```bash
pip install -r requirements.txt
python ingest.py                  # build the index from data/
export GEMINI_API_KEY=...         # PowerShell: $env:GEMINI_API_KEY="..."
python ask.py                     # ask questions in the terminal
python eval_answers.py            # run the 16-question evaluation
```

Settings used for the results above: `index=_500ft, mode=hybrid, top_k=10, neighbors=1, multiq=1, min_score=0.6`.

## Stack

Python · sentence embeddings · hybrid retrieval · Gemini API (free tier) · Streamlit [if you deploy it]

## What I'd do next

- Evaluate on a second, unseen tender with a held-out question set
- Resolve "see Bid Data Sheet" cross-references explicitly
- Report answer quality separately for scanned and born-digital pages****
