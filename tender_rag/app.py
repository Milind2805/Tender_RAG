# app.py
# Streamlit demo for Tender RAG: wraps ask.answer() (hybrid retrieval + Gemini) in a small web UI.
# Run locally from the repo folder:  streamlit run app.py
import os

import streamlit as st

st.set_page_config(page_title="Tender RAG", page_icon="📄")

# On Streamlit Cloud the key comes from the app's Secrets; locally ask.py reads it from .env.
# It must be in the environment BEFORE ask.py is imported (ask.py creates the Gemini client on import).
try:
    if "GEMINI_API_KEY" in st.secrets:
        os.environ["GEMINI_API_KEY"] = st.secrets["AQ.Ab8RN6KAwrVj6BqTXoQdjtFgQ4eJuk3gNEcXq5HxfylsVp5Fng"]
except Exception:
    pass  # no secrets file when running locally

MAX_QUESTIONS = 10  # per browser session, to protect the shared free Gemini quota
EXAMPLES = [
    "How long must the bid remain valid after the bid submission closing date?",
    "What is the period of validity of the submitted tender proposal?",
    "What is the amount of earnest money deposit (EMD)?",
]


@st.cache_resource(show_spinner="Loading the search index (first run only)...")
def load_pipeline():
    import ask  # loads chunks, builds the BM25 index, creates the Gemini client

    ask.search("warm up", 1, mode="hybrid")  # loads the embedding model once
    return ask


with st.sidebar:
    st.header("About")
    st.write(
        "Question answering over a public government tender document. "
        "Hybrid retrieval (BM25 + embeddings) feeds Gemini, which must cite the page."
    )
    st.write("Evaluated on a hand-written 16-question set; see the README for results and limitations.")
    st.write("Code: github.com/Milind2805/Tender_RAG")
    st.caption("Free demo: the first load can take a minute, and the Gemini free tier can be busy.")

st.title("📄 Tender RAG")
st.caption("Ask about the tender document. Answers can be wrong: always check the cited page.")

try:
    ask = load_pipeline()
except Exception:
    st.error("The app could not start. Check that GEMINI_API_KEY is set in the app's Secrets.")
    st.stop()

if "n_asked" not in st.session_state:
    st.session_state.n_asked = 0

question = None
cols = st.columns(len(EXAMPLES))
for col, ex in zip(cols, EXAMPLES):
    if col.button(ex, use_container_width=True):
        question = ex
typed = st.chat_input("Ask a question about the tender")
if typed:
    question = typed

if question:
    if st.session_state.n_asked >= MAX_QUESTIONS:
        st.warning(f"Demo limit of {MAX_QUESTIONS} questions per session reached. Refresh the page to start again.")
    else:
        st.session_state.n_asked += 1
        st.markdown(f"**Q:** {question}")
        with st.spinner("Searching the tender..."):
            try:
                text, hits = ask.answer(question)
            except Exception:  # e.g. Gemini free-tier rate limit or a retired model name
                st.session_state.n_asked -= 1
                st.error("Could not get an answer right now (the free Gemini quota may be busy). "
                         "Wait about 15 seconds and try again.")
                st.stop()
        st.markdown(text)

        seen, shown = set(), []
        for h in hits:
            if h["idx"] not in seen:
                seen.add(h["idx"])
                shown.append(h)
        with st.expander(f"Retrieved passages ({len(shown)})"):
            for h in shown:
                st.markdown(f"**{h['source']} p.{h['page']}**")
                st.text(h["text"][:600])
        st.caption(f"Questions used this session: {st.session_state.n_asked}/{MAX_QUESTIONS}")
