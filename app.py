# app.py
from __future__ import annotations
import re, time, math
from pathlib import Path
import streamlit as st

from src.index import InvertedIndex
from src.preprocess import TextPreprocessor, make_ngrams
from src.corpus import CorpusReader

INDEX_PATH = "models/index.json"
DOCS_DIR   = "data/wiki_split_extract_2k"
EDGE_PATH  = "models/edge_index.json"

# ---------- helpers ----------
def snippet(text: str, terms: list[str], window: int = 220) -> str:
    flat = " ".join(text.split())
    pos = -1
    for t in terms:
        t = t.replace("_", " ")
        m = re.search(re.escape(t), flat, flags=re.IGNORECASE)
        if m:
            pos = m.start(); break
    if pos == -1:
        return flat[:window] + ("…" if len(flat) > window else "")
    start = max(0, pos - window // 2); end = min(len(flat), start + window)
    return ("…" if start > 0 else "") + flat[start:end] + ("…" if end < len(flat) else "")

@st.cache_resource
def load_components():
    idx = InvertedIndex.load(INDEX_PATH)
    cr  = CorpusReader(DOCS_DIR)
    tp  = TextPreprocessor(use_stemming=True, keep_digits=True)
    # edge_index.json : { "edge_index": {term: [prefixes...]}, ... }
    edge_map = InvertedIndex.load_edge_index(EDGE_PATH)  # dict[str, list[str]]
    vocab = list(edge_map.keys())
    return idx, cr, tp, vocab

def do_full_search(query: str, use_bigrams: bool, top_k: int):
    q_toks = tp.process(query)
    q_terms = q_toks + (make_ngrams(q_toks, 2) if use_bigrams else [])
    results = idx.search(query, tp, use_bigrams=use_bigrams, top_k=top_k)
    # persist in state
    ss["results"] = results
    ss["q_tokens"] = q_toks
    ss["q_terms"] = q_terms
    ss["use_bigrams"] = use_bigrams
    ss["top_k"] = top_k
    ss["query"] = query
    ss["page"] = 1
    ss["open_docs"] = set()

# ---------- UI setup ----------
st.set_page_config(page_title="IR Search", layout="wide")
st.markdown("""
<style>
.purple-hero {background:#e6dcff;height:120px;position:fixed;left:0;right:0;top:0;z-index:-1;}
.hero-title {text-align:center;margin-top:40px;font-size:32px;font-weight:700;}
.card{border-bottom:1px solid #eee;padding:12px 6px}
.muted{color:#666}.pill{background:#ede7ff;padding:2px 8px;border-radius:14px;font-size:12px}
.sugg-row{display:flex;flex-wrap:wrap;gap:8px;margin-top:6px}
.sugg{background:#f1eefc;border:1px solid #e0dbff;border-radius:14px;padding:4px 10px;cursor:pointer;}
.sugg:hover{background:#e9e5fb}
</style>
<div class="purple-hero"></div>
""", unsafe_allow_html=True)

# safety
if not Path(INDEX_PATH).exists():
    st.error(f"Index introuvable : {INDEX_PATH}"); st.stop()
if not Path(DOCS_DIR).exists():
    st.error(f"Dossier docs introuvable : {DOCS_DIR}"); st.stop()
if not Path(EDGE_PATH).exists():
    st.error(f"Edge index introuvable : {EDGE_PATH} (lance le build avec save_edge_index)."); st.stop()

idx, cr, tp, vocab = load_components()

# ---------- state init ----------
ss = st.session_state
ss.setdefault("results", [])
ss.setdefault("q_tokens", [])
ss.setdefault("q_terms", [])
ss.setdefault("use_bigrams", True)
ss.setdefault("top_k", 10)
ss.setdefault("query", "")
ss.setdefault("page", 1)
ss.setdefault("open_docs", set())

# ---------- header + search ----------
st.markdown('<div class="hero-title">Search Engine…</div>', unsafe_allow_html=True)

q = st.text_input(
    "Search",
    value=ss.get("query", "chateau gaillard"),
    label_visibility="collapsed",
    placeholder="Search for documents…"
).strip()

col1, col2, col3 = st.columns([1,1,3])
with col2:
    use_bigrams = st.toggle("Bigrams", value=ss["use_bigrams"])
with col3:
    top_k = st.slider("Top-K", 5, 50, ss["top_k"])
do_search = col1.button("Search", type="primary", use_container_width=True)

# ---------- Suggestions dynamiques ----------
# On propose des termes qui commencent par le préfixe saisi (min 3 chars),
# triés par df décroissant puis longueur, max 8 suggestions.
sugg_clicked = None
if len(q) >= 3:
    prefix = q.lower()
    # candidats qui commencent par le préfixe
    cand = [t for t in vocab if t.startswith(prefix)]
    # tri par popularité (df) puis longueur
    cand.sort(key=lambda t: (-idx.df.get(t, 0), len(t)))
    suggestions = cand[:8]

    if suggestions:
        st.caption("Suggestions :")
        # boutons cliquables inline
        c = st.container()
        with c:
            cols = st.columns(min(8, len(suggestions)))
            for i, term in enumerate(suggestions):
                if cols[i].button(term, key=f"sugg-{term}"):
                    sugg_clicked = term

# Si on clique une suggestion : on remplit la requête et on lance la recherche direct
if sugg_clicked:
    do_full_search(sugg_clicked, use_bigrams, top_k)
    st.rerun()

# ---------- on Search click ----------
if do_search and q:
    with st.spinner("Searching…"):
        time.sleep(0.15)
        do_full_search(q, use_bigrams, top_k)

# on garde query/options même sans clic
else:
    ss["query"] = q
    ss["use_bigrams"] = use_bigrams
    ss["top_k"] = top_k

# ---------- render results ----------
results = ss["results"]
page_size = 10
if not results:
    st.caption("Tape une requête et clique **Search** (ou choisis une suggestion).")
else:
    st.markdown("---")
    st.caption(f"Found **{len(results)}** result(s)")

    total_pages = max(1, math.ceil(len(results) / page_size))
    page = max(1, min(ss["page"], total_pages))
    start = (page - 1) * page_size; end = start + page_size
    page_results = results[start:end]

    for rank, (doc_id, score) in enumerate(page_results, start=start + 1):
        text = cr.read(doc_id)
        prev = snippet(text, ss["q_terms"], 240)
        st.markdown(f"""
        <div class="card">
          <div style="font-weight:600">{rank}. {doc_id}</div>
          <div class="muted">score <span class="pill">{score:.4f}</span></div>
          <div style="margin-top:6px">{prev}</div>
        </div>
        """, unsafe_allow_html=True)

        c1, _ = st.columns([1,6])
        with c1:
            key = f"view-{doc_id}"
            if st.checkbox("View content", key=key, value=(doc_id in ss["open_docs"])):
                ss["open_docs"].add(doc_id)
            else:
                ss["open_docs"].discard(doc_id)

        if doc_id in ss["open_docs"]:
            st.text_area(" ", value=text[:5000], height=240, label_visibility="collapsed")

    # pagination
    pc1, pc2, pc3, _, _ = st.columns(5)
    with pc1:
        if st.button("⟵", disabled=page <= 1):
            ss["page"] = page - 1
            st.rerun()
    with pc2:
        st.write(f"Page {page}/{total_pages}")
    with pc3:
        if st.button("⟶", disabled=page >= total_pages):
            ss["page"] = page + 1
            st.rerun()
