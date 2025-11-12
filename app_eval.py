# app_eval.py
from __future__ import annotations
import json, time, math, os, hashlib, datetime
from typing import Dict, List, Tuple
from collections import defaultdict, Counter

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from src.search import SearchEngine
from src.metrics import precision_at_k, recall_at_k, hit_at_1, mrr, ndcg_at_k
from src.index import InvertedIndex
from src.preprocess import TextPreprocessor
from src.suggest import Autocomplete

# ============ Style global (cartes façon dashboard) ============
st.set_page_config(page_title="IR Evaluation Dashboard", layout="wide")

# CSS léger pour des cartes propres (compatible dark/light)
st.markdown("""
<style>
/* Conteneur carte */
.card {
  border-radius: 14px;
  padding: 18px 20px;
  margin: 8px 0 12px 0;
  box-shadow: 0 8px 18px rgba(0,0,0,.12);
  background: linear-gradient(135deg, rgba(20,32,44,.75), rgba(20,32,44,.55));
  border: 1px solid rgba(255,255,255,.08);
}
/* Carte claire (accent) */
.card-accent {
  background: linear-gradient(135deg, #0e3a5b, #0f4c75);
  border: 1px solid rgba(255,255,255,.08);
  color: #fff;
}
.card h3, .card h4, .card h5, .kpi-title { margin: 0; }
.kpi-value { font-size: 32px; font-weight: 800; margin-top: 6px; }
.kpi-sub { opacity: .85; font-size: 13px; }
.badge {
  display: inline-block;
  padding: 4px 10px;
  border-radius: 999px;
  font-size: 12px;
  border: 1px solid rgba(255,255,255,.15);
  background: rgba(255,255,255,.06);
}
.sep { height: 8px; }
</style>
""", unsafe_allow_html=True)

# ---------- helpers ----------
def load_jsonl_queries(path: str) -> List[Tuple[str, str, str]]:
    """Retourne [(qid, query, answer_doc)] à partir d'un jsonl {Answer file, Queries}."""
    triples, qid = [], 1
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            answer = obj.get("Answer file") or obj.get("answer") or obj.get("answer_file")
            queries = obj.get("Queries") or obj.get("queries") or []
            for q in queries:
                triples.append((f"Q{qid}", q, answer))
                qid += 1
    return triples

def to_qrels(triples: List[Tuple[str, str, str]]) -> Tuple[Dict[str, Dict[str, int]], Dict[str, str]]:
    """(qrels, queries_by_qid) à partir de (qid, query, answer_doc). qrels binaire."""
    qrels: Dict[str, Dict[str, int]] = defaultdict(dict)
    qtxt: Dict[str, str] = {}
    for qid, q, ans in triples:
        qtxt[qid] = q
        if ans:
            qrels[qid][ans] = 1
    return qrels, qtxt

def rank_of(doc_id: str, run: List[str]) -> int | None:
    try:
        return run.index(doc_id) + 1
    except ValueError:
        return None

def file_info(path: str) -> dict:
    if not os.path.exists(path):
        return {"exists": False}
    stat = os.stat(path)
    size_kb = round(stat.st_size / 1024, 1)
    with open(path, "rb") as f:
        h = hashlib.md5(f.read()).hexdigest()[:10]
    return {
        "exists": True,
        "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime)),
        "size_kb": size_kb,
        "md5_10": h,
    }

# ---------- build utils ----------
def build_all_artifacts(data_dir: str, models_dir: str, use_bigrams: bool = True):
    """Build index + edge n-grams and save to models_dir (écrase les anciens)."""
    preproc = TextPreprocessor(use_stemming=True, keep_digits=True)
    idx = InvertedIndex()
    st.toast("Construction de l’index…")
    idx.build(data_dir, preproc, use_bigrams=use_bigrams)
    idx.save(models_dir)                 # (ré)écrit models/index.json
    st.toast("Génération des edge n-grams…")
    idx.save_edge_index(models_dir)      # (ré)écrit models/edge_index.json
    st.toast("Index + edge n-grams sauvegardés", icon="✅")
    return idx, preproc

def ensure_tfidf_loaded(engine: SearchEngine, tf_scheme: str = "log", force: bool = False):
    """Construit TF-IDF en mémoire si absent, ou si force=True."""
    if force or not getattr(engine.index, "doc_tfidf", None):
        engine.index.build_tfidf(tf_scheme=tf_scheme)

# --- Petite notif synthèse après build ---
def notify_build_result(before: dict, after: dict, path: str = "models/index.json"):
    if not after.get("exists"):
        st.toast(f"❌ {path} introuvable après le build.")
        return
    if not before.get("exists"):
        st.toast(f"✅ {path} créé — taille: {after.get('size_kb', 0):.1f} KB")
        return
    if before.get("md5_10") == after.get("md5_10"):
        st.toast(f"ℹ️ Aucun changement — taille: {after.get('size_kb', 0):.1f} KB")
        return
    # modifié
    delta = None
    if isinstance(before.get("size_kb"), (int, float)) and isinstance(after.get("size_kb"), (int, float)):
        delta = after["size_kb"] - before["size_kb"]
    fleche = "↑" if (delta and delta > 0) else ("↓" if (delta and delta < 0) else "→")
    delta_txt = f"{fleche} {delta:+.1f} KB" if delta is not None else "taille modifiée"
    st.toast(f"✅ {path} mis à jour — {delta_txt}, nouvelle taille: {after.get('size_kb', 0):.1f} KB")

# ============ Header =============
st.title("IR Evaluation Dashboard")

# Sidebar
with st.sidebar:
    models_dir = "models"
    data_dir = "data/wiki_split_extract_2k"
    jsonl_path = "data/requetes.jsonl"
    st.subheader("Création de l'index inversé et edge n-grams")
    use_bigrams_build = st.checkbox("Use bigrams (build)", value=True, help="Sert lors du build de l’index.")
    build_now = st.button("Build/Rebuild index + edge n-grams")
    force_tfidf = st.checkbox("Forcer (re)build TF-IDF", value=False)
    build_tfidf_now = st.button("Build/Rebuild TF-IDF")

    st.markdown("---")
    st.subheader("Méthode de recherche")
    method = st.radio("Moteur", ["BM25", "TF-IDF Cosine", "Hybrid RRF", "Hybrid Interp"], index=0)
    use_bigrams = st.checkbox("Use bigrams (search)", value=True)
    top_k = st.number_input("top_k affiché", 1, 100, 10)

    st.markdown("**Paramètres**")
    k1 = st.slider("BM25: k1 (info)", 0.5, 2.0, 1.2, 0.1)
    b = st.slider("BM25: b (info)", 0.0, 1.0, 0.75, 0.05)
    tf_scheme = st.selectbox("Cosine TF scheme", ["log", "raw"], index=0)
    k_lex = st.number_input("Hybrid: k_lex (BM25)", 10, 2000, 200, 10)
    k_vec = st.number_input("Hybrid: k_vec (Cosine)", 10, 2000, 200, 10)
    rrf_k = st.number_input("RRF: rrf_k", 1, 200, 60, 1)
    alpha = st.slider("Interp: alpha (poids BM25)", 0.0, 1.0, 0.6, 0.05)

    st.markdown("---")
    st.subheader("Évaluation")
    k_eval = st.number_input("k pour @k", 1, 100, 10)
    run_eval = st.button("Lancer l'évaluation")

# Cache simple de l'engine
@st.cache_resource(show_spinner=False)
def get_engine(models_dir: str) -> SearchEngine:
    eng = SearchEngine(models_dir=models_dir)
    eng.load()
    return eng

engine = get_engine(models_dir)

# Actions build/rebuild
if build_now:
    try:
        before = file_info(f"{models_dir}/index.json")
        idx, _ = build_all_artifacts(data_dir, models_dir, use_bigrams=use_bigrams_build)
        engine = SearchEngine(models_dir=models_dir)
        engine.load()
        after = file_info(f"{models_dir}/index.json")
        notify_build_result(before, after, path=f"{models_dir}/index.json")
    except Exception as e:
        st.error(f"Échec build: {e}")

if build_tfidf_now:
    try:
        ensure_tfidf_loaded(engine, tf_scheme=tf_scheme, force=force_tfidf)
        st.toast("TF-IDF (in-memory)", icon="✅")
    except Exception as e:
        st.error(f"Échec TF-IDF: {e}")

# Onglets
tab_eval, tab_compare = st.tabs(["Evaluation", "Comparaison"])



# --- Tab Eval ---
with tab_eval:
    if run_eval:
        triples = load_jsonl_queries(jsonl_path)
        qrels, queries = to_qrels(triples)
        n = len(queries)
        st.write(f"**Évaluation sur {n} requêtes**")

        if method in ("TF-IDF Cosine", "Hybrid RRF", "Hybrid Interp"):
            ensure_tfidf_loaded(engine, tf_scheme=tf_scheme, force=False)

        rows = []
        sum_p = sum_r = sum_hit1 = sum_mrr = sum_ndcg = 0.0
        rank_hist = Counter()
        # exemple : Q1: "langue roumain" 
        for qid, text in queries.items():
            cut = max(k_eval, 50)
            if method == "BM25":
                res = engine.search(text, top_k=cut)
            elif method == "TF-IDF Cosine":
                res = engine.index.search_cosine(text, engine.preproc, use_bigrams=use_bigrams, top_k=cut, tf_scheme=tf_scheme)
            elif method == "Hybrid RRF":
                res = engine.index.search_hybrid_rrf(text, engine.preproc, use_bigrams=use_bigrams, k_lex=k_lex, k_vec=k_vec, top_k=cut, rrf_k=rrf_k)
            else:
                res = engine.index.search_hybrid_interp(text, engine.preproc, use_bigrams=use_bigrams, k_lex=k_lex, k_vec=k_vec, top_k=cut, alpha=alpha)

            run = [d for d, _ in res]
            # Q1: doc.. 
            rels = qrels.get(qid, {})

            p = precision_at_k(run, rels, k=k_eval)
            r = recall_at_k(run, rels, k=k_eval)
            a1 = hit_at_1(run, rels)
            mr = mrr(run, rels)
            nd = ndcg_at_k(run, rels, k=k_eval)

            sum_p += p; sum_r += r; sum_hit1 += a1; sum_mrr += mr; sum_ndcg += nd

            answer_doc = next(iter(rels.keys())) if rels else None
            rank = rank_of(answer_doc, run) if answer_doc else None
            bucket = ("miss" if rank is None else (f">{k_eval}" if rank > k_eval else rank))
            rank_hist[bucket] += 1

            rows.append({
                "qid": qid, "query": text, "answer": answer_doc,
                f"P@{k_eval}": round(p, 3),
                f"R@{k_eval}": round(r, 3),
                "Hit@1": round(a1, 3),
                "MRR": round(mr, 3),
                f"nDCG@{k_eval}": round(nd, 3),
                "rank": rank
            })

        st.subheader("Scores moyens (macro)")
        st.write({
            f"P@{k_eval}": round(sum_p/n, 3),
            f"R@{k_eval}": round(sum_r/n, 3),
            "Hit@1": round(sum_hit1/n, 3),
            "MRR": round(sum_mrr/n, 3),
            f"nDCG@{k_eval}": round(sum_ndcg/n, 3),
        })

        st.subheader("Distribution des rangs")
        keys_ordered = [i for i in range(1, k_eval+1)] + [f">{k_eval}", "miss"]
        dist = {str(k): rank_hist.get(k, 0) for k in keys_ordered}
        st.bar_chart(dist)

        st.subheader("Détails par requête")
        st.dataframe(rows, use_container_width=True)

        st.download_button(
            "⬇️ Export CSV",
            data="\n".join([",".join(map(str, r.values())) for r in rows]),
            file_name="eval_results.csv",
            mime="text/csv"
        )
    else:
        st.info("Configure et clique 🚀 dans la sidebar pour évaluer.")

# ------------------ Helpers UI pour la comparaison ------------------
def donut_gauge(title: str, value: float, maxv: float = 1.0):
    """Jauge circulaire (donut) Plotly."""
    v = max(0.0, min(value, maxv))
    fig = go.Figure(data=[go.Pie(
        values=[v, maxv - v],
        labels=['', ''],
        hole=.75,
        textinfo='none'
    )])
    # couleurs par défaut (s’adapte au thème)
    fig.update_traces(marker=dict(colors=["#FFB703", "#334155"]), hoverinfo='skip')
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        annotations=[dict(text=f"<b>{value:.3f}</b>", x=0.5, y=0.5, font_size=20, showarrow=False),
                     dict(text=title, x=0.5, y=0.15, font_size=12, showarrow=False)]
    )
    return fig

def kpi_card(title: str, value: float | str, sub: str = "", accent: bool = False):
    klass = "card card-accent" if accent else "card"
    st.markdown(f"""
    <div class="{klass}">
      <div class="kpi-title">{title}</div>
      <div class="kpi-value">{value}</div>
      <div class="kpi-sub">{sub}</div>
    </div>
    """, unsafe_allow_html=True)

# --- Tab Comparison (NOUVEAU DESIGN) ---
with tab_compare:
    st.markdown("### Comparaison multi-méthodes")

    # Choix méthodes + paramètres
    colx, coly = st.columns([1,1])
    with colx:
        do_bm25 = st.checkbox("BM25", value=True)
        do_cos = st.checkbox("TF-IDF Cosine (log)", value=True)
        do_rrf = st.checkbox("Hybrid RRF", value=True)
        do_interp = st.checkbox("Hybrid Interp", value=True)
        metric_for_gauge = st.selectbox("Jauge principale", ["P@k", "R@k", "Hit@1", "MRR", "nDCG@k"], index=0)
    with coly:
        k_eval_cmp = st.number_input("k pour @k (comparaison)", 1, 100, 10, key="k_eval_cmp")
        k_lex_cmp = st.number_input("k_lex (RRF/Interp)", 10, 2000, 200, 10, key="klexcmp")
        k_vec_cmp = st.number_input("k_vec (RRF/Interp)", 10, 2000, 200, 10, key="kveccmp")
        rrf_k_cmp = st.number_input("rrf_k (RRF)", 1, 200, 60, 1, key="rrfkc")
        alpha_cmp = st.slider("alpha (Interp)", 0.0, 1.0, 0.6, 0.05, key="alphacmp")

    if st.button("▶️ Lancer la comparaison"):
        triples = load_jsonl_queries(jsonl_path)
        qrels, queries = to_qrels(triples)
        methods = []
        if do_bm25: methods.append(("BM25", None))
        if do_cos: methods.append(("TF-IDF Cosine", "log"))
        if do_rrf: methods.append(("Hybrid RRF", None))
        if do_interp: methods.append(("Hybrid Interp", None))

        if any(m[0] in ("TF-IDF Cosine","Hybrid RRF","Hybrid Interp") for m in methods):
            ensure_tfidf_loaded(engine, tf_scheme="log", force=False)

        results = []
        for name, _ in methods:
            n = len(queries)
            sum_p = sum_r = sum_hit1 = sum_mrr = sum_ndcg = 0.0
            t0 = time.time()
            for qid, text in queries.items():
                cut = max(k_eval_cmp, 50)
                if name == "BM25":
                    res = engine.search(text, top_k=cut)
                elif name == "TF-IDF Cosine":
                    res = engine.index.search_cosine(text, engine.preproc, use_bigrams=use_bigrams, top_k=cut, tf_scheme="log")
                elif name == "Hybrid RRF":
                    res = engine.index.search_hybrid_rrf(text, engine.preproc, use_bigrams=use_bigrams, k_lex=k_lex_cmp, k_vec=k_vec_cmp, top_k=cut, rrf_k=rrf_k_cmp)
                else:
                    res = engine.index.search_hybrid_interp(text, engine.preproc, use_bigrams=use_bigrams, k_lex=k_lex_cmp, k_vec=k_vec_cmp, top_k=cut, alpha=alpha_cmp)

                run = [d for d, _ in res]
                rels = qrels.get(qid, {})
                sum_p     += precision_at_k(run, rels, k=k_eval_cmp)
                sum_r     += recall_at_k(run, rels, k=k_eval_cmp)
                sum_hit1  += hit_at_1(run, rels)
                sum_mrr   += mrr(run, rels)
                sum_ndcg  += ndcg_at_k(run, rels, k=k_eval_cmp)
            dt = (time.time() - t0) * 1000
            results.append({
                "Méthode": name,
                "P@k": round(sum_p/n, 3),
                "R@k": round(sum_r/n, 3),
                "Hit@1": round(sum_hit1/n, 3),
                "MRR": round(sum_mrr/n, 3),
                "nDCG@k": round(sum_ndcg/n, 3),
                "Latence (ms)": int(dt)
            })

        if not results:
            st.warning("Sélectionne au moins une méthode.")
        else:
            df = pd.DataFrame(results)

            # ======== Bandeau KPI (cartes) ========
            best_row = df.loc[df["P@k"].idxmax()]
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                kpi_card("Meilleure P@k", f"{best_row['P@k']:.3f}",
                         sub=f"🏆 {best_row['Méthode']}", accent=True)
            with c2:
                kpi_card("Meilleure nDCG@k", f"{df.loc[df['nDCG@k'].idxmax(), 'nDCG@k']:.3f}",
                         sub=f"🏆 {df.loc[df['nDCG@k'].idxmax(), 'Méthode']}")
            with c3:
                kpi_card("Meilleure MRR", f"{df.loc[df['MRR'].idxmax(), 'MRR']:.3f}",
                         sub=f"🏆 {df.loc[df['MRR'].idxmax(), 'Méthode']}")
            with c4:
                kpi_card("Latence min", f"{df['Latence (ms)'].min():.0f} ms",
                         sub=f"⚡ {df.loc[df['Latence (ms)'].idxmin(), 'Méthode']}")

            # ======== Jauges (donut) ========
            g1, g2, g3 = st.columns(3)
            metric_map = {"P@k": "P@k", "R@k":"R@k", "Hit@1":"Hit@1", "MRR":"MRR", "nDCG@k":"nDCG@k"}
            key = metric_map[metric_for_gauge]
            # on met la meilleure méthode au centre
            row_g = df.loc[df[key].idxmax()]
            with g1:
                st.plotly_chart(donut_gauge(f"{metric_for_gauge} (best)", row_g[key], 1.0), use_container_width=True)
            with g2:
                # moyenne globale
                st.plotly_chart(donut_gauge(f"{metric_for_gauge} (moy.)", df[key].mean(), 1.0), use_container_width=True)
            with g3:
                # deuxième (si dispo)
                if len(df) > 1:
                    second = df.sort_values(key, ascending=False).iloc[1]
                    st.plotly_chart(donut_gauge(f"{metric_for_gauge} (2ᵉ)", second[key], 1.0), use_container_width=True)
                else:
                    st.plotly_chart(donut_gauge(f"{metric_for_gauge}", row_g[key], 1.0), use_container_width=True)

            st.markdown('<div class="sep"></div>', unsafe_allow_html=True)

            # ======== Barres par métrique ========
            st.subheader("Scores par méthode")
            colm1, colm2 = st.columns(2)
            with colm1:
                st.bar_chart(df.set_index("Méthode")["P@k"])
                st.bar_chart(df.set_index("Méthode")["Hit@1"])
            with colm2:
                st.bar_chart(df.set_index("Méthode")["R@k"])
                st.bar_chart(df.set_index("Méthode")["MRR"])

            st.bar_chart(df.set_index("Méthode")["nDCG@k"])

            # ======== Tableau récap ========
            st.subheader("Tableau comparatif")
            st.dataframe(df, use_container_width=True)
