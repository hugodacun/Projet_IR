from __future__ import annotations
import json, time, math, os, hashlib, datetime
from typing import Dict, List, Tuple
from collections import defaultdict, Counter
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px   

from src.search import SearchEngine
from src.metrics import precision_at_k, recall_at_k, hit_at_1, mrr, ndcg_at_k
from src.index import InvertedIndex
from src.preprocess import TextPreprocessor
from src.suggest import Autocomplete

# ============ Style global de l'interface (cartes façon dashboard) ============
st.set_page_config(page_title="IR Evaluation Dashboard", layout="wide")

# CSS 
st.markdown("""
<style>
.card {
  border-radius: 14px;
  padding: 18px 20px;
  margin: 8px 0 12px 0;
  box-shadow: 0 8px 18px rgba(0,0,0,.12);
  background: linear-gradient(135deg, rgba(20,32,44,.75), rgba(20,32,44,.55));
  border: 1px solid rgba(255,255,255,.08);
}
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

# --------- Couleurs et légende rubans ---------
# Palette (inchangée)
COLORS = {
    "BM25":        "#9eacd3",  # bleu foncé
    "TF-IDF Cosine": "#37486E",  # bleu un peu moins foncé
    "Hybrid RRF":  "#5373A7",  # bleu moyen
    "Hybrid Interp": "#d9dde0",  # bleu plus clair
}


def ribbon_legend(labels: list[str], colors: list[str]) -> go.Figure:
    """
    Légende custom stable (aucun zoom/pan), en coordonnées 'paper'
    pour éviter tout autoscale et chevauchement.
    """
    fig = go.Figure()

    # Géométrie en coordonnées paper (0..1)
    x0, y0 = 0.06, 0.30     # origine du premier ruban
    w, h   = 0.20, 0.15     # largeur/hauteur d’un ruban
    dx     = 0.05           # décalage pour l’inclinaison (parallélogramme)
    pad    = 0.07           # petit espace entre rubans

    for i, (lab, col) in enumerate(zip(labels, colors)):
        xi = x0 + i * (w - dx + pad)

        # Parallélogramme : coordonnées en 'paper'
        path = (
            f"M {xi},{y0} "
            f"L {xi+w},{y0} "
            f"L {xi+w-dx},{y0+h} "
            f"L {xi-dx},{y0+h} Z"
        )
        fig.add_shape(
            type="path",
            path=path,
            fillcolor=col,
            line=dict(width=0),
            xref="paper", yref="paper",  # <<< clé pour stabilité
            layer="below"
        )

        # Label centré dans le ruban
        fig.add_annotation(
            x=xi + (w / 2),          # <-- ICI : w/2 sans le - dx/2
            y=y0 + (h / 2),
            xref="paper", yref="paper",
            text=lab,
            showarrow=False,
            font=dict(color="white", size=15),
            align="center"
        )


    # On cache axes et on fixe une taille confortable
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(
        height=130,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        dragmode=False
    )
    return fig

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
    delta = None
    if isinstance(before.get("size_kb"), (int, float)) and isinstance(after.get("size_kb"), (int, float)):
        delta = after["size_kb"] - before["size_kb"]
    fleche = "↑" if (delta and delta > 0) else ("↓" if (delta and delta < 0) else "→")
    delta_txt = f"{fleche} {delta:+.1f} KB" if delta is not None else "taille modifiée"
    st.toast(f"✅ {path} mis à jour — {delta_txt}, nouvelle taille: {after.get('size_kb', 0):.1f} KB")

# ============ Header ============
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

    st.markdown("**Paramètres**")

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

        # ----- Scores moyens (macro) : affichage en cartes -----
        st.subheader("Scores moyens ")

        avg_scores = {
            f"P@{k_eval}": round(sum_p / n, 3),
            f"R@{k_eval}": round(sum_r / n, 3),
            "Hit@1":      round(sum_hit1 / n, 3),
            "MRR":        round(sum_mrr / n, 3),
            f"nDCG@{k_eval}": round(sum_ndcg / n, 3),
        }

        # une colonne par métrique
        cols = st.columns(len(avg_scores))

        for col, (name, value) in zip(cols, avg_scores.items()):
            with col:
                st.metric(label=name, value=f"{value:.3f}")


        st.subheader("Distribution des rangs")

        # 1) Même logique de buckets que plus haut
        keys_ordered = [i for i in range(1, k_eval + 1)] + [f">{k_eval}", "miss"]

        # 2) Labels affichés (tout en string)
        x_labels = [str(k) for k in keys_ordered]

        # 3) Valeurs associées (on prend bien les clés originales : int, ">k", "miss")
        y_values = [rank_hist.get(k, 0) for k in keys_ordered]

        # (option debug, tu peux commenter après test)
        # st.write("keys_ordered =", keys_ordered)
        # st.write("x_labels =", x_labels)
        # st.write("y_values =", y_values)

        # 4) Graph statique avec go.Bar
        fig_ranks = go.Figure(
            data=[
                go.Bar(
                x=x_labels,
                y=y_values,
                )
            ]
        )   

        fig_ranks.update_layout(
            margin=dict(l=10, r=10, t=10, b=40),
            xaxis_title="Rang de la bonne réponse",
            yaxis_title="Nombre de requêtes",
            # on force un axe catégoriel dans l'ordre donné
            xaxis=dict(
                type="category",
                categoryorder="array",
                categoryarray=x_labels,
            ),
        )

        st.plotly_chart(
            fig_ranks,
            use_container_width=True,
            config={"staticPlot": True, "displayModeBar": False},
        )


        st.subheader("Détails par requête")
        st.dataframe(rows, use_container_width=True)


        st.download_button(
            "⬇️ Export CSV",
            data="\n".join([",".join(map(str, r.values())) for r in rows]),
            file_name="eval_results.csv",
            mime="text/csv"
        )
    else:
        st.info("Configure et clique dans la sidebar pour évaluer.")

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

# --- Tab Comparison ---
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
                         sub=f" {best_row['Méthode']}", accent=True)
            with c2:
                kpi_card("Meilleure nDCG@k", f"{df.loc[df['nDCG@k'].idxmax(), 'nDCG@k']:.3f}",
                         sub=f" {df.loc[df['nDCG@k'].idxmax(), 'Méthode']}")
            with c3:
                kpi_card("Meilleure MRR", f"{df.loc[df['MRR'].idxmax(), 'MRR']:.3f}",
                         sub=f" {df.loc[df['MRR'].idxmax(), 'Méthode']}")
            with c4:
                kpi_card("Latence min", f"{df['Latence (ms)'].min():.0f} ms",
                         sub=f" {df.loc[df['Latence (ms)'].idxmin(), 'Méthode']}")

            # ======== Jauges (donut) ========
            g1, g2, g3 = st.columns(3)
            metric_map = {"P@k": "P@k", "R@k":"R@k", "Hit@1":"Hit@1", "MRR":"MRR", "nDCG@k":"nDCG@k"}
            key = metric_map[metric_for_gauge]
            row_g = df.loc[df[key].idxmax()]
            with g1:
                st.plotly_chart(donut_gauge(f"{metric_for_gauge} (best)", row_g[key], 1.0), use_container_width=True)
            with g2:
                st.plotly_chart(donut_gauge(f"{metric_for_gauge} (moy.)", df[key].mean(), 1.0), use_container_width=True)
            with g3:
                if len(df) > 1:
                    second = df.sort_values(key, ascending=False).iloc[1]
                    st.plotly_chart(donut_gauge(f"{metric_for_gauge} (2ᵉ)", second[key], 1.0), use_container_width=True)
                else:
                    st.plotly_chart(donut_gauge(f"{metric_for_gauge}", row_g[key], 1.0), use_container_width=True)

            st.markdown('<div class="sep"></div>', unsafe_allow_html=True)

            # ======== Graphiques barres (sans légende) ========
            # ======== Graphiques barres (sans légende) ========
            def plot_metric_group(df: pd.DataFrame, metric: str) -> go.Figure:
                """
                4 barres séparées, plus fines, sans légende.
                Axe Y auto-zoomé autour des valeurs du metric.
                """
                fig = go.Figure()
                xs = list(range(len(df)))  # positions 0,1,2,3

                # valeurs numériques du metric
                vals = [float(v) for v in df[metric].tolist()]
                vmin, vmax = min(vals), max(vals)
                delta = vmax - vmin

                # on définit une petite marge autour (pour bien voir les différences)
                if delta == 0:
                    # tous identiques : on met une fenêtre serrée autour de la valeur
                    margin = max(0.01, vmin * 0.2)
                else:
                    margin = delta * 0.5   # tu peux augmenter à 0.7 si tu veux encore plus zoomé

                y_low  = max(0.0, vmin - margin)
                y_high = min(1.0, vmax + margin)

                for i, (_, row) in enumerate(df.iterrows()):
                    method = row["Méthode"]
                    value = float(row[metric])
                    fig.add_bar(
                        x=[xs[i]],
                        y=[value],
                        width=0.35,
                        marker_color=COLORS.get(method, None),
                        name=method,
                        hovertemplate=f"{method}<br>{metric}: %{{y:.3f}}<extra></extra>"
                )

                fig.update_layout(
                    yaxis=dict(range=[y_low, y_high])  # comme tu as déjà
                )

                # On ne veut pas de texte ni de grille sur l’axe x
                fig.update_xaxes(
                    showticklabels=False,
                    showgrid=False,
                    zeroline=False
                )

                # 👉 Ajouter une ligne horizontale qui joue le rôle d'axe x
                fig.add_shape(
                    type="line",
                    x0=-0.5,                  # un peu avant la 1ère barre
                    x1=len(xs) - 0.5,         # un peu après la dernière barre
                    y0=y_low, y1=y_low,       # pile en bas de la fenêtre
                    line=dict(color="#e5e7eb", width=1),
                    layer="below"
                )

                return fig



            st.subheader("Scores par méthode")
            config_static = {"staticPlot": True, "displayModeBar": False}

            colm1, colm2 = st.columns(2)

            with colm1:
                # 1er graphique
                st.plotly_chart(
                    plot_metric_group(df, "P@k"),
                    use_container_width=True,
                    config=config_static,
                )
                # ESPACE vertical
                st.markdown("<div style='height:25px;'></div>", unsafe_allow_html=True)

                # 2e graphique
                st.plotly_chart(
                    plot_metric_group(df, "Hit@1"),
                    use_container_width=True,
                    config=config_static,
                )

            with colm2:
                # 3e graphique
                st.plotly_chart(
                    plot_metric_group(df, "R@k"),
                    use_container_width=True,
                    config=config_static,
                )
                # ESPACE vertical
                st.markdown("<div style='height:25px;'></div>", unsafe_allow_html=True)

                # 4e graphique
                st.plotly_chart(
                    plot_metric_group(df, "MRR"),
                    use_container_width=True,
                    config=config_static,
                )   

            # encore un petit espace avant le dernier
            st.markdown("<div style='height:30px;'></div>", unsafe_allow_html=True)

            st.plotly_chart(
                plot_metric_group(df, "nDCG@k"),
                use_container_width=True,
                config=config_static,
            )


            # ======== Tableau récap ========
            def styled_table(df: pd.DataFrame) -> go.Figure:
                """
                Tableau comparatif stylé avec en-tête foncée et zébrage des lignes.
                """
                # On enlève l'index 0,1,2,3 qui ne sert à rien dans l'affichage
                df_display = df.reset_index(drop=True)

                headers = list(df_display.columns)
                cells = [df_display[col].tolist() for col in headers]

                n_rows = len(df_display)
                # zébrage des lignes : une couleur sur les lignes paires, une autre sur les impaires
                row_colors = [
                    "rgba(15,23,42,0.95)" if i % 2 == 0 else "rgba(31,41,55,0.95)"
                    for i in range(n_rows)
                ]

                fig = go.Figure(
                    data=[
                        go.Table(
                            columnwidth=[60] + [40] * (len(headers) - 1),
                            header=dict(
                                values=headers,
                                fill_color="#020617",         # bandeau très foncé
                                line_color="#0f172a",
                                align="center",
                                font=dict(color="white", size=14, family="Segoe UI"),
                                height=38,
                            ),
                            cells=dict(
                                values=cells,
                                fill_color=[row_colors],      # zébrage
                                line_color="#0f172a",
                                align=["left"] + ["center"] * (len(headers) - 1),
                                font=dict(color="#e5e7eb", size=13),
                                height=32,
                            ),
                        )
                    ]
                )

                fig.update_layout(
                    margin=dict(l=0, r=0, t=0, b=0),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                return fig

            st.subheader("Tableau comparatif")
            st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

            st.plotly_chart(
                styled_table(df),
                use_container_width=True,
                config={"staticPlot": True, "displayModeBar": False}
            )

