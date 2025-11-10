#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script de test pour le moteur de recherche.
- BM25 (toujours)
- Cosinus TF-IDF et Hybride 
- Évaluation sur un JSONL (Answer file + Queries)

Exemples:
  python3 test.py --query "intelligence artificielle" --docs_dir data/wiki_split_extract_2k --method bm25 --top_k 10 --bigrams
  python3 test.py --query "intelligence artificielle" --method cosine --bigrams
  python3 test.py --query "intelligence artificielle" --method hybrid --bigrams

Évaluation:
  python3 test.py --eval --queries_file data/requetes.jsonl --docs_dir data/wiki_split_extract_2k --bigrams
  python3 test.py --eval --eval_methods bm25,cosine,hybrid --queries_file data/requetes.jsonl
"""

import os
import sys
import time
import math
import argparse
import json
from typing import List, Tuple

# Assure l'import de "src" quand on lance depuis la racine
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.preprocess import TextPreprocessor
from src.index import InvertedIndex
from src.corpus import CorpusReader


# ----------------------------- CLI -----------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Test moteur de recherche (BM25 / Cosinus / Hybride).")

    # Mode recherche ad hoc
    p.add_argument("--query", "-q", type=str, help="Requête à tester (si non --eval).")
    p.add_argument("--docs_dir", type=str, default="data/wiki_split_extract_2k",
                   help="Dossier contenant les .txt à indexer.")
    p.add_argument("--method", type=str, default="bm25",
                   choices=["bm25", "cosine", "hybrid"],
                   help="Méthode : bm25, cosine (TF-IDF), hybrid (BM25+cosine).")
    p.add_argument("--top_k", type=int, default=10, help="Nombre de résultats à afficher.")
    p.add_argument("--bigrams", action="store_true", help="Utiliser des bigrams.")
    p.add_argument("--alpha", type=float, default=0.6,
                   help="Poids BM25 si fusion par interpolation.")
    p.add_argument("--show_snippet", action="store_true",
                   help="Affiche un extrait des documents.")

    # Mode évaluation
    p.add_argument("--eval", action="store_true",
                   help="Évalue les méthodes sur un fichier JSONL (Answer file + Queries).")
    p.add_argument("--queries_file", type=str, default="data/requetes.jsonl",
                   help="Chemin du fichier JSONL.")
    p.add_argument("--eval_methods", type=str, default="bm25,cosine,hybrid",
                   help="Méthodes à évaluer, séparées par des virgules.")

    return p.parse_args()


# ------------------------ Métriques ------------------------

def _ndcg_binary(rank: int) -> float:
    """NDCG binaire : 1 / log2(rank+1) si trouvé, sinon 0."""
    if rank is None or rank <= 0:
        return 0.0
    return 1.0 / (math.log2(rank + 1))

def _reciprocal_rank(rank: int) -> float:
    return 0.0 if rank is None or rank <= 0 else 1.0 / rank


# ------------------------ Évaluation ------------------------

def evaluate_methods(
    idx: InvertedIndex,
    preproc: TextPreprocessor,
    queries_file: str,
    use_bigrams: bool = True,
    methods: List[str] = None,
    ks: Tuple[int, ...] = (1, 3, 10),
) -> None:
    """
    Évalue différentes méthodes sur un JSONL :
      {"Answer file": "wiki_XXXX.txt", "Queries": ["q1", "q2", ...]}
    """
    if methods is None:
        methods = ["bm25", "cosine", "hybrid"]

    has_tfidf = hasattr(idx, "build_tfidf")
    has_cosine = hasattr(idx, "search_cosine")
    has_hybrid_rrf = hasattr(idx, "search_hybrid_rrf")
    has_hybrid_interp = hasattr(idx, "search_hybrid_interp")

    available = []
    for m in methods:
        m = m.strip().lower()
        if m == "bm25":
            available.append("bm25")
        elif m == "cosine" and has_tfidf and has_cosine:
            available.append("cosine")
        elif m == "hybrid" and has_tfidf and (has_hybrid_rrf or has_hybrid_interp):
            available.append("hybrid")
    if not available:
        raise RuntimeError("Aucune méthode disponible parmi: " + ", ".join(methods))

    # Construit TF-IDF si besoin
    if any(m in ("cosine", "hybrid") for m in available) and has_tfidf:
        idx.build_tfidf(tf_scheme="log")

    # Charge les requêtes
    examples = []
    with open(queries_file, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            ans = data.get("Answer file")
            qs = data.get("Queries", [])
            if ans and qs:
                examples.append((ans, qs))
    if not examples:
        raise ValueError(f"Aucun exemple valide dans {queries_file}")

    max_k = max(ks)
    hits = {m: {k: [] for k in ks} for m in available}
    mrrs = {m: {k: [] for k in ks} for m in available}
    ndcgs = {m: {k: [] for k in ks} for m in available}

    def run_method(method: str, query: str, top_k: int):
        if method == "bm25":
            return idx.search(query, preproc, use_bigrams=use_bigrams, top_k=top_k)
        elif method == "cosine":
            return idx.search_cosine(query, preproc, use_bigrams=use_bigrams, top_k=top_k)
        else:
            return idx.search_hybrid_interp(query, preproc, use_bigrams=use_bigrams,
                                            top_k=top_k, alpha=0.6)
            """
            if has_hybrid_rrf:
                return idx.search_hybrid_rrf(query, preproc, use_bigrams=use_bigrams, top_k=top_k)
            return idx.search_hybrid_interp(query, preproc, use_bigrams=use_bigrams,
                                            top_k=top_k, alpha=0.6)
            """

    total_queries = 0
    for gold_file, queries in examples:
        for q in queries:
            total_queries += 1
            for m in available:
                results = run_method(m, q, top_k=max_k)
                ranked_ids = [d for d, _ in results]
                rank = next((i for i, d in enumerate(ranked_ids, 1) if d == gold_file), None)
                for k in ks:
                    hit = 1.0 if rank and rank <= k else 0.0
                    hits[m][k].append(hit)
                    mrrs[m][k].append(_reciprocal_rank(rank) if rank and rank <= k else 0.0)
                    ndcgs[m][k].append(_ndcg_binary(rank) if rank and rank <= k else 0.0)

    # Résumé
    print(f"\n📊 Évaluation sur {total_queries} requêtes ({queries_file})")
    header = ["Méthode"]
    for k in ks:
        header += [f"Hit@{k}", f"MRR@{k}", f"NDCG@{k}"]
    print(" | ".join(f"{h:>10}" for h in header))
    print("-" * (13 * len(header)))

    for m in available:
        row = [m.upper()]
        for k in ks:
            H = sum(hits[m][k]) / len(hits[m][k])
            M = sum(mrrs[m][k]) / len(mrrs[m][k])
            N = sum(ndcgs[m][k]) / len(ndcgs[m][k])
            row += [f"{H:6.3f}", f"{M:6.3f}", f"{N:6.3f}"]
        print(" | ".join(f"{c:>10}" for c in row))


# --------------------------- Recherche ---------------------------

def ensure_methods(idx: InvertedIndex, method: str):
    """Vérifie la présence des fonctions requises dans l’index."""
    has_tfidf = hasattr(idx, "build_tfidf")
    has_cosine = hasattr(idx, "search_cosine")
    has_hybrid = hasattr(idx, "search_hybrid_rrf") or hasattr(idx, "search_hybrid_interp")
    if method == "cosine" and not (has_tfidf and has_cosine):
        raise RuntimeError("La méthode 'cosine' nécessite build_tfidf() et search_cosine().")
    if method == "hybrid" and not (has_tfidf and has_hybrid):
        raise RuntimeError("La méthode 'hybrid' nécessite build_tfidf() + search_hybrid_*.") 
    return has_tfidf, has_cosine, has_hybrid


# --------------------------- MAIN ---------------------------

def main():
    args = parse_args()

    if not os.path.isdir(args.docs_dir):
        raise FileNotFoundError(f"Dossier introuvable: {args.docs_dir}")

    preproc = TextPreprocessor(use_stemming=True, keep_digits=True)
    idx = InvertedIndex()

    t0 = time.perf_counter()
    idx.build(args.docs_dir, preproc, use_bigrams=args.bigrams)
    print(f"✅ Index BM25 construit en {time.perf_counter()-t0:.2f}s — N={idx.N}, avgdl={idx.avgdl:.1f}, vocab={len(idx.df)}")

    # Évaluation
    if args.eval:
        method_list = [m.strip() for m in args.eval_methods.split(",") if m.strip()]
        evaluate_methods(idx, preproc, args.queries_file,
                         use_bigrams=args.bigrams, methods=method_list)
        return

    # Recherche ad hoc
    if not args.query:
        print("⚠️  Pas de --query ni --eval.")
        return

    has_tfidf, has_cosine, has_hybrid = ensure_methods(idx, args.method)
    if args.method in ("cosine", "hybrid"):
        t1 = time.perf_counter()
        idx.build_tfidf(tf_scheme="log")
        print(f"✅ TF-IDF construit en {time.perf_counter()-t1:.2f}s")

    q = args.query.strip()
    print(f"\n🔎 Requête: {q!r} | méthode: {args.method.upper()} | top_k={args.top_k} | bigrams={args.bigrams}")

    t2 = time.perf_counter()
    if args.method == "bm25":
        results = idx.search(q, preproc, use_bigrams=args.bigrams, top_k=args.top_k)
    elif args.method == "cosine":
        results = idx.search_cosine(q, preproc, use_bigrams=args.bigrams, top_k=args.top_k)
    else:
        if hasattr(idx, "search_hybrid_rrf"):
            results = idx.search_hybrid_rrf(q, preproc, use_bigrams=args.bigrams, top_k=args.top_k)
        else:
            results = idx.search_hybrid_interp(q, preproc, use_bigrams=args.bigrams,
                                               top_k=args.top_k, alpha=args.alpha)
    print(f"🕒 Recherche exécutée en {time.perf_counter()-t2:.3f}s\n")

    if not results:
        print("Aucun résultat.")
        return

    for rank, (doc_id, score) in enumerate(results, start=1):
        print(f"{rank:2d}. {doc_id} -> {score:.4f}")
        if args.show_snippet:
            try:
                path = os.path.join(args.docs_dir, doc_id)
                with open(path, encoding="utf-8") as f:
                    snippet = f.read(400).replace("\n", " ")
                print(f"    {snippet}…\n")
            except Exception as e:
                print(f"    (Impossible d'afficher l'extrait: {e})\n")


if __name__ == "__main__":
    main()
