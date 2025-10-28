# tf-idf.py
# -----------------------------------------
# Moteur TF-IDF (index, search, eval)
# -----------------------------------------
import argparse, json, math, re, sys
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
from scipy.sparse import csr_matrix
from tqdm import tqdm
from unidecode import unidecode

# ---------- Prétraitement ----------
# mots très fréquents mais peu informatifs
STOPWORDS = {
    "le","la","les","un","une","des","de","du","au","aux","et","ou","mais","donc","or","ni","car",
    "a","à","dans","en","sur","sous","pour","par","avec","sans","chez","comme","plus","moins",
    "ce","cet","cette","ces","se","sa","son","ses","leur","leurs","il","elle","ils","elles","on",
    "je","tu","nous","vous","me","te","moi","toi","notre","votre","nos","vos","qui","que","quoi",
    "où","quand","comment","pourquoi","est","sont","été","etre","être","ai","as","avons","avez",
    "ont","avait","avaient","être","faire","fait","afin","ainsi","très","tres","plus","moins","ne",
    "pas","plus","tout","toute","tous","toutes","l","d","s"
}

TOKEN_RE = re.compile(r"[a-z0-9]+")

# Fonction pour nettoyer et norlmaliser le texte
def preprocess(text: str):
    text = text.lower()
    text = unidecode(text)              # é -> e, etc.
    toks = TOKEN_RE.findall(text)
    return [t for t in toks if t not in STOPWORDS and len(t) > 1]

# ---------- Chargement des documents ----------
def load_documents(data_dir: Path):
    docs = []
    files = sorted([p for p in data_dir.glob("*.txt")])
    for i, p in enumerate(tqdm(files, desc="Lecture docs")):
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except:
            txt = p.read_text(encoding="latin-1", errors="ignore")
        docs.append((i, p.name, txt))
    return docs

# ---------- Construction TF-IDF (CSR) ----------
def build_tfidf(docs):
    """
    docs: list of (doc_id, filename, raw_text)
    returns:
      X: csr_matrix shape (N_docs, V)
      idf: np.array shape (V,)
      vocab: dict token->col
      doc_ids: list of filenames indexed by row
      row_norms: np.array (N_docs,) normes L2
    """
    # 1) Tokenisation par doc + comptages
    doc_tokens = []
    df_counter = Counter()
    for (doc_id, _, txt) in tqdm(docs, desc="Tokenisation"):
        toks = preprocess(txt)
        doc_tokens.append(toks)
        for t in set(toks):
            df_counter[t] += 1

    # 2) Vocabulaire (token -> colonne)
    vocab = {t:i for i, (t, _) in enumerate(sorted(df_counter.items(), key=lambda x:x[0]))}
    V = len(vocab)
    N = len(docs)

    # 3) TF bruts -> listes CSR
    data, rows, cols = [], [], []
    for row, toks in enumerate(tqdm(doc_tokens, desc="TF->CSR")):
        tf = Counter(toks)
        for tok, c in tf.items():
            if tok not in vocab: 
                continue
            col = vocab[tok]
            # TF log-scalé
            val = 1.0 + math.log(c)
            data.append(val); rows.append(row); cols.append(col)

    X_tf = csr_matrix((np.array(data, dtype=np.float32),
                       (np.array(rows), np.array(cols))),
                      shape=(N, V), dtype=np.float32)

    # 4) IDF lissé
    df = np.zeros(V, dtype=np.int32)
    for tok, col in vocab.items():
        df[col] = df_counter[tok]
    idf = np.log((N + 1) / (df + 1)) + 1.0
    idf = idf.astype(np.float32)

    # 5) TF-IDF + normalisation L2 par document
    # X * diag(idf)
    X = X_tf.multiply(idf)
    # normes
    row_norms = np.sqrt((X.multiply(X)).sum(axis=1)).A1
    row_norms[row_norms == 0] = 1.0
    # normalise en place
    inv_norms = 1.0 / row_norms
    X = X.multiply(inv_norms[:, None])

    doc_ids = [fn for _, fn, _ in docs]
    return X.tocsr(), idf, vocab, doc_ids, row_norms  # row_norms renvoyé si besoin plus tard

# ---------- Vectorisation d'une requête ----------
def query_vector(q: str, vocab, idf):
    toks = preprocess(q)
    tf = Counter([t for t in toks if t in vocab])
    if not tf:
        return None  # requête vide après nettoyage
    cols, data = [], []
    for t, c in tf.items():
        col = vocab[t]
        val = (1.0 + math.log(c)) * idf[col]
        cols.append(col)
        data.append(val)
    V = len(idf)
    q_vec = csr_matrix((np.array(data, dtype=np.float32),
                        (np.zeros(len(cols)), np.array(cols))),
                       shape=(1, V), dtype=np.float32)
    # normalisation L2
    n = math.sqrt(q_vec.multiply(q_vec).sum())
    if n > 0:
        q_vec = q_vec.multiply(1.0 / n)
    return q_vec.tocsr()

# ---------- Recherche top-k ----------
def search_topk(q: str, X, vocab, idf, doc_ids, k=10):
    qv = query_vector(q, vocab, idf)
    if qv is None:
        return []
    # cosinus = qv (L2) dot X^T (L2) -> produit scalaire
    scores = (qv @ X.T).toarray().flatten()  # (N_docs,)
    if k >= len(scores):
        top_idx = np.argsort(-scores)
    else:
        top_idx = np.argpartition(-scores, k)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
    return [(doc_ids[i], float(scores[i])) for i in top_idx]

# ---------- Évaluation (requetes.jsonl) ----------
def evaluate(requetes_path: Path, X, vocab, idf, doc_ids, agg="mean", k=10):
    """
    agg: 'mean' (moyenne des scores des différentes formulations)
         'max'  (meilleur résultat parmi les formulations)
    """
    gold_hit_at_1 = []
    gold_hit_at_3 = []
    mrrs = []

    name_to_rank = {fn:i for i, fn in enumerate(doc_ids)}  # si besoin

    # Lecture robuste du JSONL : ignore lignes vides, gère BOM, et signale l’erreur
    lines = []
    with requetes_path.open("r", encoding="utf-8") as f:
        for ln_no, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue  # ignore ligne vide
            raw = raw.lstrip("\ufeff")  # supprime BOM si présent
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as e:
                msg = (
                    f"Fichier JSONL invalide à la ligne {ln_no}: {e.msg} "
                    f"(col {e.colno}).\n"
                    f"Ligne brute: {raw[:200]}"
                )
                raise RuntimeError(msg) from e
            lines.append(obj)


    for ex in tqdm(lines, desc="Éval"):
        gold = ex["Answer file"]
        queries = ex["Queries"]

        # pour agréger : on additionne les vecteurs de score doc-centrés
        # (simple et équitable entre formulations)
        acc_scores = np.zeros(len(doc_ids), dtype=np.float32)
        max_scores = np.full(len(doc_ids), -np.inf, dtype=np.float32)

        for q in queries:
            qv = query_vector(q, vocab, idf)
            if qv is None:
                continue
            s = (qv @ X.T).toarray().flatten() # (N_docs,)
            acc_scores += s
            max_scores = np.maximum(max_scores, s)

        if agg == "max":
            scores = max_scores
        else:
            n = max(1, len(queries))
            scores = acc_scores / n

        # rangs
        order = np.argsort(-scores)
        # métriques
        # Hit@k
        top1 = [doc_ids[order[0]]] if len(order) > 0 else []
        hit1 = 1 if gold in top1 else 0
        top3 = set([doc_ids[i] for i in order[:3]])
        hit3 = 1 if gold in top3 else 0
        # MRR@k
        rank = None
        for r, idx in enumerate(order[:k], start=1):
            if doc_ids[idx] == gold:
                rank = r
                break
        mrr = 1.0 / rank if rank is not None else 0.0

        gold_hit_at_1.append(hit1)
        gold_hit_at_3.append(hit3)
        mrrs.append(mrr)

    res = {
        "n": len(mrrs),
        "Hit@1": float(np.mean(gold_hit_at_1)),
        "Hit@3": float(np.mean(gold_hit_at_3)),
        "MRR@{}".format(k): float(np.mean(mrrs)),
        "agg": agg,
        "k": k
    }
    return res

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="TF-IDF minimal IR")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_index = sub.add_parser("index", help="Construire TF-IDF en mémoire et sauvegarder npz (optionnel)")
    ap_index.add_argument("--data_dir", type=str, required=True)
    ap_index.add_argument("--save", type=str, default="", help="chemin .npz pour sauvegarder X; + vocab/idf/doc_ids en .json/.npy")

    ap_search = sub.add_parser("search", help="Chercher top-k")
    #ap_search.add_argument("--data_dir", type=str, required=True)
    ap_search.add_argument("--k", type=int, default=10)

    ap_eval = sub.add_parser("eval", help="Évaluer sur requetes.jsonl")
    #ap_eval.add_argument("--data_dir", type=str, required=True)
    #ap_eval.add_argument("--queries", type=str, required=True)
    ap_eval.add_argument("--k", type=int, default=10)
    ap_eval.add_argument("--agg", type=str, choices=["mean","max"], default="mean")

    args = ap.parse_args()

    if args.cmd in ("search","eval"):
        data_dir = Path("data") #Path(args.data_dir)
        docs = load_documents(data_dir)
        X, idf, vocab, doc_ids, _ = build_tfidf(docs)

    if args.cmd == "index":
        data_dir = Path("data") #Path(args.data_dir)
        docs = load_documents(data_dir)
        X, idf, vocab, doc_ids, _ = build_tfidf(docs)
        if args.save:
            out = Path(args.save)
            np.savez_compressed(out, data=X.data, indices=X.indices, indptr=X.indptr, shape=X.shape)
            np.save(out.with_suffix(".idf.npy"), idf)
            Path(out.with_suffix(".vocab.json")).write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
            Path(out.with_suffix(".docids.json")).write_text(json.dumps(doc_ids, ensure_ascii=False), encoding="utf-8")
            print(f"Sauvé: {out}, {out.with_suffix('.idf.npy')}, {out.with_suffix('.vocab.json')}, {out.with_suffix('.docids.json')}")

    elif args.cmd == "search":
        query = input("Enter your research: ")
        res = search_topk(query, X, vocab, idf, doc_ids, k=args.k)
        for fn, sc in res:
            print(f"{sc: .4f}\t{fn}")

    elif args.cmd == "eval":
        requetes_path = Path("requetes.jsonl") #Path(args.queries)
        res = evaluate(requetes_path, X, vocab, idf, doc_ids, agg=args.agg, k=args.k)
        print(json.dumps(res, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
