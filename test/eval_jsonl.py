from __future__ import annotations
import json
from typing import Dict, List, Tuple
from src.search import SearchEngine
from src.metrics import precision_at_k, recall_at_k

JSONL_PATH = "data/requetes.jsonl"
K = 10
TOP = 50   # récupère un peu plus large, on coupera à k pour les métriques

def load_queries_from_jsonl(path: str) -> List[Tuple[str, str, str]]:
    """
    Retourne une liste de triples (qid, query, answer_doc).
    qid est une string "Q1", "Q2", ...
    """
    triples = []
    qid = 1
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            answer = obj.get("Answer file") or obj.get("answer") or obj.get("answer_file")
            queries = obj.get("Queries") or obj.get("queries") or []
            for q in queries:
                triples.append((f"Q{qid}", q, answer))
                qid += 1
    return triples

def main():
    # 1) charger requêtes/ground-truth
    triples = load_queries_from_jsonl(JSONL_PATH)
    # construire qrels binaire: qrels[qid][doc_id] = 1
    qrels: Dict[str, Dict[str, int]] = {}
    queries_by_qid: Dict[str, str] = {}
    for qid, query, doc in triples:
        queries_by_qid[qid] = query
        qrels.setdefault(qid, {})[doc] = 1

    # 2) charger index
    engine = SearchEngine()
    engine.load()

    # 3) évaluation
    sum_p = sum_r = 0.0
    n = len(qrels)

    print(f"Evaluating on {n} queries — metrics at k={K}\n")

    for qid, query in queries_by_qid.items():
        results = engine.search(query, top_k=TOP)  # [(doc_id, score), ...]
        run = [doc_id for doc_id, _ in results]
        rels = qrels[qid]  # {answer_doc: 1}

        p = precision_at_k(run, rels, k=K)    # Hit@k / k
        r = recall_at_k(run, rels, k=K)       # Hit@k (0 ou 1, car 1 pertinent)

        sum_p += p
        sum_r += r

        # petit affichage: rang du doc attendu si présent
        answer_doc = next(iter(rels.keys()))
        try:
            rank = run.index(answer_doc) + 1
            where = f"FOUND rank={rank}"
        except ValueError:
            where = "MISS"
        print(f"{qid}: P@{K}={p:.3f}  R@{K}={r:.3f}  [{where}]  | {query}")

    if n:
        print("\n— Macro-averages —")
        print(f"P@{K}: {sum_p / n:.3f}")
        print(f"R@{K}: {sum_r / n:.3f}")

if __name__ == "__main__":
    main()
