from __future__ import annotations
from typing import List, Dict
import math

#---------------------------------------Precision@K----------------------------------------------------#
# run c'est la liste des docs retourné , relevant c'est les docs pertinent, pex avoir soit valeur 0 pas pertinent soit supérieur à 0 donc pertinent 
def precision_at_k(run: List[str], relevant: Dict[str, int], k: int = 10) -> float:
    if k <= 0: return 0.0
    topk = run[:k]
    hits = sum(1 for d in topk if relevant.get(d, 0) > 0)
    # ici on eux diviser que par K ( classic) ou faire cette méthode si jamais le nombre de docs retourné est plu petit que K 
    return hits / min(k, len(run)) if run else 0.0
#---------------------------------------Recall@K----------------------------------------------------#
# on divise le nombre de documents pertinents trouvés dans K par le nombre de tous les documents pertinentstrouvés 
def recall_at_k(run: List[str], relevant: Dict[str, int], k: int = 10) -> float:
    total_rel = sum(1 for r in relevant.values() if r > 0)
    if total_rel == 0: return 0.0
    topk = run[:k]
    found = sum(1 for d in topk if relevant.get(d, 0) > 0)
    return found / total_rel
#---------------------------------------Hit@K----------------------------------------------------#
def hit_at_k(run, relevant, k=10):
    return 1 if any(d in relevant and relevant[d]>0 for d in run[:k]) else 0
#---------------------------------------Hit@1----------------------------------------------------#
# plus adapté dans notre cas puisque dans notre cas on a un seule doc pertinent par requete 
def hit_at_1(run: List[str], relevant: Dict[str, int]) -> float:
    """Accuracy@1: 1 si le doc pertinent est en rang 1, sinon 0."""
    if not run:
        return 0.0
    return 1.0 if relevant.get(run[0], 0) > 0 else 0.0
#---------------------------------------MRR--------------------------------------------------------#
def mrr(run, relevant):
    for i, d in enumerate(run, start=1):
        if relevant.get(d, 0) > 0:
            return 1.0 / i
    return 0.0
#---------------------------------------MRR@K(queries) RR@K  --------------------------------------------------------#
def reciprocal_rank(run: List[str], relevant: Dict[str, int], k: int | None = None) -> float:
    if k is not None:
        run = run[:k]
    for i, d in enumerate(run, start=1):
        if relevant.get(d, 0) > 0:
            return 1.0 / i
    return 0.0

def mean_reciprocal_rank(runs: Dict[str, List[str]], qrels: Dict[str, Dict[str, int]], k: int | None = None) -> float:
    # runs:  {qid: [doc_ids triés]}
    # qrels: {qid: {doc_id: gain}}
    scores = [reciprocal_rank(runs[qid], qrels.get(qid, {}), k=k) for qid in runs]
    return sum(scores) / len(scores) if scores else 0.0

#---------------------------------------NDCG@K----------------------------------------------------#
# penalise les documents pertinents qui sont mal classés (plus bas dans la liste des résultats) mais pénalise moins sévèrement que MRR
def ndcg_at_k(run, relevant, k=10):
    dcg = 0.0
    for i, d in enumerate(run[:k], start=1):
        rel = 1.0 if relevant.get(d, 0) > 0 else 0.0
        if rel:
            dcg += rel / math.log2(i + 1)
    # IDCG = nombre de pertinents possibles dans top-k (ici 1) à la meilleure position (rang 1) -> 1.0
    idcg = 1.0 if any(v > 0 for v in relevant.values()) else 0.0
    return (dcg / idcg) if idcg > 0 else 0.0
