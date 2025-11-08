from __future__ import annotations
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
import json, os, math

from src.corpus import CorpusReader
from src.preprocess import TextPreprocessor, make_ngrams, make_edge_ngrams

class InvertedIndex:
    """
    Index inversé minimal pour IR :
      - postings: term -> { doc_id: tf , doc_id2: tf2 ... }
      - df:       term -> nb de documents contenant le terme
      - doc_len:  doc_id -> nb de tokens (après prétraitement et n-grams)
      - N:        nb de documents
      - avgdl:    longueur moyenne des docs
    """  
    def __init__(self):
        # posting  pour chaque terme ( unigram bigram) on garde les docs où il apparait et sa fréquence TF 
        self.postings: Dict[str, Dict[str, int]] = defaultdict(dict)
        # Document frequency: nombre de documents contenant le terme
        self.df: Dict[str, int] = {}
        # nb de tokens (après prétraitement et n-grams)
        self.doc_len: Dict[str, int] = {}
        #nbre de documents 
        self.N: int = 0
        #longueur moyenne des documents 
        self.avgdl: float = 0.0
    
    #But: construire l'index 
    def build(self, docs_dir: str, preproc: TextPreprocessor, use_bigrams: bool = True):
        # lecture des docs via la classe CorpusReader
        cr = CorpusReader(docs_dir)
        self.N = len(cr)

        for doc_id, text in cr.iter_docs():
            # 1) tokens unigrams
            tokens = preproc.process(text)
            # 2) (option) bigrams de tokens
            if use_bigrams:
                tokens += make_ngrams(tokens, n=2)
            #tokens = unigrams , bigrams
            # longueur doc
            self.doc_len[doc_id] = len(tokens)

            # tf par terme dans ce doc
            tf = Counter(tokens) # pair[term,tf]
            for term, c in tf.items():
                self.postings[term][doc_id] = c

        # df et avgdl
        self.df = {t: len(docmap) for t, docmap in self.postings.items()}
        self.avgdl = (sum(self.doc_len.values()) / self.N) if self.N else 0.0

    # ---------- Création fichier index.json ----------
    def save(self, out_dir: str = "models"):
        os.makedirs(out_dir, exist_ok=True)
        # prépare un dictionnaire 
        data = {
            "N": self.N,
            "avgdl": self.avgdl,
            "df": self.df,
            "doc_len": self.doc_len,
            "postings": self.postings,
        }
        #ouvre ou créer le fichier index.json 
        with open(os.path.join(out_dir, "index.json"), "w", encoding="utf-8") as f:
            # sérialise le dict data en json dans le fichier
            # ensure_ascii juste pour s'assurer que les caractères non ASCII s'affiche proprement. 
            json.dump(data, f, ensure_ascii=False) 
    
    # ---------- Creation du fichier index_edge_ngrams.json ----------
    def save_edge_index(self, out_dir: str = "models"):
        """
        Crée un nouveau fichier JSON contenant les edge n-grams
        de chaque terme du vocabulaire (tous les mots de l'index).
        """
        os.makedirs(out_dir, exist_ok=True)

        # Construire le vocabulaire à partir des postings
        vocab = list(self.postings.keys())

        #ne garde que les unigrams
        unigrams = []
        for term in vocab:
            # ignorer les bigrams ou tokens spéciaux
            if "_" in term or " " in term or "-" in term or len(term) <= 1:
                continue
            # ignorer tout token contenant un chiffre
            if any(c.isdigit() for c in term):
                continue
            unigrams.append(term)

        # Générer edge n-grams pour chaque mot du vocabulaire
        edge_index = {term: make_edge_ngrams(term) for term in unigrams}

        edge_index = dict(sorted(edge_index.items()))  # trier par terme
        
        print(f"Généré edge n-grams pour {len(edge_index)} termes.")

        # Sauvegarde dans models/edge_index.json
        data = {
            "vocab_size": len(unigrams),
            "edge_index": edge_index
        }

        with open(os.path.join(out_dir, "edge_index.json"), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    #pas besoin de self 
    @staticmethod 
    def load(path: str = "models/index.json") -> "InvertedIndex":
        with open(path, encoding="utf-8") as f:
            # contenu de ficheir json
            data = json.load(f)
        idx = InvertedIndex()
        idx.N = int(data["N"])
        idx.avgdl = float(data["avgdl"])
        idx.df = {t: int(v) for t, v in data["df"].items()}
        idx.doc_len = {d: int(v) for d, v in data["doc_len"].items()}
        # retype postings
        idx.postings = defaultdict(dict, {t: {d: int(tf) for d, tf in docs.items()} for t, docs in data["postings"].items()})
        return idx
    
    @staticmethod 
    def load_edge_index(path="models/edge_index.json"):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data["edge_index"]



    #---------------------------------Metriques de recherche---------------------------------
    # ---------- BM25 ----------
    # tf : term frequency dans le doc d
    # df : document frequency (nombre de docs contenant t)
    # dl : longueur du doc d (nb de tokens indexés).
    # N : nombre total de documents.
    # avgdl : longueur moyenne des docs.
    # k1 : contrôle la saturation de tf.
    
    #bm25_term = score du mot pour un document donné (plus grand = plus pertinent).
    def _bm25_term(self, tf: int, df: int, dl: int, k1: float = 1.2, b: float = 0.75) -> float:
        # IDF Robertson/Sparck Jones (stable)
        idf = math.log(1 + (self.N - df + 0.5) / (df + 0.5))
        denom = tf + k1 * (1 - b + b * (dl / (self.avgdl or 1.0)))
        return idf * ((tf * (k1 + 1)) / denom) if denom else 0.0
    
    #Renvoie les top_k documents les plus pertinents pour la requête.
    def search(self, query: str, preproc: TextPreprocessor, use_bigrams: bool = True, top_k: int = 10) -> List[Tuple[str, float]]:
        # mêmes features que côté docs
        q_tokens = preproc.process(query)
        #si on utilise bigrams pour l'indexation de docs, on les utilise aussi pour la requête
        if use_bigrams:
            q_tokens += make_ngrams(q_tokens, n=2)
        # scores par doc
        scores: Dict[str, float] = defaultdict(float)
        # Un ensemble pour mémoriser quels termes de la requête ont déjà été traités.
        used = set()
        for t in q_tokens:
            if t in used:   # évite de compter 2x le même terme de requête
                continue
            used.add(t)
            # docs représente { doc_id1: tf1 , doc_id2: tf2 ... }
            docs = self.postings.get(t)
            if not docs:
                continue
            df = self.df.get(t, 0)
            #On parcourt tous les documents qui contiennent t, avec leur fréquence tf dans ce document.
            for doc_id, tf in docs.items():
                scores[doc_id] += self._bm25_term(tf=tf, df=df, dl=self.doc_len[doc_id])
        #On trie les paires (doc_id, score) par score décroissant et on retourne les top_k meilleurs.
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    # ---------- Inspecteur ----------
    #Retourne {doc_id: tf} pour un terme
    def postings_for(self, term: str) -> Dict[str, int]:
        return dict(self.postings.get(term, {}))
