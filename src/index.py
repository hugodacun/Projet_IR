# src/index.py
from __future__ import annotations
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
import datetime, json, os, math

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
        # TF-IDF-related (initialisés vides)
        self.idf = {}
        self.doc_tfidf = {}
        self.doc_norm = {}
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

        #ouvre ou créer le fichier index.json  s'il existe déjà on ( tronque le fichier à 0 octet puis réécrire en l'écrasant)
        with open(os.path.join(out_dir, "index.json"), "w", encoding="utf-8") as f:
            # sérialise le dict data en json dans le fichier
            # ensure_ascii juste pour s'assurer que les caractères non ASCII s'affiche proprement. 
            json.dump(data, f, ensure_ascii=False, indent=2) 
    #pas besoin de self 

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

    # -------------------TF-IDF des mots de document---------------------------------------------

    def _idf(self, df:int) -> float:
        """
        - on utilise 1 pour éviter d'avoir des résutats négatifs ainsi d'avoir 0 
        - IDF négatif est pour les termes ultras fréquents 
        """
        return math.log(1.0 + (self.N + 1.0) / (df + 1.0))
        
    def build_tfidf(self, tf_scheme: str = "log"):
        """ doc {term: tf-idf}
        Construit les vecteurs TF-IDF documentaires et leurs normes.
        tf_scheme: "raw" (tf), "log" (1+log tf)
        """
        # 1) idf par terme idf->{terme, idf}
        self.idf = {t: self._idf(df) for t, df in self.df.items()}

        # 2) TF-IDF par document (sparse dict) + norme
        # pour chaque document, on garde un dict { term: tf-idf }
        self.doc_tfidf = {}
        #norme du vecteur de doc 
        self.doc_norm = {}

        # Pour itérer doc->(term, tf), on reconstruit en inversant postings
        # doc -> Dic{term, tf}
        doc_terms: Dict[str, Dict[str, int]] = defaultdict(dict)
        # postings = term -> Dic{doc,tf}
        for term, docmap in self.postings.items():
            for d, tf in docmap.items():
                doc_terms[d][term] = tf

        for d, tfmap in doc_terms.items():
            # vec->{terme,TF-IDF}
            vec: Dict[str, float] = {}
            # pondération tf
            for t, tf in tfmap.items():
                # le log limite l'influence des termes répétés sans fin (sublinear)
                if tf_scheme == "log":
                    w_tf = 1.0 + math.log(tf)
                else:  # "raw" (linear)
                    w_tf = float(tf)
                # TF-IDF mesure l’importance d’un mot dans un document par rapport à l’ensemble du corpus.
                vec[t] = w_tf * self.idf.get(t, 0.0)

            # norme L2 du document 
            # w = (weight) poids d'un terme  dans un documents
            norm = math.sqrt(sum(w*w for w in vec.values())) or 1.0

            self.doc_tfidf[d] = vec
            self.doc_norm[d] = norm

    # -------------------TF-IDF des mots de document---------------------------------------------
    #prend les tokens de la requête déjà prétraités et renvoie son vecteur TF-IDF sous forme de dict {terme: poids}
    def _query_tfidf(self, q_tokens: List[str], tf_scheme: str = "log") -> Dict[str, float]:
        # tf de chaque terme dans la requête
        q_tf = Counter(q_tokens)
        q_vec: Dict[str, float] = {}
        for t, tf in q_tf.items():
            # même schéma TF que pour les docs
            if tf_scheme == "log":
                w_tf = 1.0 + math.log(tf)
            else:
                w_tf = float(tf)
            # utiliser idf si connu, sinon 0 (terme out-of-vocab)
            w = w_tf * self.idf.get(t, 0.0)
            if w != 0.0:
                q_vec[t] = w
        return q_vec
    # classer les documents par similarité cosinus avec la requête
    # plus le cosinus est proche de 1 plus le document est pertinent 
    def search_cosine(self, query: str, preproc: TextPreprocessor, use_bigrams: bool = True, top_k: int = 10, tf_scheme: str = "log") -> List[Tuple[str, float]]:
        """
        Recherche vectorielle (TF-IDF, cosinus) sur l'index existant.
        Nécessite d'avoir appelé build_tfidf() après build().
        """
        if not self.doc_tfidf:
            raise RuntimeError("build_tfidf() doit être appelé avant search_cosine().")

        # 1) préparer la requête
        q_tokens = preproc.process(query)
        if use_bigrams:
            q_tokens += make_ngrams(q_tokens, n=2)

        q_vec = self._query_tfidf(q_tokens, tf_scheme=tf_scheme)
        if not q_vec:
            return []
        # la norme L2 c'est la racine de la somme du poids puissance 2 de chaque terme 
        # Vecteur de poids de mots et on calcule sa norme euclidienne 
        # si norme est 0 , on renvoie 1 pour éviter de diviser par  0 dans le cosinus 
        q_norm = math.sqrt(sum(w*w for w in q_vec.values())) or 1.0

        # 2) candidats: union des docs qui contiennent les termes de q
        candidate_docs = set()
        for t in q_vec.keys():
            candidate_docs.update(self.postings.get(t, {}).keys())

        # 3) cosinus = (q·d) / (||q||*||d||)
        scores: Dict[str, float] = {}
        for d in candidate_docs:
            d_vec = self.doc_tfidf[d]
            # produit scalaire sparse
            dot = 0.0
            # itérer sur les termes de la requête (souvent moins nombreux)
            for t, wq in q_vec.items():
                #on vérifie si le terme existe bien dans le document 
                wd = d_vec.get(t)
                if wd is not None:
                    dot += wq * wd
            if dot != 0.0:
                scores[d] = dot / (q_norm * (self.doc_norm.get(d) or 1.0))

        # 4) tri scores -> Dict{Doc,score} 
        # on trie ces paires par score décroissant et on retourne les top_k meilleurs.
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    #objectif : Mettre sur la même échelle des scores venant de sources différentes (ex. BM25 vs cosinus) avant une fusion pondérée.
    def _minmax(self, d: Dict[str, float]) -> Dict[str, float]:
        if not d:
            return {}
        vals = list(d.values())
        mn, mx = min(vals), max(vals)
        if mx == mn:
            return {k: 0.0 for k in d}
        # Le plus petit devient 0.0, le plus grand 1.0, les autres sont linéairement étirés entre les deux.
        return {k: (v - mn) / (mx - mn) for k, v in d.items()}
    
    # k_lex = combien de candidats prendre du canal lexical
    # K_vec = combien de candidats prendre du canal vectoriel
    #fusion par les rang
    def search_hybrid_rrf(self, query: str, preproc: TextPreprocessor, use_bigrams: bool = True, k_lex: int = 200, k_vec: int = 200, top_k: int = 20, rrf_k: int = 60) -> List[Tuple[str, float]]:
        """
        Fusion RRF (Reciprocal Rank Fusion) entre BM25 et cosinus TF-IDF.
        Nécessite d'avoir appelé build_tfidf() au préalable.
        """
        if not self.doc_tfidf:
            raise RuntimeError("build_tfidf() doit être appelé avant search_hybrid_rrf().")

        # 1) résultats BM25 (rangs)
        bm25 = self.search(query, preproc, use_bigrams=use_bigrams, top_k=k_lex)
        # Convertit la liste triée en dictionnaire de rangs : doc_id -> rang (rang commence à 1).
        rank_lex = {doc_id: r for r, (doc_id, _) in enumerate(bm25, start=1)}

        # 2) résultats cosinus (rangs)
        vec = self.search_cosine(query, preproc, use_bigrams=use_bigrams, top_k=k_vec)
        rank_vec = {doc_id: r for r, (doc_id, _) in enumerate(vec, start=1)}

        # 3) RRF
        scores: Dict[str, float] = {}
        for d, r in rank_lex.items():
            scores[d] = scores.get(d, 0.0) + 1.0 / (rrf_k + r)
        for d, r in rank_vec.items():
            scores[d] = scores.get(d, 0.0) + 1.0 / (rrf_k + r)

        # 4) tri final
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    #fusion par les scores normalisés 
    def search_hybrid_interp(self,query: str, preproc: TextPreprocessor, use_bigrams: bool = True, k_lex: int = 200, k_vec: int = 200, top_k: int = 20, alpha: float = 0.6) -> List[Tuple[str, float]]:
        """
        Interpolation pondérée entre scores BM25 et cosinus TF-IDF.
        Normalise min-max chaque canal sur ses candidats avant fusion.
        Nécessite build_tfidf().
        """
        if not self.doc_tfidf:
            raise RuntimeError("build_tfidf() doit être appelé avant search_hybrid_interp().")

        # 1) BM25 (scores)
        bm25 = self.search(query, preproc, use_bigrams=use_bigrams, top_k=k_lex)
        bm25_scores = {d: s for d, s in bm25}

        # 2) Cosinus (scores)
        vec = self.search_cosine(query, preproc, use_bigrams=use_bigrams, top_k=k_vec)
        vec_scores = {d: s for d, s in vec}

        # 3) normalisation min-max
        nb = self._minmax(bm25_scores)
        nv = self._minmax(vec_scores)

        # 4) fusion
        all_docs = set(nb) | set(nv)
        #alpha : poids du canal lexical
        #1-alpha : poids du canal vectoriel
        #fused -> Dict{doc,score}
        fused = {d: alpha * nb.get(d, 0.0) + (1.0 - alpha) * nv.get(d, 0.0) for d in all_docs}

        # 5) tri
        return sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    