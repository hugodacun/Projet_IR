from __future__ import annotations
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
import json, os, math

from src.index import InvertedIndex
from src.preprocess import TextPreprocessor
from src.suggest import Autocomplete

class SearchEngine:
    """
    Classe principale qui combine :
      - prétraitement
      - indexation
      - recherche BM25
      - suggestion (autocomplete)
    """
    def __init__(self, docs_dir="data/wiki_split_extract_2k", models_dir="models"):
        self.docs_dir = docs_dir
        self.models_dir = models_dir
        self.preproc = TextPreprocessor(use_stemming=True, keep_digits=True)
        self.index = InvertedIndex()
        self.autocomplete = None

    def build_index(self):
        print("Construction de l'index...")
        self.index.build(self.docs_dir, self.preproc, use_bigrams=True)
        self.index.save(self.models_dir)
        self.index.save_edge_index(self.models_dir)
        print("Index créé et sauvegardé.")

    def load(self):
        print("Chargement de l'index...")
        self.index = InvertedIndex.load(f"{self.models_dir}/index.json")
        self.autocomplete = Autocomplete(f"{self.models_dir}/edge_index.json")
        print("Index et suggestions chargés.")

    def search(self, query: str, top_k=10):
        print(f"Recherche : '{query}'")
        results = self.index.search(query, self.preproc, use_bigrams=True, top_k=top_k)
        return results

    def suggest(self, prefix: str, top_k=5):
        if not self.autocomplete:
            self.autocomplete = Autocomplete(f"{self.models_dir}/edge_index.json")
        return self.autocomplete.suggest(prefix, top_k)