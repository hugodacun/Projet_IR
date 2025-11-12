from __future__ import annotations
from src.index import InvertedIndex
from src.preprocess import TextPreprocessor
from src.suggest import Autocomplete
import os

class SearchEngine:
    def __init__(self, docs_dir="data/wiki_split_extract_2k", models_dir="models"):
        self.docs_dir = docs_dir
        self.models_dir = models_dir
        self.preproc = TextPreprocessor(use_stemming=True, keep_digits=True)
        self.index = InvertedIndex()
        self.autocomplete: Autocomplete | None = None

    def build_index(self, use_bigrams: bool = True, rebuild_autocomplete: bool = True):
        print("Construction de l'index…")
        self.index.build(self.docs_dir, self.preproc, use_bigrams=use_bigrams)
        self.index.save(self.models_dir)
        if rebuild_autocomplete:
            self.index.save_edge_index(self.models_dir)
        print("Index créé et sauvegardé.")

    def load(self):
        print("Chargement de l'index…")
        self.index = InvertedIndex.load(f"{self.models_dir}/index.json")
        edge_path = f"{self.models_dir}/edge_index.json"
        if not os.path.exists(edge_path):
            # sécurité si l'edge index n'existe pas encore
            self.index.save_edge_index(self.models_dir)
        self.autocomplete = Autocomplete(edge_path)
        print("Index et suggestions chargés.")

    def search(self, query: str, top_k=10):
        return self.index.search(query, self.preproc, use_bigrams=True, top_k=top_k)

    def suggest(self, prefix: str, top_k=5):
        if not self.autocomplete:
            self.autocomplete = Autocomplete(f"{self.models_dir}/edge_index.json")
        return self.autocomplete.suggest(prefix, top_k)
