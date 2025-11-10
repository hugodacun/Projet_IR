# src/corpus.py
from __future__ import annotations
import os
from typing import Iterable, Tuple

class CorpusReader:
    def __init__(self):
        self.docs_dir = "data/wiki_split_extract_2k"
        docs_dir = self.docs_dir
        ## trie des fichiers .txt
        self.files = sorted(
            f for f in os.listdir(docs_dir) if f.endswith(".txt")
        )
    ## len() retourne ici le nombre de fichier.txt qu'on a dans docs_dir
    def __len__(self) -> int:
        return len(self.files)
    ## retourne toute la liste des fichiers si limit est None sinon retourne le nombre de fichiers déterminé par limit 
    def list_files(self, limit: int | None = None):
        return self.files if limit is None else self.files[:limit]
 
    def read(self, doc_id: str) -> str:
        path = os.path.join(self.docs_dir, doc_id) 
        ## open retourne un objet du fichier présent dans le path 
        file = open(path, encoding="utf-8")
        try:
            ## lecture du contenu du fichier 
            data = file.read()
        finally:
            file.close()
        return data
    
    def iter_docs(self) -> Iterable[Tuple[str, str]]:
        pairs = []
        for fname in self.files:
            pairs.append((fname, self.read(fname)))  # tuple
        return pairs 