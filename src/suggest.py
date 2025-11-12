from __future__ import annotations
import json
from bisect import bisect_left

class Autocomplete:
    """
    Lit models/edge_index.json et propose des termes du vocabulaire
    par préfixe. Ici on se base sur la liste des mots (clés du JSON).
    """
    def __init__(self, edge_index_path: str = "models/edge_index.json"):
        with open(edge_index_path, encoding="utf-8") as f:
            data = json.load(f)
        self.edge_index: dict[str, list[str]] = data["edge_index"]
        self.words = sorted(self.edge_index.keys())

    def suggest(self, prefix: str, top_k: int = 5) -> list[str]:
        prefix = prefix.lower().strip()
        if len(prefix) < 2:
            return []
        i = bisect_left(self.words, prefix)
        out = []
        for w in self.words[i:]:
            if w.startswith(prefix):
                out.append(w)
                if len(out) >= top_k:
                    break
            else:
                if len(w) < len(prefix) or w[:len(prefix)] > prefix:
                    break
        return out
