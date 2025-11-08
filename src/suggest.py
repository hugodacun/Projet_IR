import json
from bisect import bisect_left
from src.index import InvertedIndex


class Autocomplete:
    """
    Gère les suggestions à partir de l'edge_index.json.
    Permet de proposer des mots au fur et à mesure que l'utilisateur tape.
    """
    def __init__(self, edge_index_path="models/edge_index.json"):
        self.edge_index = InvertedIndex().load_edge_index() 
        self.words = sorted(self.edge_index.keys())

    def suggest(self, prefix: str, top_k: int = 5):
        prefix = prefix.lower().strip()
        if len(prefix) < 2:
            return []

        i = bisect_left(self.words, prefix)
        suggestions = []
        for w in self.words[i:]:
            if w.startswith(prefix):
                suggestions.append(w)
                if len(suggestions) >= top_k:
                    break
            else:
                if len(w) < len(prefix) or w[:len(prefix)] > prefix:
                    break
        return suggestions


if __name__ == "__main__":
    ac = Autocomplete()

    while True:
        prefix = input("Tape un préfixe (ou 'quit' pour sortir) > ").strip()
        if prefix.lower() == "quit":
            break

        suggestions = ac.suggest(prefix, top_k=10)
        print(f"Suggestions pour '{prefix}': {suggestions}\n")