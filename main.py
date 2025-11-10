from __future__ import annotations
import sys
from pathlib import Path
from src.search import SearchEngine

if __name__ == "__main__":
    eng = SearchEngine()

    if not Path("models/index.json").exists():
        print("-> index absent : construction…")
        eng.build_index(use_bigrams=True, rebuild_autocomplete=True)

    eng.load()

    if len(sys.argv) > 1:
        q = " ".join(sys.argv[1:])
        for i, (doc, score) in enumerate(eng.search(q, top_k=10), 1):
            print(f"{i:2d}. {doc:18s} {score:.4f}")
    else:
        while True:
            q = input("\nQuery (ENTER pour quitter) > ").strip()
            if not q:
                break
            print("Suggestions:", eng.suggest(q, top_k=5))
            for i, (doc, score) in enumerate(eng.search(q, top_k=10), 1):
                print(f"{i:2d}. {doc:18s} {score:.4f}")
