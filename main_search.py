# main_search.py
import sys
from src.preprocess import TextPreprocessor
from src.index import InvertedIndex

def main():
    query = " ".join(sys.argv[1:]) or "chateau gaillard"  # requête par Ligne de commande 
    tp = TextPreprocessor(use_stemming=True, keep_digits=True)
    idx = InvertedIndex.load("models/index.json")  # charge l'index sauvegardé
    # search() renvoie liste de tuples de (doc_id, score), que 10 docs 
    results = idx.search(query, tp, use_bigrams=True, top_k=13)
    #enumerate() parcourt cette liste de docs et aussi définit le rang r qui commence par 1 
    for r, (doc, score) in enumerate(results, 1):
        # affichage du rang sur deux caractère (2d)
        # affichage du nom de document sur au moins 18 caractères (18s)
        # affichage du score en flottant avec 4 décimales (4f)
        print(f"{r:2d}. {doc:18s} {score:.4f}")

if __name__ == "__main__":
    main() 