from collections import Counter
from src.corpus import CorpusReader
from src.preprocess import TextPreprocessor

def main():
    docs_dir = "data/wiki_split_extract_2k"

    # 1) Charger le corpus
    cr = CorpusReader(docs_dir)
    print("Corpus:", len(cr), "docs | exemples:", cr.list_files(3))

    # 2) Choisir un doc et lire le texte
    doc_id = cr.list_files(1)[0]
    text = cr.read(doc_id)
    print("\nDoc:", doc_id)
    print("Extrait:", text[:250].replace("\n", " "), "...")
 
    # 3) Prétraiter
    tp = TextPreprocessor(use_stemming=True, keep_digits=True)
    tokens = tp.process(text)
    print("\nTokens (20):", tokens[:20])

    # 4) Fréquences (visualisation)
    freq = Counter(tokens)
    print("\nTop 15:")
    ## counter() compte le nombre d'occurence d'un token dans le document (TF) most_common() renvoie les 15 les plus fréquents 
    for token, c in Counter(tokens).most_common(15):
        print(token, c)
if __name__ == "__main__":
    main()
