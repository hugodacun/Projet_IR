from src.preprocess import TextPreprocessor
from src.index import InvertedIndex

if __name__ == "__main__":
    DOCS_DIR = "data/wiki_split_extract_2k"   # adapte si besoin
    tp = TextPreprocessor(use_stemming=True, keep_digits=True)
    idx = InvertedIndex()
    idx.build(DOCS_DIR, tp, use_bigrams=True)  # unigrams + bigrams
    idx.save("models")
    print(f"Index: N={idx.N}, avgdl={idx.avgdl:.2f}, vocab={len(idx.df)}")
