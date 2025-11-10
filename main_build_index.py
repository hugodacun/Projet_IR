from src.preprocess import TextPreprocessor
from src.index import InvertedIndex

def main(): 
    DOCS_DIR = "data/wiki_split_extract_2k"  
    tp = TextPreprocessor(use_stemming=True, keep_digits=True)
    idx = InvertedIndex()
    idx.build(DOCS_DIR, tp, use_bigrams=True)
    idx.save("models")
    #vocab ici représente le nombre de termes distincts qu'on a indexés. 
    print(f"Index: N={idx.N}, avgdl={idx.avgdl:.2f}, vocab={len(idx.df)}")  
    
if __name__ == "__main__":
    main()

