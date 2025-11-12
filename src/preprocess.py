from __future__ import annotations
import re  ## utile pour les expressions régulières
import unicodedata ## pour normaliser Unicode ( retirer les accents proprement )
from typing import List ## annotation de type List

class TextPreprocessor: 
    STOP_WORDS = """ au aux avec ce ces dans de des du elle en et eux il je la le les leur lui ma mais me
    meme nos notre nous on ou par pas pour qu que qui sa se ses son sur ta te tes toi
    tu un une vos votre vous c d j l a y ete etes etant est et auj hui """
    ## on mais tous les stop words sous forme de list avec split ["c","d"...]
    ## set pour enlever les doublons dans la liste 
    STOP_FR_WORDS = set(STOP_WORDS.split())
    ## après normalisation, on a cette regexp qui doit etre respecté par les token 
    TOKEN_REGEXP = re.compile("[A-Za-z0-9]+") 
    ## deux options pour le prétraitement : soit activer/désactiver stemming , soit garder:supprimer nombers(digits)
    def __init__(self, use_stemming : bool = True, keep_digits : bool = True):
     self.use_stemming = use_stemming 
     self.keep_digits = keep_digits 
    ## sépare les lettres de leurs accents per exemple élève => e'le've => e l e v e => eleve 
    def _strip_accents(self, s: str) -> str:
     return "".join( 
            c for c in unicodedata.normalize("NFKD", s)
            if unicodedata.category(c) != "Mn"
        ) 
    ## on transforme les mots en miniscule et enlever les accents 
    def normalize(self, text: str) -> str:
      return self._strip_accents(text.lower())
    ## annotation List[str] 
    def tokenize(self, text: str) -> List[str]:
        ## renvoie les tokens (les mots ou chiffres qui réspectent le regexp)
        tokens = self.TOKEN_REGEXP.findall(text)  
        ## si keep_digit est false , donc on supprime les nombre et on garde que les mots 
        if not self.keep_digits:
            tokens = [t for t in tokens if not t.isdigit()]
        return tokens
    ## filtre : enlever les stopwords s'il y'en a  et aussi les mots qui contient 1 seule caractère. 
    def filter_tokens(self, tokens: List[str]) -> List[str]:
        return [t for t in tokens if t not in self.STOP_FR_WORDS and len(t) > 1]
    
    def stem(self, toks: List[str]) -> List[str]:
        # Si le stemming est désactivé, on renvoie tel quel
        if not self.use_stemming:
            return toks

        out: List[str] = []
        for t in toks:
            # règle: si mot finit par 's' et longueur > 3, on retire le 's'
            if t.endswith("s") and len(t) > 3: 
             out.append(t[:-1])  
            else:
                out.append(t)
        return out
    def process(self, text: str) -> List[str]:
        # Stratégie : normaliser → tokeniser → filtrer → stem
        text = self.normalize(text) ## lower case, supprime les accents 
        tokens = self.tokenize(text) ## on sépare les phrases sous fomre de tokens 
        tokens = self.filter_tokens(tokens) ## supprime les stopwords et mots de un caractère 
        tokens = self.stem(tokens) ## enleve les s du pluriel 
        return tokens
    
# --- N-grams-----------------------------------------------------------
## on construit des ngrams à partir de la liste des tokens génére après traitement 
def make_ngrams(tokens: list[str], n: int = 2) -> list[str]:
    if n < 2 or len(tokens) < n:
        return []
    return ["_".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

#def make_edge_ngrams(word: str) -> list[str]:
    #"""
    #Génère les préfixes successifs (edge n-grams) d'un mot.
    #Ex: "ecologie" -> ["eco", "ecolog"]
    #"""
   # word = word.lower()
  #  return [word[:i] for i in range(3, len(word),3)]

def make_edge_ngrams(word: str, min_len: int = 3, max_len: int | None = None) -> list[str]:
    word = word.lower()
    # si la longueur du mots n'est pas spécifié on prend la longueur du du mots 
    if max_len is None:
        max_len = len(word)
    max_len = min(max_len, len(word))
    # list[str]: A list of edge n-grams as lowercase strings. Returns an empty list if the word is shorter than min_len.
    if len(word) < min_len:
        return []
    return [word[:L] for L in range(min_len, max_len + 1)]