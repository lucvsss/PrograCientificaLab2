import re
import string
from collections import Counter
from typing import List, Dict, Iterable, Optional

class TextPreprocessor:
    def __init__(
        self,
        stopwords: Optional[Iterable[str]] = None,
        min_token_len: int = 2,
        lowercase: bool = True,
        remove_punctuation: bool = True,
        remove_numbers: bool = True,
        remove_stopwords: bool = True
    ):
        
        self.stopwords = set(stopwords) if stopwords is not None else set()
        self.min_token_len = min_token_len
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        self.remove_stopwords = remove_stopwords
        self.vocabulario: Dict[str, int] = {}
        self.frecuencias: Counter = Counter()

    def to_lowercase(self, text: str) -> str:
        return text.lower() if self.lowercase else text

    def strip_punctuation(self, text: str) -> str:
        if not self.remove_punctuation:
            return text

        punct = string.punctuation + "¿¡“”‘’—–"
        return text.translate(str.maketrans(punct, " " * len(punct)))

    def strip_special_and_numbers(self, text: str) -> str:
        if not self.remove_numbers:
            return text

        return re.sub(r"[^a-zA-ZáéíóúüñÁÉÍÓÚÜÑ\s]", " ", text)

    def tokenize(self, text: str) -> List[str]:
        return text.split()

    def filter_stopwords(self, tokens: List[str]) -> List[str]:
        if not self.remove_stopwords:
            return tokens

        return [t for t in tokens if t not in self.stopwords]

    def filter_short_tokens(self, tokens: List[str]) -> List[str]:
        return [t for t in tokens if len(t) >= self.min_token_len]

    def process(self, text: str) -> List[str]:
        text = self.to_lowercase(text)
        text = self.strip_punctuation(text)
        text = self.strip_special_and_numbers(text)
        tokens = self.tokenize(text)
        tokens = self.filter_stopwords(tokens)
        tokens = self.filter_short_tokens(tokens)
        return tokens
    
    def process_ngram(self, text: str) -> List[str]:
        text = self.to_lowercase(text)
        text = self.strip_punctuation(text)
        text = self.strip_special_and_numbers(text)
        tokens = self.tokenize(text)
        tokens = self.filter_short_tokens(tokens)
        return tokens

    def process_corpus(self, textos: List[str]) -> List[List[str]]:
        resultado = []
        for texto in textos:
            tokens = self.process(texto)
            resultado.append(tokens)
            self.frecuencias.update(tokens)
        self.vocabulario = {palabra: idx for idx, palabra in enumerate(sorted(self.frecuencias))}
        return resultado
    
    def process_corpus_ngram(self, textos: List[str]) -> List[List[str]]:
        resultado = []
        for texto in textos:
            tokens = self.process_ngram(texto)
            resultado.append(tokens)
            self.frecuencias.update(tokens)
        self.vocabulario = {palabra: idx for idx, palabra in enumerate(sorted(self.frecuencias))}
        return resultado

    def palabras_mas_frecuentes(self, n: int = 20):
        return self.frecuencias.most_common(n)
