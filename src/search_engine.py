import pandas as pd
import numpy as np

from .tfidf import TFIDFVectorizer, cosine_similarity
from .preprocessing import TextPreprocessor


class SemanticSearchEngine:

    def __init__(self, preprocessor, vectorizer):
        self.preprocessor = preprocessor
        self.vectorizer = vectorizer
        self.matriz_tfidf = None
        self.df_corpus = None 

    def fit(self, df_corpus, columna_tokens="texto_procesado"):
        self.df_corpus = df_corpus.reset_index(drop=True)
        documentos = df_corpus[columna_tokens].tolist()
        self.matriz_tfidf = self.vectorizer.fit_transform(documentos)
        return self

    def buscar(self, query, k=5):
        tokens_query = self.preprocessor.process(query)
        vector_query = self.vectorizer.vectorizar_texto_nuevo(tokens_query)
        resultado = self._rankear(vector_query, k)
        return resultado.reset_index(drop=True)

    def buscar_por_indice(self, idx_versiculo, k=5):
        vector_query = self.matriz_tfidf[idx_versiculo]
        resultado = self._rankear(vector_query, k + 1)
        resultado = resultado[resultado.index != idx_versiculo].head(k)
        return resultado.reset_index(drop=True)

    def _rankear(self, vector_query, k):
        similitudes = np.array([
            cosine_similarity(vector_query, self.matriz_tfidf[i])
            for i in range(self.matriz_tfidf.shape[0])
        ])
        
        top_idx = np.argsort(similitudes)[::-1][:k]
        resultado = self.df_corpus.iloc[top_idx].copy()
        resultado["similitud"] = similitudes[top_idx]

        columnas = [c for c in ["libro", "capitulo", "versiculo", "texto_original", "similitud"]
                    if c in resultado.columns]
        return resultado[columnas]
