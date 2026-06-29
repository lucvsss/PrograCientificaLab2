"""
search_engine.py — SearchEngine: motor de búsqueda semántico basado en TF-IDF + coseno.
La similitud del coseno está implementada manualmente.
"""
import numpy as np
import pandas as pd
from tfidf import TFIDFVectorizer
from preprocessor import TextPreprocessor
from corpus import BibleCorpus


class SearchEngine:
    """
    Motor de búsqueda semántico basado en similitud TF-IDF.
    
    Permite buscar versículos similares a:
        - Consultas en lenguaje free (texto libre)
        - Versículos específicos del corpus
    
    Implementa similitud coseno manualmente (sin usar sklearn).
    """

    def __init__(self, corpus: BibleCorpus,
                 preprocessor: TextPreprocessor,
                 vectorizer: TFIDFVectorizer):
        """
        Inicializa el motor de búsqueda.
        
        Args:
            corpus: BibleCorpus con los versículos
            preprocessor: TextPreprocessor ya construido
            vectorizer: TFIDFVectorizer ya ajustado (con matriz)
        """
        self.corpus = corpus
        self.preprocessor = preprocessor
        self.vectorizer = vectorizer

    # ──────────────────────────────────────────────
    # Similitud del coseno - Implementación manual
    # ──────────────────────────────────────────────
    
    @staticmethod
    def cosine_similarity_vector(query_vector: np.ndarray,
                                 document_matrix: np.ndarray) -> np.ndarray:
        """
        Calcula similitud coseno entre un vector query y cada fila de la matriz.
        
        Como ambos vectores están normalizados L2, la similitud coseno es
        simplemente el producto punto: cos(θ) = query · documento
        
        Args:
            query_vector: Vector del query (normalizado, shape: (vocabulario,))
            document_matrix: Matriz de documentos (normalizada, shape: (N_docs, vocabulario))
            
        Returns:
            Array con similitudes para cada documento (shape: (N_docs,))
        """
        return document_matrix @ query_vector  # Producto punto

    @staticmethod
    def cosine_similarity_matrix(matrix_a: np.ndarray,
                                 matrix_b: np.ndarray) -> np.ndarray:
        """
        Calcula matriz de similitudes coseno entre dos conjuntos de documentos.
        
        Args:
            matrix_a: Primera matriz de documentos (shape: (N, dimensión))
            matrix_b: Segunda matriz de documentos (shape: (M, dimensión))
            
        Returns:
            Matriz de similitudes (shape: (N, M))
        """
        return matrix_a @ matrix_b.T  # Producto punto entre matrices

    # ──────────────────────────────────────────────
    # Búsqueda
    # ──────────────────────────────────────────────
    
    def search(self, query: str, k: int = 10) -> pd.DataFrame:
        """
        Busca los K versículos más similares a una consulta en lenguaje libre.
        
        Proceso:
            1. Procesar consulta (preprocesamiento)
            2. Vectorizar con TF-IDF
            3. Calcular similitud coseno con todo el corpus
            4. Retornar top K resultados
        
        Args:
            query: Texto de búsqueda (ej: "love thy neighbor")
            k: Número de resultados a retornar
            
        Returns:
            DataFrame con columnas: rank, book, chapter, verse, text, similarity
        """
        # Preprocesar consulta
        query_tokens = self.preprocessor.tokens_for_text(query)
        if not query_tokens:
            # Retornar DataFrame vacío si el query no generó tokens
            return pd.DataFrame(columns=['rank', 'book', 'chapter',
                                         'verse', 'text', 'similarity'])

        # Vectorizar query
        query_vector = self.vectorizer.transform([query_tokens])[0]  # shape: (vocab,)

        # Calcular similitud coseno con todos los versículos
        similarities = self.cosine_similarity_vector(query_vector,
                                                     self.vectorizer.matrix)

        # Obtener índices de los K más similares
        top_k_indices = np.argsort(similarities)[::-1][:k]
        
        results = []
        for rank, verse_index in enumerate(top_k_indices, 1):
            verse = self.corpus.verses[verse_index]
            results.append({
                'rank': rank,
                'book': verse.book_name,
                'chapter': verse.chapter,
                'verse': verse.verse_num,
                'text': verse.text,
                'similarity': round(float(similarities[verse_index]), 4)
            })
        
        return pd.DataFrame(results)

    def search_by_verse(self, book_name: str, chapter: int,
                        verse_num: int, k: int = 10) -> pd.DataFrame:
        """
        Busca versículos similares a uno específico del corpus.
        
        Args:
            book_name: Nombre del libro (ej: "Genesis")
            chapter: Número del capítulo
            verse_num: Número del versículo
            k: Número de resultados a retornar (no incluye el versículo original)
            
        Returns:
            DataFrame con los K versículos más similares
        """
        # Encontrar el índice del versículo especificado
        for verse_index, verse in enumerate(self.corpus.verses):
            if (verse.book_name == book_name and
                    verse.chapter == chapter and 
                    verse.verse_num == verse_num):
                
                # Obtener vector del versículo
                query_vector = self.vectorizer.matrix[verse_index]
                
                # Calcular similitudes
                similarities = self.cosine_similarity_vector(
                    query_vector, self.vectorizer.matrix)
                
                # Obtener top K+1 (para excluir el versículo mismo)
                top_k_indices = np.argsort(similarities)[::-1][:k + 1]
                
                results = []
                for verse_idx in top_k_indices:
                    if verse_idx == verse_index:
                        continue  # Saltar el versículo mismo
                    
                    similar_verse = self.corpus.verses[verse_idx]
                    results.append({
                        'rank': len(results) + 1,
                        'book': similar_verse.book_name,
                        'chapter': similar_verse.chapter,
                        'verse': similar_verse.verse_num,
                        'text': similar_verse.text,
                        'similarity': round(float(similarities[verse_idx]), 4)
                    })
                    if len(results) == k:
                        break
                return pd.DataFrame(results)
        raise ValueError(f"Versículo no encontrado: {book_name} {chapter}:{verse_num}")
