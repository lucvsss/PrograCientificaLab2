"""
tfidf.py — TFIDFVectorizer: implementación propia de TF-IDF.
No se usa sklearn ni ninguna librería que lo implemente.
"""
import math
import numpy as np
from collections import Counter
from preprocessor import TextPreprocessor


class TFIDFVectorizer:
    """
    Implementación manual de TF-IDF (Term Frequency – Inverse Document Frequency).

    TF(t, d)  = frecuencia del término t en el documento d / total términos en d
    IDF(t)    = log( (N + 1) / (df(t) + 1) ) + 1   [suavizado]
    TF-IDF(t,d) = TF(t,d) × IDF(t)

    Los vectores resultantes se normalizan L2 para facilitar la similitud del coseno.
    """

    def __init__(self, preprocessor: TextPreprocessor, max_features: int = 5000):
        """
        Inicializa el vectorizador TF-IDF.
        
        Args:
            preprocessor: TextPreprocessor ya construido
            max_features: Número máximo de características (palabras) a considerar.
                         Se seleccionan las palabras con mayor frecuencia de documento.
        
        Atributos:
            vocabulary: Lista de palabras seleccionadas
            word2idx: Mapeo palabra → índice de columna
            idf: Vector de valores IDF para cada palabra
            matrix: Matriz TF-IDF de documentos × vocabulario
        """
        self.preprocessor = preprocessor
        self.max_features = max_features
        self.vocabulary: list[str] = []          # palabras seleccionadas
        self.word2idx: dict[str, int] = {}       # palabra → índice de columna
        self.idf: np.ndarray = np.array([])      # vector de valores IDF
        self.matrix: np.ndarray = np.array([])   # matriz TF-IDF (documentos × vocabulario)
        self._fitted = False

    # ──────────────────────────────────────────────
    # Métodos internos (helpers)
    # ──────────────────────────────────────────────
    
    def _compute_tf(self, tokens: list[str]) -> dict[str, float]:
        """
        Calcula la frecuencia relativa de términos (Term Frequency).
        
        TF(término) = (frecuencia del término en el documento) / (total de términos)
        
        Args:
            tokens: Lista de tokens del documento
            
        Returns:
            Diccionario término → valor TF (entre 0 y 1)
        """
        total_tokens = len(tokens)
        if total_tokens == 0:
            return {}
        
        token_counts = Counter(tokens)
        return {word: count / total_tokens for word, count in token_counts.items()}

    def _build_vocabulary(self, all_token_lists: list[list[str]]):
        """
        Construye el vocabulario seleccionando las palabras más frecuentes.
        
        Criterio: Se seleccionan las max_features palabras con mayor frecuencia
        de documento (DF = número de documentos que contienen la palabra).
        
        Args:
            all_token_lists: Lista de listas de tokens (un documento por elemento)
            
        Returns:
            Counter con las frecuencias de documento
        """
        document_frequency = Counter()
        
        # Contar en cuántos documentos aparece cada palabra
        for tokens in all_token_lists:
            unique_words = set(tokens)
            for word in unique_words:
                document_frequency[word] += 1
        
        # Seleccionar las max_features palabras con mayor DF
        selected_words = [word for word, _ in document_frequency.most_common(self.max_features)]
        self.vocabulary = selected_words
        self.word2idx = {word: index for index, word in enumerate(selected_words)}
        
        return document_frequency

    def _compute_idf(self, document_frequency: Counter, num_documents: int) -> np.ndarray:
        """
        Calcula los valores Inverse Document Frequency (IDF).
        
        Fórmula: IDF(término) = log((N + 1) / (DF + 1)) + 1
        - N = número total de documentos
        - DF = número de documentos que contienen el término
        - Se usa suavizado para evitar log(0)
        
        Args:
            document_frequency: Counter con frecuencias de documento
            num_documents: Total de documentos en el corpus
            
        Returns:
            Array NumPy con valores IDF para cada palabra del vocabulario
        """
        idf_values = np.zeros(len(self.vocabulary))
        
        for index, word in enumerate(self.vocabulary):
            df = document_frequency.get(word, 0)
            idf_values[index] = math.log((num_documents + 1) / (df + 1)) + 1
        
        return idf_values

    # ──────────────────────────────────────────────
    # API Pública
    # ──────────────────────────────────────────────
    
    def fit_transform(self, token_lists: list[list[str]] = None) -> np.ndarray:
        """
        Ajusta el vectorizador y transforma documentos a matriz TF-IDF.
        
        Proceso:
            1. Construire vocabulario (palabras más frecuentes)
            2. Calcular valores IDF
            3. Para cada documento: calcular vector TF-IDF
            4. Normalizar vectores con norma L2
        
        Args:
            token_lists: Lista de listas de tokens. Si es None, usa el corpus completo
                        del preprocesador.
        
        Returns:
            Matriz NumPy de forma (num_documentos, num_palabras)
            Cada fila es un documento vectorizado y normalizado.
        """
        if token_lists is None:
            token_lists = self.preprocessor.processed_verses

        num_documents = len(token_lists)
        print(f"[TF-IDF] Ajustando sobre {num_documents:,} documentos...")

        # Construir vocabulario e IDF
        document_frequency = self._build_vocabulary(token_lists)
        self.idf = self._compute_idf(document_frequency, num_documents)

        # Crear matriz TF-IDF
        vocabulary_size = len(self.vocabulary)
        tfidf_matrix = np.zeros((num_documents, vocabulary_size), dtype=np.float32)

        for doc_index, tokens in enumerate(token_lists):
            tf_values = self._compute_tf(tokens)
            
            for word, tf_value in tf_values.items():
                if word in self.word2idx:
                    col_index = self.word2idx[word]
                    tfidf_matrix[doc_index, col_index] = tf_value * self.idf[col_index]

        # Normalizar cada fila (documento) con norma L2
        row_norms = np.linalg.norm(tfidf_matrix, axis=1, keepdims=True)
        row_norms[row_norms == 0] = 1.0  # Evitar división por cero
        tfidf_matrix /= row_norms

        self.matrix = tfidf_matrix
        self._fitted = True
        
        print(f"[TF-IDF] Matriz: {tfidf_matrix.shape[0]:,} × {tfidf_matrix.shape[1]:,}")
        return tfidf_matrix

    def transform(self, token_lists: list[list[str]]) -> np.ndarray:
        """
        Transforma nuevos documentos usando el vocabulario e IDF ya ajustados.
        
        Método para usar después de fit_transform. Aplica el mismo proceso pero
        sobre documentos nuevos, sin recalcular IDF.
        
        Args:
            token_lists: Nuevos documentos a vectorizar
            
        Returns:
            Matriz TF-IDF de los nuevos documentos
        """
        if not self._fitted:
            raise RuntimeError("Debe llamar a fit_transform() primero.")
        
        num_documents = len(token_lists)
        vocabulary_size = len(self.vocabulary)
        tfidf_matrix = np.zeros((num_documents, vocabulary_size), dtype=np.float32)
        
        for doc_index, tokens in enumerate(token_lists):
            tf_values = self._compute_tf(tokens)
            for word, tf_value in tf_values.items():
                if word in self.word2idx:
                    col_index = self.word2idx[word]
                    tfidf_matrix[doc_index, col_index] = tf_value * self.idf[col_index]
        
        # Normalizar cada fila
        row_norms = np.linalg.norm(tfidf_matrix, axis=1, keepdims=True)
        row_norms[row_norms == 0] = 1.0
        tfidf_matrix /= row_norms
        
        return tfidf_matrix

    def get_book_vectors(self) -> tuple[list[str], np.ndarray]:
        """
        Agrega vectores TF-IDF por libro (promedios de versículos).
        
        Para cada libro, calcula un vector que es el promedio de los vectores
        TF-IDF de todos sus versículos. Útil para análisis de similitud entre libros.
        
        Returns:
            Tupla (nombres_libros, matriz_vectores_libros)
            - nombres_libros: Lista de nombres de libros
            - matriz_vectores_libros: Array NumPy de shape (num_libros, num_palabras)
        """
        if not self._fitted:
            raise RuntimeError("Debe llamar a fit_transform() primero.")
        
        corpus = self.preprocessor.corpus
        book_names = corpus.get_book_names()
        
        # Inicializar acumuladores
        book_vectors = np.zeros((len(book_names), len(self.vocabulary)), dtype=np.float32)
        book_verse_counts = np.zeros(len(book_names), dtype=int)

        # Mapear nombre de libro → índice
        book_name_to_index = {name: index for index, name in enumerate(book_names)}

        # Acumular vectores de versículos por libro
        for verse_index, verse in enumerate(corpus.verses):
            book_index = book_name_to_index.get(verse.book_name)
            if book_index is not None:
                book_vectors[book_index] += self.matrix[verse_index]
                book_verse_counts[book_index] += 1

        # Promediar (dividir por número de versículos por libro)
        safe_counts = np.where(book_verse_counts > 0, book_verse_counts, 1).reshape(-1, 1)
        book_vectors /= safe_counts
        
        # Re-normalizar con norma L2
        row_norms = np.linalg.norm(book_vectors, axis=1, keepdims=True)
        row_norms[row_norms == 0] = 1.0
        book_vectors /= row_norms

        return book_names, book_vectors
