"""
preprocessor.py — TextPreprocessor: pipeline de preprocesamiento textual.
Implementado desde cero sin nltk, usando solo regex y estructuras nativas.
"""
import re
import string
from collections import Counter
from corpus import BibleCorpus, Verse


class TextPreprocessor:
    """
    Pipeline de preprocesamiento textual:
      1. Minúsculas
      2. Eliminar puntuación
      3. Eliminar caracteres especiales y números
      4. Tokenización
      5. Eliminación de stopwords
      6. Construcción de vocabulario
      7. Cálculo de frecuencias

    Decisiones de diseño:
    - Se eliminan números porque en la Biblia KJV los números en el texto
      suelen ser referencias (versículos, años) y no aportan contenido semántico.
    - Se usa un conjunto propio de stopwords adaptado al inglés arcaico (KJV):
      incluye palabras como 'thee', 'thou', 'thy', 'ye', 'hath', etc.
    - No se aplica stemming para preservar la legibilidad en las visualizaciones.
    """

    def __init__(self, corpus: BibleCorpus):
        """
        Inicializa el preprocesador de texto.
        
        Atributos:
            corpus: Objeto BibleCorpus que contiene los versículos
            stopwords: Conjunto de palabras comunes a ignorar
            vocab: Diccionario palabra → índice (ordenado por frecuencia)
            freq: Contador de frecuencias de todas las palabras
            processed_verses: Lista de tokens para cada versículo
        """
        self.corpus = corpus
        self.stopwords = corpus.STOPWORDS
        self.vocab: dict[str, int] = {}                    # palabra → índice
        self.freq: Counter = Counter()                     # frecuencias globales
        self.processed_verses: list[list[str]] = []        # tokens por versículo
        self._built = False

    # ─────────────────────────────────────────────
    # Pipeline de preprocesamiento (pasos individuales)
    # ─────────────────────────────────────────────
    
    def to_lower(self, text: str) -> str:
        """Convierte el texto a minúsculas."""
        return text.lower()

    def remove_punctuation(self, text: str) -> str:
        """Elimina signos de puntuación."""
        return text.translate(str.maketrans('', '', string.punctuation))

    def remove_special_and_numbers(self, text: str) -> str:
        """Elimina caracteres especiales y números (solo mantiene letras y espacios)."""
        return re.sub(r'[^a-z\s]', '', text)

    def tokenize(self, text: str) -> list[str]:
        """Divide el texto en palabras (tokens)."""
        return text.split()

    def remove_stopwords(self, tokens: list[str]) -> list[str]:
        """
        Elimina palabras comunes (stopwords) y palabras muy cortas.
        Las palabras muy cortas suelen ser ruido o caracteres sueltos.
        """
        return [word for word in tokens if word not in self.stopwords and len(word) > 1]

    def preprocess_text(self, text: str) -> list[str]:
        """
        Aplica todo el pipeline de preprocesamiento a un texto.
        
        Pasos:
            1. Convertir a minúsculas
            2. Eliminar puntuación
            3. Eliminar caracteres especiales y números
            4. Tokenizar (dividir en palabras)
            5. Eliminar stopwords
        
        Args:
            text: Texto sin procesar
            
        Returns:
            Lista de tokens limpios y preparados
        """
        text = self.to_lower(text)
        text = self.remove_punctuation(text)
        text = self.remove_special_and_numbers(text)
        tokens = self.tokenize(text)
        tokens = self.remove_stopwords(tokens)
        return tokens

    # ─────────────────────────────────────────────
    # Construcción del vocabulario y análisis
    # ─────────────────────────────────────────────
    
    def build(self):
        """
        Preprocesa todos los versículos del corpus y construye:
        - Vocabulario (palabras ordenadas por frecuencia)
        - Matriz de tokens por versículo
        - Contador de frecuencias
        """
        print("[Preprocessor] Preprocesando corpus...")
        
        all_tokens = []
        
        # Procesar cada versículo y recolectar todos los tokens
        for verse in self.corpus.verses:
            tokens = self.preprocess_text(verse.text)
            self.processed_verses.append(tokens)
            all_tokens.extend(tokens)

        # Contar frecuencias
        self.freq = Counter(all_tokens)
        
        # Crear vocabulario ordenado por frecuencia (palabras más comunes primero)
        sorted_words = sorted(self.freq.keys(), key=lambda w: -self.freq[w])
        self.vocab = {word: idx for idx, word in enumerate(sorted_words)}
        
        self._built = True
        
        # Reporte
        unique_words = len(self.vocab)
        total_tokens = len(all_tokens)
        print(f"[Preprocessor] Vocabulario: {unique_words:,} palabras únicas | "
              f"Tokens totales: {total_tokens:,}")

    def get_top_words(self, n: int = 30) -> list[tuple[str, int]]:
        """Retorna las n palabras más frecuentes en el corpus."""
        return self.freq.most_common(n)

    def get_verse_tokens(self, verse_index: int) -> list[str]:
        """
        Obtiene los tokens preprocesados de un versículo específico.
        
        Args:
            verse_index: Índice del versículo en el corpus
            
        Returns:
            Lista de tokens (palabras procesadas)
        """
        if not self._built:
            raise RuntimeError("Debe llamar a build() primero.")
        return self.processed_verses[verse_index]

    def tokens_for_text(self, text: str) -> list[str]:
        """
        Preprocesa un texto externo (por ejemplo, un query de búsqueda).
        Útil para procesar texto que no forma parte del corpus.
        
        Args:
            text: Texto a procesar
            
        Returns:
            Lista de tokens procesados
        """
        return self.preprocess_text(text)
