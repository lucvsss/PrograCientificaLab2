"""
classifier.py — VerseClassifier: clasifica versículos → libro de la Biblia.
Usa Naive Bayes multinomial (scikit-learn) con entrada TF-IDF propia.
"""
import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tfidf import TFIDFVectorizer
from corpus import BibleCorpus
from preprocessor import TextPreprocessor


class VerseClassifier:
    """
    Clasifica versículos para determinar a qué libro de la Biblia pertenecen.
    
    Usa modelos de machine learning sobre vectores TF-IDF de los versículos.
    
    Modelos disponibles:
        - 'nb':  Naive Bayes Multinomial (rápido, menos preciso en general)
        - 'lr':  Logistic Regression (más lento, típicamente más preciso)
    
    Flujo de trabajo:
        1. Preparar características (matriz TF-IDF ya calculada)
        2. Dividir en entrenamiento (80%) y prueba (20%)
        3. Entrenar modelo seleccionado
        4. Evaluar rendimiento
    """

    def __init__(self, corpus: BibleCorpus,
                 preprocessor: TextPreprocessor,
                 vectorizer: TFIDFVectorizer,
                 model_type: str = 'nb'):
        """
        Inicializa el clasificador.
        
        Args:
            corpus: BibleCorpus con versículos etiquetados
            preprocessor: TextPreprocessor ya construido
            vectorizer: TFIDFVectorizer ya ajustado
            model_type: 'nb' para Naive Bayes, 'lr' para Logistic Regression
        """
        self.corpus = corpus
        self.preprocessor = preprocessor
        self.vectorizer = vectorizer
        self.model_type = model_type
        self.model = None
        
        # Mapeo libro ↔ índice numérico
        self.label_encoder: dict[str, int] = {}      # libro → número
        self.label_decoder: dict[int, str] = {}      # número → libro
        
        # Datos de entrenamiento y prueba
        self.X_train = self.X_test = None
        self.y_train = self.y_test = None
        self.y_pred = None
        
        # Métricas
        self.accuracy = 0.0
        self.book_names_in_train: list[str] = []

    def _encode_labels(self) -> np.ndarray:
        """
        Crea mapeos entre nombres de libros y etiquetas numéricas.
        
        Returns:
            Array NumPy con etiqueta numérica para cada versículo
        """
        # Obtener nombre del libro para cada versículo
        books = [verse.book_name for verse in self.corpus.verses]
        unique_books = sorted(set(books))
        
        # Crear mapeos bidireccionales
        self.label_encoder = {book: index for index, book in enumerate(unique_books)}
        self.label_decoder = {index: book for book, index in self.label_encoder.items()}
        
        # Convertir nombres a etiquetas numéricas
        labels = np.array([self.label_encoder[book] for book in books])
        return labels

    def train(self, test_size: float = 0.2, random_state: int = 42):
        """
        Entrena el clasificador sobre los datos.
        
        Process:
            1. Codificar etiquetas (nombres de libros → números)
            2. Dividir datos: 80% entrenamiento, 20% prueba
            3. Entrenar modelo seleccionado
            4. Evaluar en conjunto de prueba
        
        Args:
            test_size: Proporción de datos para prueba (default 0.2 = 20%)
            random_state: Seed para reproducibilidad
            
        Returns:
            self (para encadenamiento de métodos)
        """
        print(f"[Classifier] Entrenando modelo '{self.model_type}'...")

        # Preparar características y etiquetas
        X_features = self.vectorizer.matrix              # (N_versículos, N_características)
        y_labels = self._encode_labels()                  # (N_versículos,)

        # Dividir en conjuntos de entrenamiento y prueba
        # stratify=y asegura proporciones similares de cada clase
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_features, y_labels, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=y_labels
        )

        # Entrenar modelo seleccionado
        if self.model_type == 'nb':
            # Naive Bayes Multinomial requiere características no negativas
            X_train_nb = np.clip(self.X_train, 0, None)
            X_test_nb = np.clip(self.X_test, 0, None)
            
            self.model = MultinomialNB(alpha=0.1)
            self.model.fit(X_train_nb, self.y_train)
            self.y_pred = self.model.predict(X_test_nb)
        else:
            # Logistic Regression
            self.model = LogisticRegression(
                max_iter=500,
                C=5.0,
                solver='saga',
                multi_class='multinomial',
                random_state=random_state,
                n_jobs=-1  # Usar todos los procesadores disponibles
            )
            self.model.fit(self.X_train, self.y_train)
            self.y_pred = self.model.predict(self.X_test)

        # Calcular precisión
        self.accuracy = accuracy_score(self.y_test, self.y_pred)
        self.book_names_in_train = [
            self.label_decoder[i] for i in sorted(self.label_decoder)
        ]
        
        print(f"[Classifier] Precisión: {self.accuracy:.4f} "
              f"({self.accuracy*100:.2f}%)")
        return self

    def report(self) -> str:
        """Retorna el classification report completo."""
        target_names = self.book_names_in_train
        return classification_report(
            self.y_test, self.y_pred,
            target_names=target_names,
            zero_division=0
        )

    def confusion_matrix(self) -> np.ndarray:
        return confusion_matrix(self.y_test, self.y_pred)

    def predict(self, text: str) -> str:
        """Predice el libro de un texto nuevo."""
        tokens = self.preprocessor.tokens_for_text(text)
        vec = self.vectorizer.transform([tokens])
        if self.model_type == 'nb':
            vec = np.clip(vec, 0, None)
        pred_idx = self.model.predict(vec)[0]
        return self.label_decoder[pred_idx]
