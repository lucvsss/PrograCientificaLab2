"""
sentiment.py — SentimentAnalyzer: análisis de sentimiento sobre el corpus bíblico.
Implementa un analizador léxico propio basado en un lexicón de polaridad,
sin usar librerías externas de NLP como VADER o TextBlob.
"""
import numpy as np
import pandas as pd
from corpus import BibleCorpus
from preprocessor import TextPreprocessor


class SentimentAnalyzer:
    """
    Analizador de sentimiento basado en lexicón de polaridad.

    Estrategia:
      - Cada palabra tiene una puntuación de sentimiento en [-1, +1]
      - El puntaje del versículo es la media de sus palabras con puntaje conocido
      - Se agrega por capítulo y por libro

    Lexicón incluido: subconjunto representativo de palabras con carga emocional
    relevante para el corpus bíblico (inglés KJV).

    Limitaciones discutidas en el análisis:
      - Vocabulario arcaico (KJV) puede no coincidir con lexicones modernos
      - Palabras con doble sentido (ej. "judgment" puede ser positivo o negativo)
      - Metáforas y alegorías pierden la carga semántica literal
      - La negación no se procesa (falta de contexto)
    """

    # Lexicón de polaridad básico para inglés bíblico
    LEXICON: dict[str, float] = {
        # Muy positivos (+0.8 a +1.0)
        'love': 1.0, 'glory': 0.9, 'joy': 1.0, 'peace': 0.9, 'blessed': 1.0,
        'blessing': 0.9, 'salvation': 1.0, 'holy': 0.8, 'grace': 0.9,
        'merciful': 0.9, 'mercy': 0.8, 'righteous': 0.8, 'righteousness': 0.8,
        'praise': 0.9, 'wonderful': 0.9, 'good': 0.7, 'goodness': 0.8,
        'faithful': 0.8, 'truth': 0.7, 'light': 0.6, 'heal': 0.8,
        'healed': 0.8, 'healing': 0.8, 'rejoice': 0.9, 'rejoicing': 0.9,
        'comfort': 0.8, 'hope': 0.7, 'thankful': 0.8, 'thanks': 0.6,
        'victory': 0.8, 'life': 0.6, 'perfect': 0.7, 'sing': 0.7,
        'salvation': 0.9, 'savior': 0.9, 'heaven': 0.8, 'paradise': 0.9,
        'eternal': 0.6, 'everlasting': 0.6, 'abundant': 0.7, 'prosper': 0.7,
        'prosperity': 0.7, 'gift': 0.7, 'gifts': 0.7, 'strength': 0.6,
        'wise': 0.6, 'wisdom': 0.7, 'pure': 0.7, 'beautiful': 0.7,
        'friend': 0.6, 'friendship': 0.6, 'honor': 0.7, 'honourable': 0.7,
        'beloved': 0.8, 'compassion': 0.8, 'kind': 0.6, 'kindness': 0.7,
        'gentle': 0.7, 'forgive': 0.8, 'forgiveness': 0.8, 'redeem': 0.8,
        'redemption': 0.8, 'glory': 0.9,
        # Moderadamente positivos (+0.3 a +0.7)
        'trust': 0.5, 'faith': 0.6, 'believe': 0.5, 'prayer': 0.5,
        'pray': 0.5, 'covenant': 0.4, 'promise': 0.5, 'remember': 0.3,
        'help': 0.5, 'helper': 0.5, 'protect': 0.5, 'protection': 0.5,
        'safe': 0.5, 'safety': 0.5, 'righteous': 0.7, 'holy': 0.7,
        'obey': 0.4, 'obedience': 0.4, 'justice': 0.5,
        # Neutros / ligeramente positivos
        'god': 0.2, 'lord': 0.2, 'king': 0.1, 'servant': 0.1,
        # Moderadamente negativos (-0.3 a -0.7)
        'sin': -0.7, 'sinner': -0.7, 'evil': -0.8, 'wicked': -0.8,
        'wickedness': -0.8, 'iniquity': -0.7, 'transgression': -0.7,
        'fear': -0.4, 'afraid': -0.5, 'trouble': -0.5, 'troubled': -0.5,
        'suffer': -0.6, 'suffering': -0.6, 'pain': -0.7, 'sorrow': -0.7,
        'sorrowful': -0.7, 'weep': -0.6, 'weeping': -0.6, 'mourn': -0.6,
        'mourning': -0.6, 'enemy': -0.6, 'enemies': -0.6, 'hate': -0.8,
        'hatred': -0.8, 'curse': -0.7, 'cursed': -0.7, 'shame': -0.6,
        'ashamed': -0.6, 'anger': -0.6, 'angry': -0.6, 'wrath': -0.8,
        # Muy negativos (-0.8 a -1.0)
        'death': -0.8, 'die': -0.7, 'dead': -0.7, 'kill': -0.9,
        'destroy': -0.8, 'destruction': -0.8, 'darkness': -0.6,
        'plague': -0.8, 'famine': -0.8, 'sword': -0.5, 'war': -0.6,
        'battle': -0.4, 'blood': -0.5, 'slaughter': -0.9, 'fire': -0.5,
        'abomination': -0.9, 'perish': -0.8, 'hell': -0.9, 'damnation': -1.0,
        'condemned': -0.8, 'judgment': -0.4, 'punish': -0.7,
        'punishment': -0.7, 'wound': -0.6, 'wounded': -0.6, 'desolate': -0.7,
        'desolation': -0.7, 'lament': -0.6, 'lamentation': -0.6,
        'misery': -0.8, 'affliction': -0.7, 'afflicted': -0.7,
        'oppression': -0.7, 'oppressed': -0.7, 'captive': -0.6,
        'captivity': -0.6, 'prison': -0.5, 'groan': -0.6,
    }

    def __init__(self, corpus: BibleCorpus, preprocessor: TextPreprocessor):
        """
        Inicializa el analizador de sentimiento.
        
        Args:
            corpus: BibleCorpus de entrada
            preprocessor: TextPreprocessor ya construido
        """
        self.corpus = corpus
        self.preprocessor = preprocessor
        self.verse_sentiment_scores: list[float] = []
        self.df_results: pd.DataFrame = pd.DataFrame()

    def _score_verse(self, tokens: list[str]) -> float:
        """
        Calcula la puntuación de sentimiento de un versículo.
        
        Estrategia: Busca palabras con polaridad conocida (lexicón) y
        promediar sus valores. Si no hay palabras conocidas, retorna 0.
        
        Args:
            tokens: Lista de palabras (procesadas) del versículo
            
        Returns:
            Puntuación promedio de sentimiento [-1, +1]
        """
        sentiments = [self.LEXICON[word] for word in tokens if word in self.LEXICON]
        if not sentiments:
            return 0.0
        return float(np.mean(sentiments))

    def analyze(self):
        """
        Analiza el sentimiento de todos los versículos del corpus.
        
        Para cada versículo:
            1. Obtener sus tokens preprocesados
            2. Calcular sentimiento
            3. Registrar información: libro, capítulo, versículo
        
        Returns:
            self (para encadenamiento de métodos)
        """
        print("[Sentiment] Analizando sentimiento...")
        
        sentiment_records = []
        for verse_index, verse in enumerate(self.corpus.verses):
            tokens = self.preprocessor.processed_verses[verse_index]
            sentiment_score = self._score_verse(tokens)
            sentiment_records.append({
                'book_id': verse.book_id,
                'book_name': verse.book_name,
                'testament': verse.testament,
                'chapter': verse.chapter,
                'verse_num': verse.verse_num,
                'sentiment': sentiment_score
            })

        self.df_results = pd.DataFrame(sentiment_records)
        print(f"[Sentiment] Puntajes calculados para {len(sentiment_records):,} versículos.")
        return self

    def by_book(self) -> pd.DataFrame:
        """
        Agrega el análisis de sentimiento por libro.
        
        Para cada libro calcula:
            - avg_sentiment: Sentimiento promedio
            - std_sentiment: Desviación estándar
            - min_sentiment: Mínimo
            - max_sentiment: Máximo
            - n_verses: Número de versículos
        
        Returns:
            DataFrame con una fila por libro
        """
        return (self.df_results
                .groupby(['book_id', 'book_name', 'testament'])['sentiment']
                .agg(['mean', 'std', 'min', 'max', 'count'])
                .reset_index()
                .rename(columns={
                    'mean': 'avg_sentiment',
                    'std': 'std_sentiment',
                    'min': 'min_sentiment',
                    'max': 'max_sentiment',
                    'count': 'n_verses'
                })
                .sort_values('book_id'))

    def by_chapter(self, book_name: str) -> pd.DataFrame:
        """
        Agrega sentimiento por capítulo dentro de un libro específico.
        
        Args:
            book_name: Nombre del libro (ej: "Genesis")
            
        Returns:
            DataFrame con sentimiento promedio por capítulo
        """
        book_data = self.df_results[self.df_results['book_name'] == book_name]
        return (book_data.groupby('chapter')['sentiment']
                .mean()
                .reset_index()
                .rename(columns={'sentiment': 'avg_sentiment'}))

    def extremes(self, n: int = 5) -> dict[str, pd.DataFrame]:
        """
        Retorna los libros más positivos y más negativos según sentimiento.
        
        Args:
            n: Número de libros a retornar en cada categoría
            
        Returns:
            Diccionario con claves 'most_positive' y 'most_negative'
        """
        by_book_data = self.by_book()
        return {
            'most_positive': by_book_data.nlargest(n, 'avg_sentiment'),
            'most_negative': by_book_data.nsmallest(n, 'avg_sentiment')
        }
