"""
corpus.py — BibleCorpus: carga y estructura jerárquica del corpus bíblico.
"""
import pandas as pd
import os


class Verse:
    """Clase que representa un versículo de la Biblia."""
    
    def __init__(self, verse_id: str, book_id: int, chapter: int,
                 verse_num: int, text: str, book_name: str, testament: str):
        """
        Inicializa un versículo con su metadata e contenido.
        
        Args:
            verse_id: Identificador único del versículo
            book_id: ID del libro (numérico)
            chapter: Número del capítulo
            verse_num: Número del versículo dentro del capítulo
            text: Contenido textual del versículo
            book_name: Nombre legible del libro
            testament: Testamento ('OT' = Antiguo, 'NT' = Nuevo)
        """
        self.verse_id = verse_id
        self.book_id = book_id
        self.chapter = chapter
        self.verse_num = verse_num
        self.text = text
        self.book_name = book_name
        self.testament = testament

    def __repr__(self):
        """Representación legible del versículo."""
        preview = self.text[:60] + ("..." if len(self.text) > 60 else "")
        return f"[{self.book_name} {self.chapter}:{self.verse_num}] {preview}"


class BibleCorpus:
    """
    Carga y organiza el corpus bíblico en una estructura jerárquica:
    Biblia → Testamento → Libro → Capítulo → Versículo.
    """

    # Stopwords en inglés (implementación propia, sin nltk)
    STOPWORDS = {
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
        'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his',
        'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
        'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
        'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
        'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having',
        'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if',
        'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for',
        'with', 'about', 'against', 'between', 'into', 'through', 'during',
        'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
        'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',
        'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
        'all', 'both', 'each', 'few', 'more', 'most', 'other', 'some',
        'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
        'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should',
        'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn',
        'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn',
        'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn',
        'unto', 'upon', 'shall', 'thee', 'thou', 'thy', 'ye', 'hath', 'doth',
        'said', 'came', 'come', 'went', 'go', 'one', 'two', 'also', 'may',
        'us', 'him', 'his', 'them', 'their'
    }

    def __init__(self, verses_path: str, keys_path: str):
        self.verses_path = verses_path
        self.keys_path = keys_path
        self.df: pd.DataFrame = pd.DataFrame()
        self.book_map: dict = {}   # book_id → {name, testament}
        self.verses: list[Verse] = []
        self._load()

    def _load(self):
        """Carga los archivos CSV y construye estructuras de datos."""
        # Cargar mapeo: libro_id → nombre y testamento
        keys_dataframe = pd.read_csv(
            self.keys_path, 
            header=0,
            names=['book_id', 'book_name', 'testament', 'genre'],
            quotechar='"'
        )
        
        for _, row in keys_dataframe.iterrows():
            book_id = int(row['book_id'])
            self.book_map[book_id] = {
                'name': row['book_name'],
                'testament': row['testament']
            }

        # Cargar versículos
        verses_dataframe = pd.read_csv(
            self.verses_path, 
            header=0,
            names=['verse_id', 'book_id', 'chapter', 'verse_num', 'text'],
            quotechar='"'
        )
        
        # Convertir tipos de datos
        verses_dataframe['book_id'] = verses_dataframe['book_id'].astype(int)
        verses_dataframe['chapter'] = verses_dataframe['chapter'].astype(int)
        verses_dataframe['verse_num'] = verses_dataframe['verse_num'].astype(int)
        
        # Enriquecer con información de libro y testamento
        verses_dataframe['book_name'] = verses_dataframe['book_id'].map(
            lambda b: self.book_map[b]['name']
        )
        verses_dataframe['testament'] = verses_dataframe['book_id'].map(
            lambda b: self.book_map[b]['testament']
        )
        
        self.df = verses_dataframe.reset_index(drop=True)

        # Construir lista de objetos Verse
        self.verses = [
            Verse(
                verse_id=row['verse_id'],
                book_id=row['book_id'],
                chapter=row['chapter'],
                verse_num=row['verse_num'],
                text=str(row['text']),
                book_name=row['book_name'],
                testament=row['testament']
            )
            for _, row in self.df.iterrows()
        ]
        print(f"[BibleCorpus] Cargados {len(self.verses):,} versículos, "
              f"{len(self.book_map)} libros.")

    def get_verses_by_book(self, book_name: str) -> list[Verse]:
        return [v for v in self.verses if v.book_name == book_name]

    def get_book_names(self) -> list[str]:
        return [self.book_map[i]['name'] for i in sorted(self.book_map)]

    def get_testament_names(self) -> list[str]:
        return ['OT', 'NT']
