"""
main.py — Punto de entrada del sistema Biblical Text Mining.
Orquesta todos los módulos en el orden requerido por el laboratorio.
"""
import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

# ── Importar módulos propios ──────────────────────────────────────────────────
from corpus import BibleCorpus
from preprocessor import TextPreprocessor
from tfidf import TFIDFVectorizer
from search_engine import SearchEngine
from classifier import VerseClassifier
from ngram_generator import NGramGenerator
from sentiment import SentimentAnalyzer
from visualizer import BibleVisualizer

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VERSES_CSV  = os.path.join(BASE_DIR, 't_kjv.csv')
KEYS_CSV    = os.path.join(BASE_DIR, 'key_english.csv')
PLOTS_DIR   = os.path.join(BASE_DIR, 'plots')


def separator(title: str):
    """Imprime un separador visual para organizar las secciones del programa."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def main():
    t0 = time.time()

    # ─────────────────────────────────────────────
    # 1. CARGA DEL CORPUS
    # ─────────────────────────────────────────────
    separator("1. CARGA DEL CORPUS")
    corpus = BibleCorpus(VERSES_CSV, KEYS_CSV)
    print(f"  Libros: {len(corpus.book_map)}")
    print(f"  Testamentos: OT + NT")
    print(f"  Versículo ejemplo: {corpus.verses[0]}")

    # ─────────────────────────────────────────────
    # 2. PREPROCESAMIENTO
    # ─────────────────────────────────────────────
    separator("2. PREPROCESAMIENTO")
    preprocessor = TextPreprocessor(corpus)
    preprocessor.build()
    print(f"\n  Top 15 palabras (sin stopwords):")
    for word, freq in preprocessor.get_top_words(15):
        print(f"    {word:20s} {freq:>7,}")

    # ─────────────────────────────────────────────
    # 3. TF-IDF
    # ─────────────────────────────────────────────
    separator("3. VECTORIZACIÓN TF-IDF (implementación propia)")
    vectorizer = TFIDFVectorizer(preprocessor, max_features=5000)
    vectorizer.fit_transform()

    # ─────────────────────────────────────────────
    # 4. SENTIMIENTO
    # ─────────────────────────────────────────────
    separator("4. ANÁLISIS DE SENTIMIENTO")
    sentiment_analyzer = SentimentAnalyzer(corpus, preprocessor)
    sentiment_analyzer.analyze()
    extremes = sentiment_analyzer.extremes(n=5)
    print("\n  5 libros más positivos:")
    print(extremes['most_positive'][['book_name', 'avg_sentiment']].to_string(index=False))
    print("\n  5 libros más negativos:")
    print(extremes['most_negative'][['book_name', 'avg_sentiment']].to_string(index=False))

    # ─────────────────────────────────────────────
    # 5. VISUALIZACIONES
    # ─────────────────────────────────────────────
    separator("5. VISUALIZACIONES")
    viz = BibleVisualizer(corpus, preprocessor, vectorizer,
                          sentiment_analyzer, output_dir=PLOTS_DIR)

    print("  [1/7] Distribución de longitud de versículos...")
    viz.plot_verse_length_distribution()

    print("  [2/7] Versículos por libro...")
    viz.plot_verses_per_book()

    print("  [3/7] Top palabras frecuentes...")
    viz.plot_top_words(30)

    print("  [4/7] Heatmap de similitud entre libros (OBLIGATORIO)...")
    viz.plot_book_similarity_heatmap()

    print("  [5/7] PCA de versículos...")
    viz.plot_pca_verses(sample_size=6000)

    print("  [6/7] Sentimiento por libro...")
    viz.plot_sentiment_by_book()

    print("  [7/7] Evolución de palabras clave...")
    viz.plot_keyword_evolution()

    # ─────────────────────────────────────────────
    # 6. MOTOR DE BÚSQUEDA
    # ─────────────────────────────────────────────
    separator("6. MOTOR DE BÚSQUEDA SEMÁNTICO")
    engine = SearchEngine(corpus, preprocessor, vectorizer)

    queries = [
        "love thy neighbor as thyself",
        "in the beginning God created",
        "fear not for I am with you",
    ]
    for query in queries:
        print(f"\n  Query: \"{query}\"")
        results = engine.search(query, k=5)
        print(results[['rank', 'book', 'chapter', 'verse', 'similarity', 'text']].to_string(index=False))

    # ─────────────────────────────────────────────
    # 7. CLASIFICADOR
    # ─────────────────────────────────────────────
    separator("7. CLASIFICADOR DE VERSÍCULOS")
    clf = VerseClassifier(corpus, preprocessor, vectorizer, model_type='nb')
    clf.train(test_size=0.2)
    print(f"\n  Accuracy: {clf.accuracy:.4f}")
    print("\n  Classification Report (primeros 20 libros en report):")
    report_lines = clf.report().split('\n')
    print('\n'.join(report_lines[:25]))

    # Guardar matriz de confusión
    print("\n  Guardando matriz de confusión...")
    viz.plot_confusion_matrix(clf, top_n=20)

    # Predicciones de ejemplo
    test_verses = [
        "In the beginning God created the heaven and the earth.",
        "Jesus wept.",
        "The LORD is my shepherd; I shall not want.",
        "For God so loved the world that he gave his only begotten Son.",
    ]
    print("\n  Predicciones de ejemplo:")
    for tv in test_verses:
        pred = clf.predict(tv)
        print(f"    '{tv[:60]}...' → {pred}")

    # ─────────────────────────────────────────────
    # 8. GENERADOR N-GRAM
    # ─────────────────────────────────────────────
    separator("8. MODELO GENERADOR DE TEXTO (N-GRAM)")
    generator = NGramGenerator(corpus, preprocessor)
    generator.build(n_values=[1, 2, 3, 4])

    seeds = ['god', 'love', 'king', 'prophet']
    for seed in seeds:
        print(f"\n  — Generaciones con seed='{seed}' —")
        texts = generator.compare_models(seed_word=seed, max_len=20)
        for model, text in texts.items():
            print(f"    [{model:8s}]: {text}")

    # Visualización de comparación N-gram
    print("\n  Guardando comparación visual de modelos n-gram...")
    viz.plot_ngram_comparison(generator, seed='god')

    # Sentimiento por capítulos de Psalms
    print("\n  Guardando sentimiento por capítulos (Psalms)...")
    viz.plot_sentiment_chapters('Psalms')
    viz.plot_sentiment_chapters('Revelation')

    # ─────────────────────────────────────────────
    # RESUMEN FINAL
    # ─────────────────────────────────────────────
    separator("COMPLETADO")
    elapsed = time.time() - t0
    print(f"  Tiempo total: {elapsed:.1f}s")
    print(f"  Gráficos guardados en: {PLOTS_DIR}/")
    print(f"  Archivos generados:")
    for f in sorted(os.listdir(PLOTS_DIR)):
        print(f"    {f}")


if __name__ == '__main__':
    main()
