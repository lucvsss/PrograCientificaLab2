import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg") 

from src.models import Biblia
from src.data_loader import cargar_dataset
from src.preprocessing import TextPreprocessor
from src.tfidf import TFIDFVectorizer, cosine_similarity_matrix
from src.search_engine import SemanticSearchEngine
from src.classifier import VerseClassifier
from src.ngram_model import comparar_modelos
from src.sentiment import (
    LexiconSentimentAnalyzer,
    calcular_sentimiento_corpus,
    agregar_por_libro,
    agregar_por_capitulo,
)
from src import visualization as viz


def main():
    script_root = Path(__file__).resolve().parent
    dir_dataset = script_root / "data"
    dir_imgs = script_root / "imgs"
    dir_imgs.mkdir(exist_ok=True)

    path_bible = dir_dataset / "t_asv.csv"
    path_key = dir_dataset / "key_english.csv"
    path_genre = dir_dataset / "key_genre_english.csv"
    path_stopwords = dir_dataset / "stopwords.json"

    #Carga y construccion de la jerarquia
    df_raw = cargar_dataset(path_bible, path_key, path_genre)

    biblia = Biblia.from_dataframe(
        df_raw,
        col_libro="Book Name",
        col_testamento="Testament (OT or NT)",
        col_capitulo="Chapter",
        col_versiculo="Verse",
        col_texto="Text",
        col_genero="Genre name",
    )
    print(biblia)
    print("\nResumen por testamento:")
    print(biblia.get_resumen().to_string(index=False))
    print("\nResumen por genero literario:")
    print(biblia.get_resumen_generos().to_string(index=False))

    df = biblia.to_dataframe()

    #Preprocesamiento
    with open(path_stopwords) as f:
        stopwords = set(json.load(f))

    preprocessor = TextPreprocessor(stopwords=stopwords)
    df["texto_procesado"] = preprocessor.process_corpus(df["texto_original"].tolist())

    top20 = preprocessor.palabras_mas_frecuentes(20)
    print("\nTop 20 palabras mas frecuentes:")
    for palabra, freq in top20:
        print(f"  {palabra:<15} {freq}")

    #TF-IDF a nivel versiculo
    vectorizer = TFIDFVectorizer()
    matriz_tfidf_versiculos = vectorizer.fit_transform(df["texto_procesado"].tolist())
    print(f"\nMatriz TF-IDF de versiculos: {matriz_tfidf_versiculos.shape} "
          f"(versiculos x vocabulario)")

    viz.plot_longitud_versiculos(df).savefig(dir_imgs / "longitud_versiculos.png", dpi=120)
    viz.plot_versiculos_por_libro(df).savefig(dir_imgs / "versiculos_por_libro.png", dpi=120)
    viz.plot_top_palabras(dict(preprocessor.frecuencias)).savefig(
        dir_imgs / "top_palabras.png", dpi=120)
    viz.plot_wordcloud(dict(preprocessor.frecuencias)).savefig(
        dir_imgs / "wordcloud.png", dpi=120)
    viz.plot_grafo_coocurrencia(df["texto_procesado"].tolist()).savefig(
        dir_imgs / "grafo_coocurrencia.png", dpi=120)

    textos_por_libro = df.groupby("libro")["texto_procesado"].sum()
    vectorizer_libros = TFIDFVectorizer()
    matriz_tfidf_libros = vectorizer_libros.fit_transform(textos_por_libro.tolist())
    matriz_similitud_libros = cosine_similarity_matrix(matriz_tfidf_libros)
    viz.plot_heatmap_similitud_libros(
        matriz_similitud_libros, textos_por_libro.index.tolist()
    ).savefig(dir_imgs / "heatmap_similitud_libros.png", dpi=120)
  
    viz.plot_pca_versiculos(
        matriz_tfidf_versiculos, df["testamento"], titulo="Versiculos por testamento"
    ).savefig(dir_imgs / "pca_testamento.png", dpi=120)
    viz.plot_pca_versiculos(
        matriz_tfidf_versiculos, df["genero"], titulo="Versiculos por genero literario"
    ).savefig(dir_imgs / "pca_genero.png", dpi=120)

    #Motor de busqueda
    motor = SemanticSearchEngine(preprocessor, TFIDFVectorizer())
    motor.fit(df)
    print("\nBusqueda semantica para 'love peace and faith':")
    print(motor.buscar("love peace and faith", k=5).to_string(index=False))

    clasificador = VerseClassifier(modelo="logistic")
    clasificador.entrenar(matriz_tfidf_versiculos, df["libro"])
    resultados_clf = clasificador.evaluar()
    print(f"\nClasificador ({clasificador.nombre_modelo}) "
          f"accuracy = {resultados_clf['accuracy']:.4f}")
    viz.plot_matriz_confusion(
        resultados_clf["matriz_confusion"], resultados_clf["clases"]
    ).savefig(dir_imgs / "matriz_confusion.png", dpi=110)

    #Generador texto
    print("\nVersiculos generados (palabra inicial = 'god'):")
    generados = comparar_modelos(
        df["texto_procesado"].tolist(), ns=(1, 2, 3, 4),
        palabra_inicial="god", max_len=20,
    )
    for n, texto in generados.items():
        print(f"  n={n}: {texto}")

    #Analisis sentimiento
    analizador = LexiconSentimentAnalyzer()
    df = calcular_sentimiento_corpus(df, analizador)
    sentimiento_por_libro = agregar_por_libro(df)
    sentimiento_por_capitulo = agregar_por_capitulo(df)
    viz.plot_sentimiento_por_libro(sentimiento_por_libro).savefig(
        dir_imgs / "sentimiento_por_libro.png", dpi=120)

    print("\nLibros con sentimiento promedio mas negativo:")
    print(sentimiento_por_libro.head(5)[["libro", "mean"]].to_string(index=False))
    print("\nLibros con sentimiento promedio mas positivo:")
    print(sentimiento_por_libro.tail(5)[["libro", "mean"]].to_string(index=False))

    print("\nPipeline completo ejecutado correctamente. Figuras en imgs/.")
    return df


if __name__ == "__main__":
    main()
