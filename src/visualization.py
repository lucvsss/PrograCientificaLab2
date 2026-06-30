import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_longitud_versiculos(df, columna="texto_original"):
    longitudes = [len(str(texto).split()) for texto in df[columna]]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(longitudes, bins=40)
    ax.set_title("Distribucion de longitud de versiculos (en palabras)")
    ax.set_xlabel("Cantidad de palabras")
    ax.set_ylabel("Frecuencia")
    fig.tight_layout()
    return fig


def plot_versiculos_por_libro(df):
    conteo = df.groupby("libro").size().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(conteo.index, conteo.values)
    ax.set_title("Cantidad de versiculos por libro")
    ax.set_ylabel("N° de versiculos")
    plt.xticks(rotation=90)
    fig.tight_layout()
    return fig


def plot_heatmap_similitud_libros(matriz_similitud, nombres_libros):
    fig, ax = plt.subplots(figsize=(18, 16))
    imagen = ax.imshow(matriz_similitud, cmap="Blues")
    fig.colorbar(imagen, ax=ax)

    ax.set_xticks(range(len(nombres_libros)))
    ax.set_yticks(range(len(nombres_libros)))
    ax.set_xticklabels(nombres_libros, rotation=90, fontsize=6)
    ax.set_yticklabels(nombres_libros, fontsize=6)

    ax.set_title("Similitud de coseno entre libros (basado en TF-IDF)")
    fig.tight_layout()
    return fig


def plot_pca_versiculos(matriz_tfidf, etiquetas, titulo="Versiculos proyectados con PCA"):
    from sklearn.decomposition import PCA

    if hasattr(matriz_tfidf, "toarray"):
        matriz_tfidf = matriz_tfidf.toarray()

    pca = PCA(n_components=2)
    componentes = pca.fit_transform(matriz_tfidf)

    var_pc1 = pca.explained_variance_ratio_[0] * 100
    var_pc2 = pca.explained_variance_ratio_[1] * 100

    if hasattr(etiquetas, "values"):
        etiquetas = etiquetas.values
    df_plot = pd.DataFrame({
        "PC1": componentes[:, 0],
        "PC2": componentes[:, 1],
        "etiqueta": etiquetas,
    })

    fig, ax = plt.subplots(figsize=(10, 8))
    for etiqueta in df_plot["etiqueta"].unique():
        sub = df_plot[df_plot["etiqueta"] == etiqueta]
        ax.scatter(sub["PC1"], sub["PC2"], s=15, alpha=0.5, label=etiqueta)

    ax.set_title(titulo)
    ax.set_xlabel(f"PC1 ({var_pc1:.1f}% var. explicada)")
    ax.set_ylabel(f"PC2 ({var_pc2:.1f}% var. explicada)")

    if df_plot["etiqueta"].nunique() <= 15:
        ax.legend()
    fig.tight_layout()
    return fig


def plot_sentimiento_por_libro(df_sentimiento_agregado):
    df_ord = df_sentimiento_agregado.sort_values("mean")

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(df_ord["libro"], df_ord["mean"])
    ax.set_title("Sentimiento promedio por libro")
    ax.set_xlabel("Sentimiento promedio (negativo <- 0 -> positivo)")
    fig.tight_layout()
    return fig


def plot_wordcloud(frecuencias, titulo="Palabras mas frecuentes"):
    from wordcloud import WordCloud

    wc = WordCloud(width=900, height=500, background_color="white")
    wc = wc.generate_from_frequencies(frecuencias)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(titulo)
    return fig


def plot_matriz_confusion(cm, clases):
    fig, ax = plt.subplots(figsize=(14, 12))
    imagen = ax.imshow(cm, cmap="Blues")
    fig.colorbar(imagen, ax=ax)

    ax.set_xticks(range(len(clases)))
    ax.set_yticks(range(len(clases)))
    ax.set_xticklabels(clases, rotation=90)
    ax.set_yticklabels(clases)

    ax.set_title("Matriz de confusion - Clasificador de versiculos")
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    fig.tight_layout()
    return fig

def plot_top_palabras(frecuencias, n=20, titulo="Palabras mas frecuentes del corpus"):
    if isinstance(frecuencias, dict):
        items = sorted(frecuencias.items(), key=lambda x: x[1], reverse=True)[:n]
    else:
        items = list(frecuencias)[:n]

    palabras = [p for p, _ in items]
    conteos = [c for _, c in items]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(palabras[::-1], conteos[::-1])
    ax.set_title(titulo)
    ax.set_xlabel("Frecuencia (conteo de apariciones)")
    fig.tight_layout()
    return fig


def plot_grafo_coocurrencia(documentos_tokenizados, top_n=30, min_peso=8,
                            titulo="Grafo de co-ocurrencia de palabras"):
    import networkx as nx
    from collections import Counter
    from itertools import combinations

    frec = Counter()
    for tokens in documentos_tokenizados:
        frec.update(set(tokens))
    palabras_top = {p for p, _ in frec.most_common(top_n)}

    pares = Counter()
    for tokens in documentos_tokenizados:
        presentes = sorted(set(t for t in tokens if t in palabras_top))
        for a, b in combinations(presentes, 2):
            pares[(a, b)] += 1

    G = nx.Graph()
    for (a, b), peso in pares.items():
        if peso >= min_peso:
            G.add_edge(a, b, weight=peso)

    fig, ax = plt.subplots(figsize=(12, 10))
    if G.number_of_edges() == 0:
        ax.text(0.5, 0.5, "Sin co-ocurrencias sobre el umbral", ha="center")
        ax.axis("off")
        return fig

    pos = nx.spring_layout(G, seed=42, k=0.6)
    pesos = [G[u][v]["weight"] for u, v in G.edges()]
    tam = [frec[n] / max(frec.values()) * 1500 + 100 for n in G.nodes()]

    nx.draw_networkx_nodes(G, pos, node_size=tam, node_color="#9ecae1", ax=ax)
    nx.draw_networkx_edges(G, pos, width=[w / max(pesos) * 4 for w in pesos],
                           alpha=0.4, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=9, ax=ax)
    ax.set_title(titulo)
    ax.axis("off")
    fig.tight_layout()
    return fig
