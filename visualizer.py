"""
visualizer.py — BibleVisualizer: todas las visualizaciones del laboratorio.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.decomposition import PCA
from collections import Counter

from corpus import BibleCorpus
from preprocessor import TextPreprocessor
from tfidf import TFIDFVectorizer
from search_engine import SearchEngine
from sentiment import SentimentAnalyzer

# Paleta consistente
COLORS = plt.cm.tab20.colors
sns.set_theme(style='whitegrid', font_scale=0.9)
plt.rcParams.update({'figure.dpi': 120, 'font.family': 'DejaVu Sans'})


class BibleVisualizer:
    """
    Encapsula todas las visualizaciones del análisis del corpus bíblico.
    
    Crea gráficos que muestran:
        - Distribuciones de longitud de versículos
        - Frecuencia de palabras
        - Similitud entre libros (heatmap)
        - Proyecciones dimensionales (PCA)
        - Análisis de sentimiento
        - Comparaciones de modelos n-grama
    """

    def __init__(self, corpus: BibleCorpus,
                 preprocessor: TextPreprocessor,
                 vectorizer: TFIDFVectorizer,
                 sentiment_analyzer: SentimentAnalyzer,
                 output_dir: str = 'plots'):
        """
        Inicializa el visualizador
        
        Args:
            corpus: BibleCorpus con los versículos
            preprocessor: TextPreprocessor preprocesado
            vectorizer: TFIDFVectorizer ya ajustado
            sentiment_analyzer: SentimentAnalyzer con análisis realizado
            output_dir: Directorio donde guardar las gráficas PNG
        """
        self.corpus = corpus
        self.preprocessor = preprocessor
        self.vectorizer = vectorizer
        self.sentiment = sentiment_analyzer
        self.output_dir = output_dir
        
        # Crear directorio de salida si no existe
        import os
        os.makedirs(output_dir, exist_ok=True)

    def _save(self, figure, filename: str) -> str:
        """
        Guarda una figura matplotlib en disco y cierra el objeto.
        
        Args:
            figure: Objeto Figure de matplotlib
            filename: Nombre del archivo (sin extensión, se agrega .png)
            
        Returns:
            Ruta completa del archivo guardado
        """
        file_path = f"{self.output_dir}/{filename}.png"
        figure.savefig(file_path, bbox_inches='tight')
        plt.close(figure)
        print(f"  → Guardado: {file_path}")
        return file_path

    # ──────────────────────────────────────────────
    # 1. Distribución de longitud de versículos
    # ──────────────────────────────────────────────
    
    def plot_verse_length_distribution(self):
        """
        Visualiza la distribución de longitud de versículos en palabras.
        
        Muestra dos subgráficos:
            - Histograma general con media y mediana
            - Comparación OT vs NT
        """
        # Calcular longitudes
        all_lengths = [len(verse.text.split()) for verse in self.corpus.verses]
        ot_lengths = [len(v.text.split()) for v in self.corpus.verses if v.testament == 'OT']
        nt_lengths = [len(v.text.split()) for v in self.corpus.verses if v.testament == 'NT']

        figure, axes = plt.subplots(1, 2, figsize=(14, 5))

        # --- Gráfica 1: Histograma general ---
        axes[0].hist(all_lengths, bins=60, color='steelblue', edgecolor='white', alpha=0.85)
        
        # Líneas de media y mediana
        mean_length = np.mean(all_lengths)
        median_length = np.median(all_lengths)
        axes[0].axvline(mean_length, color='crimson', lw=2,
                        linestyle='--', label=f'Media: {mean_length:.1f}')
        axes[0].axvline(median_length, color='orange', lw=2,
                        linestyle='--', label=f'Mediana: {median_length:.1f}')
        axes[0].set_title('Distribución de Longitud de Versículos (palabras)', fontsize=13)
        axes[0].set_xlabel('Palabras por versículo')
        axes[0].set_ylabel('Frecuencia')
        axes[0].legend()

        # --- Gráfica 2: Comparación OT vs NT ---
        axes[1].hist(ot_lengths, bins=50, alpha=0.65, color='royalblue', 
                     label='Antiguo Testamento')
        axes[1].hist(nt_lengths, bins=50, alpha=0.65, color='tomato', 
                     label='Nuevo Testamento')
        axes[1].set_title('Longitud de Versículos: OT vs NT', fontsize=13)
        axes[1].set_xlabel('Palabras por versículo')
        axes[1].set_ylabel('Frecuencia')
        axes[1].legend()

        figure.suptitle('Análisis de Longitud de Versículos — KJV Bible', 
                        fontsize=14, fontweight='bold')
        figure.tight_layout()
        return self._save(figure, '01_verse_length_distribution')

    # ──────────────────────────────────────────────
    # 2. Versículos por libro
    # ──────────────────────────────────────────────
    
    def plot_verses_per_book(self):
        """
        Visualiza la cantidad de versículos por libro.
        
        Colores diferenciados por testamento.
        """
        # Contar versículos por libro
        verse_counts = (self.corpus.df
                       .groupby(['book_id', 'book_name', 'testament'])
                       .size()
                       .reset_index(name='count')
                       .sort_values('book_id'))

        figure, axis = plt.subplots(figsize=(18, 7))
        
        # Colorear por testamento
        colors_by_testament = ['royalblue' if test == 'OT' else 'tomato'
                              for test in verse_counts['testament']]
        
        bars = axis.bar(range(len(verse_counts)), verse_counts['count'],
                       color=colors_by_testament, edgecolor='white', linewidth=0.5)
        
        # Etiquetas en eje X
        axis.set_xticks(range(len(verse_counts)))
        axis.set_xticklabels(verse_counts['book_name'], rotation=75, ha='right', fontsize=7.5)
        
        axis.set_title('Cantidad de Versículos por Libro de la Biblia (KJV)', 
                      fontsize=14, fontweight='bold')
        axis.set_ylabel('Número de versículos')
        
        # Leyenda
        legend_patches = [mpatches.Patch(color='royalblue', label='Antiguo Testamento'),
                         mpatches.Patch(color='tomato', label='Nuevo Testamento')]
        axis.legend(handles=legend_patches)
        
        figure.tight_layout()
        return self._save(figure, '02_verses_per_book')

    # ──────────────────────────────────────────────
    # 3. Top N palabras más frecuentes
    # ──────────────────────────────────────────────
    def plot_top_words(self, n: int = 30):
        top = self.preprocessor.get_top_words(n)
        words, counts = zip(*top)

        fig, ax = plt.subplots(figsize=(12, 7))
        y_pos = range(len(words))
        colors_bar = plt.cm.viridis(np.linspace(0.2, 0.85, len(words)))
        bars = ax.barh(y_pos, counts, color=colors_bar, edgecolor='white')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(words, fontsize=11)
        ax.invert_yaxis()
        ax.set_title(f'Top {n} Palabras Más Frecuentes (sin stopwords)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Frecuencia')
        for bar, count in zip(bars, counts):
            ax.text(count + max(counts) * 0.005, bar.get_y() + bar.get_height() / 2,
                    f'{count:,}', va='center', fontsize=8)
        fig.tight_layout()
        return self._save(fig, '03_top_words')

    # ──────────────────────────────────────────────
    # 4. Heatmap de similitud entre libros (OBLIGATORIO)
    # ──────────────────────────────────────────────
    def plot_book_similarity_heatmap(self):
        book_names, book_vecs = self.vectorizer.get_book_vectors()
        # Similitud coseno: como los vectores están normalizados → producto punto
        sim_matrix = book_vecs @ book_vecs.T  # (66, 66)
        np.fill_diagonal(sim_matrix, 1.0)

        # Agrupar por testamento para mejor visualización
        ot_names = [n for n in book_names
                    if self.corpus.book_map[list(self.corpus.book_map.keys())[book_names.index(n)]]['testament'] == 'OT']
        book_id_map = {v['name']: k for k, v in self.corpus.book_map.items()}
        order_ot = [n for n in book_names if self.corpus.book_map[book_id_map[n]]['testament'] == 'OT']
        order_nt = [n for n in book_names if self.corpus.book_map[book_id_map[n]]['testament'] == 'NT']
        ordered = order_ot + order_nt

        idx_map = {n: i for i, n in enumerate(book_names)}
        reorder = [idx_map[n] for n in ordered]
        sim_reordered = sim_matrix[np.ix_(reorder, reorder)]

        fig, ax = plt.subplots(figsize=(22, 18))
        mask = np.eye(len(ordered), dtype=bool)
        im = ax.imshow(sim_reordered, cmap='YlOrRd', vmin=0, vmax=1, aspect='auto')
        ax.set_xticks(range(len(ordered)))
        ax.set_yticks(range(len(ordered)))
        ax.set_xticklabels(ordered, rotation=90, fontsize=6.5)
        ax.set_yticklabels(ordered, fontsize=6.5)

        # Línea divisoria OT/NT
        n_ot = len(order_ot)
        ax.axhline(n_ot - 0.5, color='navy', lw=2)
        ax.axvline(n_ot - 0.5, color='navy', lw=2)
        ax.text(n_ot / 2, -2, 'Antiguo Testamento', ha='center',
                fontsize=10, color='navy', fontweight='bold')
        ax.text(n_ot + len(order_nt) / 2, -2, 'Nuevo Testamento', ha='center',
                fontsize=10, color='darkred', fontweight='bold')

        plt.colorbar(im, ax=ax, fraction=0.02, pad=0.04, label='Similitud del Coseno')
        ax.set_title('Heatmap de Similitud TF-IDF entre Libros de la Biblia (KJV)',
                     fontsize=14, fontweight='bold', pad=15)
        fig.tight_layout()
        return self._save(fig, '04_book_similarity_heatmap')

    # ──────────────────────────────────────────────
    # 5. PCA de versículos
    # ──────────────────────────────────────────────
    def plot_pca_verses(self, sample_size: int = 5000, random_state: int = 42):
        """
        PCA bidimensional de versículos (TF-IDF → 2D).
        Se muestrea para mantener el gráfico legible.
        """
        np.random.seed(random_state)
        n = len(self.corpus.verses)
        indices = np.random.choice(n, min(sample_size, n), replace=False)

        X_sample = self.vectorizer.matrix[indices]
        testaments = [self.corpus.verses[i].testament for i in indices]
        books = [self.corpus.verses[i].book_name for i in indices]

        pca = PCA(n_components=2, random_state=random_state)
        X_2d = pca.fit_transform(X_sample)
        var = pca.explained_variance_ratio_

        fig, axes = plt.subplots(1, 2, figsize=(18, 7))

        # --- Por testamento ---
        color_map_t = {'OT': 'royalblue', 'NT': 'tomato'}
        for test in ['OT', 'NT']:
            mask = [t == test for t in testaments]
            axes[0].scatter(X_2d[mask, 0], X_2d[mask, 1],
                            c=color_map_t[test], alpha=0.3, s=8,
                            label='Antiguo Testamento' if test == 'OT' else 'Nuevo Testamento')
        axes[0].set_xlabel(f'PC1 ({var[0]*100:.1f}% var explicada)', fontsize=11)
        axes[0].set_ylabel(f'PC2 ({var[1]*100:.1f}% var explicada)', fontsize=11)
        axes[0].set_title('PCA de Versículos — por Testamento', fontsize=13, fontweight='bold')
        axes[0].legend(markerscale=3)

        # --- Por libro (top 10 libros con más versículos) ---
        book_counts = Counter(books)
        top_books = [b for b, _ in book_counts.most_common(10)]
        palette = plt.cm.tab10.colors
        for i, book in enumerate(top_books):
            mask = [b == book for b in books]
            axes[1].scatter(X_2d[mask, 0], X_2d[mask, 1],
                            c=[palette[i]], alpha=0.4, s=8, label=book)
        other_mask = [b not in top_books for b in books]
        axes[1].scatter(X_2d[other_mask, 0], X_2d[other_mask, 1],
                        c='lightgray', alpha=0.15, s=4, label='Otros')
        axes[1].set_xlabel(f'PC1 ({var[0]*100:.1f}% var explicada)', fontsize=11)
        axes[1].set_ylabel(f'PC2 ({var[1]*100:.1f}% var explicada)', fontsize=11)
        axes[1].set_title('PCA de Versículos — por Libro (top 10)', fontsize=13, fontweight='bold')
        axes[1].legend(markerscale=3, fontsize=8, loc='upper right')

        fig.suptitle(f'PCA de Versículos KJV (n={len(indices):,} muestras)',
                     fontsize=14, fontweight='bold')
        fig.tight_layout()
        return self._save(fig, '05_pca_verses')

    # ──────────────────────────────────────────────
    # 6. Análisis de sentimiento por libro
    # ──────────────────────────────────────────────
    def plot_sentiment_by_book(self):
        by_book = self.sentiment.by_book()

        fig, axes = plt.subplots(2, 1, figsize=(18, 12))

        # Barras: sentimiento promedio por libro
        colors_sent = ['tomato' if s < 0 else 'steelblue'
                       for s in by_book['avg_sentiment']]
        bars = axes[0].bar(range(len(by_book)), by_book['avg_sentiment'],
                           color=colors_sent, edgecolor='white', linewidth=0.4)
        axes[0].axhline(0, color='black', lw=1, linestyle='--')
        axes[0].set_xticks(range(len(by_book)))
        axes[0].set_xticklabels(by_book['book_name'], rotation=75, ha='right', fontsize=7.5)
        axes[0].set_title('Sentimiento Promedio por Libro (KJV)', fontsize=13, fontweight='bold')
        axes[0].set_ylabel('Sentimiento promedio')
        patches = [mpatches.Patch(color='steelblue', label='Positivo'),
                   mpatches.Patch(color='tomato', label='Negativo')]
        axes[0].legend(handles=patches)

        # Evolución a lo largo de los libros (en orden canónico)
        axes[1].plot(range(len(by_book)), by_book['avg_sentiment'],
                     color='purple', lw=1.5, alpha=0.8)
        axes[1].fill_between(range(len(by_book)), by_book['avg_sentiment'],
                             alpha=0.2, color='purple')
        axes[1].axhline(0, color='black', lw=1, linestyle='--')

        # Marcar división OT/NT
        ot_count = len([r for _, r in by_book.iterrows() if r['testament'] == 'OT'])
        axes[1].axvline(ot_count - 0.5, color='navy', lw=2, linestyle='-', alpha=0.7)
        axes[1].text(ot_count / 2, axes[1].get_ylim()[1] * 0.95,
                     'AT', ha='center', fontsize=12, color='navy', fontweight='bold')
        axes[1].text(ot_count + (len(by_book) - ot_count) / 2, axes[1].get_ylim()[1] * 0.95,
                     'NT', ha='center', fontsize=12, color='darkred', fontweight='bold')
        axes[1].set_xticks(range(len(by_book)))
        axes[1].set_xticklabels(by_book['book_name'], rotation=75, ha='right', fontsize=7.5)
        axes[1].set_title('Evolución del Sentimiento a lo largo de los Libros', fontsize=13, fontweight='bold')
        axes[1].set_ylabel('Sentimiento promedio')

        fig.tight_layout()
        return self._save(fig, '06_sentiment_by_book')

    # ──────────────────────────────────────────────
    # 7. Evolución de frecuencia de palabras clave
    # ──────────────────────────────────────────────
    def plot_keyword_evolution(self, keywords: list[str] = None):
        if keywords is None:
            keywords = ['god', 'lord', 'love', 'death', 'war', 'peace', 'prophet', 'jesus']

        df = self.corpus.df.copy()
        book_names = self.corpus.get_book_names()
        results = {kw: [] for kw in keywords}

        for book_name in book_names:
            book_verses = df[df['book_name'] == book_name]['text']
            all_text = ' '.join(book_verses).lower()
            total_words = len(all_text.split()) or 1
            for kw in keywords:
                freq = all_text.count(kw) / total_words * 1000
                results[kw].append(freq)

        fig, ax = plt.subplots(figsize=(18, 7))
        palette = plt.cm.tab10.colors
        for i, kw in enumerate(keywords):
            ax.plot(range(len(book_names)), results[kw],
                    label=kw, color=palette[i % 10], lw=1.5, alpha=0.85)

        # Línea OT/NT
        ot_count = len([n for n in book_names
                        if self.corpus.book_map[
                            [k for k, v in self.corpus.book_map.items()
                             if v['name'] == n][0]]['testament'] == 'OT'])
        ax.axvline(ot_count - 0.5, color='navy', lw=2, linestyle='--', alpha=0.6)
        ax.set_xticks(range(len(book_names)))
        ax.set_xticklabels(book_names, rotation=75, ha='right', fontsize=7)
        ax.set_title('Evolución de Frecuencia de Palabras Clave por Libro (por 1000 palabras)',
                     fontsize=13, fontweight='bold')
        ax.set_ylabel('Frecuencia relativa (‰)')
        ax.legend(loc='upper right', fontsize=9)
        fig.tight_layout()
        return self._save(fig, '07_keyword_evolution')

    # ──────────────────────────────────────────────
    # 8. Comparación de modelos n-gram
    # ──────────────────────────────────────────────
    def plot_ngram_comparison(self, ngram_generator, seed: str = 'god'):
        """Visualiza la coherencia comparativa de los modelos n-gram."""
        texts = ngram_generator.compare_models(seed_word=seed, max_len=25)

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.axis('off')
        col_labels = ['Modelo', 'Texto generado (seed: "{}") '.format(seed)]
        rows = [[model, text] for model, text in texts.items()]
        table = ax.table(cellText=rows, colLabels=col_labels,
                         cellLoc='left', loc='center',
                         colWidths=[0.12, 0.88])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.5)
        # Header styling
        for j in range(len(col_labels)):
            table[0, j].set_facecolor('#2c3e50')
            table[0, j].set_text_props(color='white', fontweight='bold')
        for i in range(1, len(rows) + 1):
            bg = '#f0f4f8' if i % 2 == 0 else 'white'
            for j in range(len(col_labels)):
                table[i, j].set_facecolor(bg)

        ax.set_title('Comparación de Textos Generados por Modelos N-Gram',
                     fontsize=13, fontweight='bold', pad=20)
        fig.tight_layout()
        return self._save(fig, '08_ngram_comparison')

    # ──────────────────────────────────────────────
    # 9. Matriz de confusión del clasificador (top N libros)
    # ──────────────────────────────────────────────
    def plot_confusion_matrix(self, classifier, top_n: int = 20):
        cm = classifier.confusion_matrix()
        labels = classifier.book_names_in_train

        # Seleccionar los N libros con más muestras de prueba
        test_counts = np.sum(cm, axis=1)
        top_idx = np.argsort(test_counts)[::-1][:top_n]
        top_idx = sorted(top_idx)
        cm_top = cm[np.ix_(top_idx, top_idx)]
        top_labels = [labels[i] for i in top_idx]

        fig, ax = plt.subplots(figsize=(14, 12))
        im = ax.imshow(cm_top, interpolation='nearest', cmap='Blues')
        plt.colorbar(im, ax=ax, fraction=0.03)
        ax.set_xticks(range(len(top_labels)))
        ax.set_yticks(range(len(top_labels)))
        ax.set_xticklabels(top_labels, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(top_labels, fontsize=8)
        ax.set_title(f'Matriz de Confusión — Top {top_n} Libros',
                     fontsize=13, fontweight='bold')
        ax.set_ylabel('Real')
        ax.set_xlabel('Predicho')
        fig.tight_layout()
        return self._save(fig, '09_confusion_matrix')

    # ──────────────────────────────────────────────
    # 10. Sentimiento extremo: capítulos de un libro
    # ──────────────────────────────────────────────
    def plot_sentiment_chapters(self, book_name: str = 'Psalms'):
        by_chap = self.sentiment.by_chapter(book_name)

        fig, ax = plt.subplots(figsize=(14, 5))
        colors_ch = ['tomato' if s < 0 else 'steelblue'
                     for s in by_chap['avg_sentiment']]
        ax.bar(by_chap['chapter'], by_chap['avg_sentiment'],
               color=colors_ch, edgecolor='white', linewidth=0.3)
        ax.axhline(0, color='black', lw=1, linestyle='--')
        ax.set_title(f'Sentimiento Promedio por Capítulo — {book_name}',
                     fontsize=13, fontweight='bold')
        ax.set_xlabel('Capítulo')
        ax.set_ylabel('Sentimiento promedio')
        fig.tight_layout()
        return self._save(fig, f'10_sentiment_{book_name.lower().replace(" ", "_")}')
