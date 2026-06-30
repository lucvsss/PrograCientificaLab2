import json, matplotlib
matplotlib.use("Agg")
import numpy as np, pandas as pd
from sklearn.metrics import f1_score, classification_report

from src.data_loader import cargar_dataset
from src.models import Biblia
from src.preprocessing import TextPreprocessor
from src.tfidf import TFIDFVectorizer, cosine_similarity_matrix
from src.search_engine import SemanticSearchEngine
from src.classifier import VerseClassifier
from src.ngram_model import comparar_modelos, NGramModel
from src.sentiment import LexiconSentimentAnalyzer, calcular_sentimiento_corpus, agregar_por_libro, agregar_por_capitulo
from src import visualization as viz

IMG = "imagenes"
R = {}

df_raw = cargar_dataset('data/t_asv.csv','data/key_english.csv','data/key_genre_english.csv')
biblia = Biblia.from_dataframe(df_raw, col_libro='Book Name', col_testamento='Testament (OT or NT)',
                               col_capitulo='Chapter', col_versiculo='Verse', col_texto='Text', col_genero='Genre name')
df = biblia.to_dataframe()
sw = set(json.load(open('data/stopwords.json')))
pre = TextPreprocessor(stopwords=sw)
df['texto_procesado'] = pre.process_corpus(df['texto_original'].tolist())

R['n_versiculos'] = int(len(df))
R['n_libros'] = int(df['libro'].nunique())
R['n_vocab'] = int(len(pre.vocabulario))
R['resumen_testamento'] = biblia.get_resumen().to_dict('records')
R['resumen_genero'] = biblia.get_resumen_generos().to_dict('records')
R['top20'] = [[p, int(c)] for p, c in pre.palabras_mas_frecuentes(20)]

longs = df['texto_original'].str.split().apply(len)
R['long_mean'] = round(float(longs.mean()), 2)
R['long_median'] = int(longs.median())
R['long_max'] = int(longs.max())
R['long_min'] = int(longs.min())
R['libro_mas_versiculos'] = df.groupby('libro').size().idxmax()
R['libro_mas_versiculos_n'] = int(df.groupby('libro').size().max())
R['libro_menos_versiculos'] = df.groupby('libro').size().idxmin()
R['libro_menos_versiculos_n'] = int(df.groupby('libro').size().min())

viz.plot_longitud_versiculos(df).savefig(f"{IMG}/longitud_versiculos.png", dpi=120)
viz.plot_versiculos_por_libro(df).savefig(f"{IMG}/versiculos_por_libro.png", dpi=120)
viz.plot_top_palabras(dict(pre.frecuencias)).savefig(f"{IMG}/top_palabras.png", dpi=120)
viz.plot_wordcloud(dict(pre.frecuencias)).savefig(f"{IMG}/wordcloud.png", dpi=120)
viz.plot_grafo_coocurrencia(df['texto_procesado'].tolist(), top_n=30, min_peso=40).savefig(f"{IMG}/grafo_coocurrencia.png", dpi=120)

tpl = df.groupby('libro')['texto_procesado'].sum()
vl = TFIDFVectorizer(); Ml = vl.fit_transform(tpl.tolist())
sim = cosine_similarity_matrix(Ml)
viz.plot_heatmap_similitud_libros(sim, tpl.index.tolist()).savefig(f"{IMG}/heatmap_similitud_libros.png", dpi=120)
nombres = tpl.index.tolist()
pares = []
for i in range(len(nombres)):
    for j in range(i+1, len(nombres)):
        pares.append((nombres[i], nombres[j], float(sim[i, j])))
pares.sort(key=lambda x: x[2], reverse=True)
R['pares_mas_similares'] = [[a, b, round(s, 3)] for a, b, s in pares[:6]]
R['pares_menos_similares'] = [[a, b, round(s, 3)] for a, b, s in pares[-6:]]

R['ngram_god'] = {n: t for n, t in comparar_modelos(df['texto_procesado'].tolist(), ns=(1,2,3,4), palabra_inicial='god', max_len=20).items()}
R['ngram_love'] = {n: t for n, t in comparar_modelos(df['texto_procesado'].tolist(), ns=(1,2,3,4), palabra_inicial='love', max_len=20).items()}

an = LexiconSentimentAnalyzer()
df = calcular_sentimiento_corpus(df, an)
sl = agregar_por_libro(df); sc = agregar_por_capitulo(df)
viz.plot_sentimiento_por_libro(sl).savefig(f"{IMG}/sentimiento_por_libro.png", dpi=120)
R['sent_neg_libros'] = [[r['libro'], round(r['mean'], 3)] for r in sl.head(5).to_dict('records')]
R['sent_pos_libros'] = [[r['libro'], round(r['mean'], 3)] for r in sl.tail(5).to_dict('records')]
sc_sorted = sc.sort_values('sentimiento')
R['cap_neg'] = [[r['libro'], int(r['capitulo']), round(r['sentimiento'], 3)] for r in sc_sorted.head(3).to_dict('records')]
R['cap_pos'] = [[r['libro'], int(r['capitulo']), round(r['sentimiento'], 3)] for r in sc_sorted.tail(3).to_dict('records')]
R['sent_pct_neutro'] = round(float((df['sentimiento'] == 0).mean()) * 100, 1)
R['sent_pct_pos'] = round(float((df['sentimiento'] > 0).mean()) * 100, 1)
R['sent_pct_neg'] = round(float((df['sentimiento'] < 0).mean()) * 100, 1)

CAP = 250
dfm = df.groupby('libro', group_keys=False).sample(frac=1, random_state=42).groupby('libro').head(CAP).reset_index(drop=True)
R['muestra_n'] = int(len(dfm))
R['muestra_cap'] = CAP
prem = TextPreprocessor(stopwords=sw)
dfm['texto_procesado'] = prem.process_corpus(dfm['texto_original'].tolist())
vecm = TFIDFVectorizer()
M = vecm.fit_transform(dfm['texto_procesado'].tolist())
R['muestra_tfidf_shape'] = [int(M.shape[0]), int(M.shape[1])]

from sklearn.decomposition import PCA
Mp = M.astype(np.float32)
pca = PCA(n_components=2, svd_solver='randomized', random_state=0).fit(Mp)
R['pca_var'] = [round(float(v)*100, 2) for v in pca.explained_variance_ratio_]
viz.plot_pca_versiculos(M, dfm['testamento'], titulo="Versiculos por testamento (muestra)").savefig(f"{IMG}/pca_testamento.png", dpi=120)
viz.plot_pca_versiculos(M, dfm['genero'], titulo="Versiculos por genero literario (muestra)").savefig(f"{IMG}/pca_genero.png", dpi=120)

for modelo in ['logistic', 'naive_bayes']:
    clf = VerseClassifier(modelo=modelo)
    clf.entrenar(M, dfm['libro'])
    res = clf.evaluar()
    f1m = f1_score(res['y_test'], res['y_pred'], average='macro', labels=res['clases'], zero_division=0)
    R[f'clf_{modelo}_acc'] = round(float(res['accuracy']), 4)
    R[f'clf_{modelo}_f1'] = round(float(f1m), 4)
    if modelo == 'logistic':
        viz.plot_matriz_confusion(res['matriz_confusion'], res['clases']).savefig(f"{IMG}/matriz_confusion.png", dpi=110)
        rep = classification_report(res['y_test'], res['y_pred'], labels=res['clases'], output_dict=True, zero_division=0)
        porlibro = [(k, v['f1-score']) for k, v in rep.items() if k in set(res['clases'])]
        porlibro.sort(key=lambda x: x[1], reverse=True)
        R['clf_mejores_libros'] = [[a, round(b, 3)] for a, b in porlibro[:5]]
        R['clf_peores_libros'] = [[a, round(b, 3)] for a, b in porlibro[-5:]]

motor = SemanticSearchEngine(prem, TFIDFVectorizer()); motor.fit(dfm)
res_b = motor.buscar("love peace and faith", k=5)
R['busqueda_query'] = "love peace and faith"
R['busqueda_top5'] = [[r['libro'], int(r['capitulo']), int(r['versiculo']), round(r['similitud'], 3), r['texto_original'][:90]] for r in res_b.to_dict('records')]

json.dump(R, open('results.json', 'w'), ensure_ascii=False, indent=2)
print("OK -> results.json + figuras en imagenes/")
print("versiculos:", R['n_versiculos'], "| vocab:", R['n_vocab'], "| muestra:", R['muestra_n'])
print("clf logistic acc:", R['clf_logistic_acc'], "| naive_bayes acc:", R['clf_naive_bayes_acc'])
print("PCA var:", R['pca_var'])
