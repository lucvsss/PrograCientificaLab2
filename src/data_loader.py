import pandas as pd

def cargar_dataset(path_bible, path_key, path_genre):
    df = pd.read_csv(path_bible)
    df = df.rename(columns={
        "id": "Verse ID",
        "b": "Book",
        "c": "Chapter",
        "v": "Verse",
        "t": "Text",
    })

    df_key = pd.read_csv(path_key)
    df_key = df_key.rename(columns={
        "b": "Book",
        "n": "Book Name",
        "t": "Testament (OT or NT)",
        "g": "Genre ID",
    })

    df_genre = pd.read_csv(path_genre)
    df_genre = df_genre.rename(columns={
        "g": "Genre ID",
        "n": "Genre name",
    })

    df = pd.merge(df, df_key, how="inner", on="Book")
    df = pd.merge(df, df_genre, how="inner", on="Genre ID")
    return df
