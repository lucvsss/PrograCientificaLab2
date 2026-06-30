import pandas as pd

PALABRAS_POSITIVAS = {
    "love", "loved", "loveth", "loving", "peace", "joy", "joyful", "rejoice",
    "rejoiced", "glad", "gladness", "bless", "blessed", "blessing", "blessings",
    "mercy", "merciful", "grace", "gracious", "good", "goodness", "righteous",
    "righteousness", "holy", "light", "life", "salvation", "saved", "save",
    "hope", "faith", "faithful", "truth", "wisdom", "wise", "comfort",
    "comforted", "glory", "glorious", "praise", "thanksgiving", "delight",
    "beautiful", "beauty", "kind", "kindness", "gentle", "gift", "heaven",
    "healed", "heal", "forgive", "forgiven", "redeem", "redeemed", "pure",
    "honour", "honor", "strength", "strong", "victory", "abundant", "prosper",
    "reward", "favour", "favor", "sweet", "pleasant", "loyal", "compassion",
}

PALABRAS_NEGATIVAS = {
    "death", "die", "died", "dead", "kill", "killed", "slay", "slain", "slew",
    "sin", "sins", "sinned", "sinner", "evil", "wicked", "wickedness",
    "iniquity", "wrath", "anger", "angry", "fury", "destroy", "destroyed",
    "destruction", "curse", "cursed", "fear", "afraid", "terror", "weep",
    "wept", "weeping", "mourn", "mourning", "sorrow", "grief", "pain",
    "suffer", "suffering", "affliction", "torment", "punish", "punishment",
    "plague", "famine", "sword", "war", "blood", "bloodshed", "enemy",
    "enemies", "darkness", "hell", "judgment", "condemn", "condemned",
    "guilt", "shame", "hate", "hated", "hatred", "wound", "wounded",
    "perish", "perished", "tears", "trouble", "distress", "vanity", "lament",
    "abomination", "wilderness", "desolate", "desolation", "captivity",
}


def build_default_lexicon():
    lexico = {}
    for palabra in PALABRAS_POSITIVAS:
        lexico[palabra] = 1
    for palabra in PALABRAS_NEGATIVAS:
        lexico[palabra] = -1
    return lexico


class SentimentAnalyzer:
    def score(self, texto):
        raise NotImplementedError


class LexiconSentimentAnalyzer(SentimentAnalyzer):
    def __init__(self, lexico=None):
        self.lexico = lexico if lexico is not None else build_default_lexicon()

    def score(self, texto):
        if not isinstance(texto, str):
            return 0.0
          
        palabras = texto.lower().split()

        positivas = 0
        negativas = 0
        for palabra in palabras:
            palabra = palabra.strip(".,;:!?\"'()[]")
            peso = self.lexico.get(palabra)
            if peso == 1:
                positivas += 1
            elif peso == -1:
                negativas += 1

        total = positivas + negativas
        if total == 0:
            return 0.0
        return (positivas - negativas) / total

    def __str__(self):
        return f"LexiconSentimentAnalyzer({len(self.lexico)} palabras en el lexico)"


class TextBlobSentimentAnalyzer(SentimentAnalyzer):
    def __init__(self):
        from textblob import TextBlob
        self.text_blob_class = TextBlob

    def score(self, texto):
        if not isinstance(texto, str):
            return 0.0
        return self.text_blob_class(texto).sentiment.polarity


def calcular_sentimiento_corpus(df, analyzer, columna_texto="texto_original"):
    df = df.copy()
    df["sentimiento"] = df[columna_texto].apply(analyzer.score)
    return df


def agregar_por_libro(df):
    return (
        df.groupby("libro")["sentimiento"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
        .sort_values("mean")
    )


def agregar_por_capitulo(df):
    return (
        df.groupby(["libro", "capitulo"])["sentimiento"]
        .mean()
        .reset_index()
    )
