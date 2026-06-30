import random
from collections import defaultdict, Counter

START = "<START>"
END = "<END>"

class NGramModel:
    def __init__(self, n):
        if n < 1:
            raise ValueError("n debe ser >= 1")
        self.n = n
        self.contexto_size = n - 1
        self.conteos = defaultdict(Counter)
        self.vocabulario = set()

    def fit(self, oraciones_tokenizadas):
        for tokens in oraciones_tokenizadas:
            secuencia = [START] * self.contexto_size + tokens + [END]
            self.vocabulario.update(tokens)
            for i in range(len(secuencia) - self.contexto_size):
                contexto = tuple(secuencia[i:i + self.contexto_size])
                siguiente = secuencia[i + self.contexto_size]
                self.conteos[contexto][siguiente] += 1
        return self

    def get_probabilidad(self, contexto, palabra):
        contador = self.conteos.get(contexto)
        if not contador:
            return 0.0
        total = sum(contador.values())
        return contador[palabra] / total

    def get_siguiente_palabra(self, contexto):
        contador = self.conteos.get(contexto)
        if not contador:
            return random.choice(list(self.vocabulario)) if self.vocabulario else END
        palabras = list(contador.keys())
        pesos = list(contador.values())
        return random.choices(palabras, weights=pesos, k=1)[0]

    def generar(self, palabra_inicial=None, max_len=30):
        if palabra_inicial:
            if self.contexto_size > 0:
                contexto = tuple([START] * (self.contexto_size - 1) + [palabra_inicial])
            else:
                contexto = tuple()
            generadas = [palabra_inicial]
        else:
            contexto = tuple([START] * self.contexto_size)
            generadas = []

        for _ in range(max_len):
            siguiente = self.get_siguiente_palabra(contexto)
            if siguiente == END:
                break
            generadas.append(siguiente)
            if self.contexto_size > 0:
                contexto = tuple((list(contexto) + [siguiente])[-self.contexto_size:])

        return " ".join(generadas)

    def __str__(self):
        return f"NGramModel(n={self.n}, vocabulario={len(self.vocabulario)} palabras)"


def comparar_modelos(oraciones_tokenizadas, ns=(1, 2, 3, 4), palabra_inicial=None, max_len=20):
    resultados = {}
    for n in ns:
        modelo = NGramModel(n).fit(oraciones_tokenizadas)
        resultados[n] = modelo.generar(palabra_inicial=palabra_inicial, max_len=max_len)
    return resultados
