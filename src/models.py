import pandas as pd

class Versiculo:
     def __init__(self, libro, capitulo, numero, texto_original):
        self.libro = libro
        self.capitulo = capitulo
        self.numero = numero
        self.texto_original = texto_original
        self.texto_procesado = []  

    def set_texto_procesado(self, tokens):
        self.texto_procesado = tokens

    def get_texto_limpio(self):
        return " ".join(self.texto_procesado)

    def __str__(self):
        return f"{self.libro} {self.capitulo}:{self.numero} - {self.texto_original[:40]}..."


class Capitulo:
    def __init__(self, numero):
        self.numero = numero
        self.versiculos = []

    def agregar_versiculo(self, versiculo):
        self.versiculos.append(versiculo)

    def get_cantidad_versiculos(self):
        return len(self.versiculos)

    def get_texto_completo(self):
        return " ".join(v.texto_original for v in self.versiculos)

    def __str__(self):
        return f"Capitulo {self.numero} ({self.get_cantidad_versiculos()} versiculos)"


class Libro:
    def __init__(self, nombre, testamento, genero=None):
        self.nombre = nombre
        self.testamento = testamento
        self.genero = genero
        self.capitulos = {}

    def set_genero(self, genero):
        self.genero = genero

    def get_genero(self):
        return self.genero

    def agregar_versiculo(self, versiculo):
        if versiculo.capitulo not in self.capitulos:
            self.capitulos[versiculo.capitulo] = Capitulo(versiculo.capitulo)
        self.capitulos[versiculo.capitulo].agregar_versiculo(versiculo)

    def get_versiculos(self):
        out = []
        for cap in self.capitulos.values():
            out.extend(cap.versiculos)
        return out

    def get_cantidad_versiculos(self):
        return len(self.get_versiculos())

    def get_cantidad_capitulos(self):
        return len(self.capitulos)

    def get_texto_completo(self):
        return " ".join(v.texto_original for v in self.get_versiculos())

    def __str__(self):
        return (f"{self.nombre} (Testamento: {self.testamento}, Genero: {self.genero}, "
                f"{self.get_cantidad_versiculos()} versiculos)")


class Testamento:
    def __init__(self, nombre):
        self.nombre = nombre
        self.libros = {}

    def agregar_libro(self, libro):
        self.libros[libro.nombre] = libro

    def get_versiculos(self):
        out = []
        for libro in self.libros.values():
            out.extend(libro.get_versiculos())
        return out

    def get_cantidad_versiculos(self):
        return len(self.get_versiculos())

    def __str__(self):
        return f"Testamento {self.nombre} ({len(self.libros)} libros)"


class Biblia:
    def __init__(self):
        self.testamentos = {}

    def agregar_versiculo(self, versiculo, testamento, nombre_libro, genero=None):
        if testamento not in self.testamentos:
            self.testamentos[testamento] = Testamento(testamento)
        test_obj = self.testamentos[testamento]

        if nombre_libro not in test_obj.libros:
            test_obj.agregar_libro(Libro(nombre_libro, testamento, genero))
        elif genero is not None and test_obj.libros[nombre_libro].get_genero() is None:
            test_obj.libros[nombre_libro].set_genero(genero)

        test_obj.libros[nombre_libro].agregar_versiculo(versiculo)

    @staticmethod
    def from_dataframe(df, col_libro="libro", col_testamento="testamento",
                       col_capitulo="capitulo", col_versiculo="versiculo",
                       col_texto="texto", col_genero="genero"):
        biblia = Biblia()
        tiene_genero = col_genero in df.columns

        for _, row in df.iterrows():
            versiculo = Versiculo(
                libro=row[col_libro],
                capitulo=int(row[col_capitulo]),
                numero=int(row[col_versiculo]),
                texto_original=str(row[col_texto]),
            )
            genero = row[col_genero] if tiene_genero else None
            biblia.agregar_versiculo(
                versiculo,
                testamento=row[col_testamento],
                nombre_libro=row[col_libro],
                genero=genero,
            )
        return biblia

    def get_libros(self):
        out = []
        for testamento in self.testamentos.values():
            out.extend(testamento.libros.values())
        return out

    def get_versiculos(self):
        out = []
        for libro in self.get_libros():
            out.extend(libro.get_versiculos())
        return out

    def get_generos(self):
        generos = []
        for libro in self.get_libros():
            g = libro.get_genero()
            if g is not None and g not in generos:
                generos.append(g)
        generos.sort()
        return generos

    def to_dataframe(self):
        rows = []
        for libro in self.get_libros():
            for versiculo in libro.get_versiculos():
                rows.append({
                    "testamento": libro.testamento,
                    "libro": libro.nombre,
                    "genero": libro.get_genero(),
                    "capitulo": versiculo.capitulo,
                    "versiculo": versiculo.numero,
                    "texto_original": versiculo.texto_original,
                    "texto_procesado": versiculo.texto_procesado,
                    "texto_limpio": versiculo.get_texto_limpio(),
                })
        return pd.DataFrame(rows)

    def get_resumen(self):
        rows = []
        for nombre, testamento in self.testamentos.items():
            rows.append({
                "testamento": nombre,
                "n_libros": len(testamento.libros),
                "n_versiculos": testamento.get_cantidad_versiculos(),
            })
        return pd.DataFrame(rows)

    def get_resumen_generos(self):
        rows = []
        for genero in self.get_generos():
            libros_genero = [l for l in self.get_libros() if l.get_genero() == genero]
            n_versiculos = sum(l.get_cantidad_versiculos() for l in libros_genero)
            rows.append({
                "genero": genero,
                "n_libros": len(libros_genero),
                "n_versiculos": n_versiculos,
            })
        return pd.DataFrame(rows)

    def __str__(self):
        return f"Biblia ({len(self.get_libros())} libros, {len(self.get_versiculos())} versiculos)"
