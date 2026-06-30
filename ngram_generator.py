"""
ngram_generator.py — NGramGenerator: modelos de lenguaje estadísticos.
Implementa unigram, bigram, trigram y n-gram configurable.
"""
import random
import math
from collections import defaultdict, Counter
from preprocessor import TextPreprocessor
from corpus import BibleCorpus

START = "<START>"
END = "<END>"


class NGramGenerator:
    """
    Generador de texto basado en modelos de lenguaje estadísticos n-gramas.
    
    Los n-gramas son secuencias de n palabras consecutivas. Este generador
    aprende patrones de transición entre palabras para generar texto artificial.
    
    Modelos soportados:
        - unigram (n=1): Probabilidad marginal de cada palabra
        - bigram (n=2): P(palabra_i | palabra anterior)
        - trigram (n=3): P(palabra_i | 2 palabras anteriores)
        - n-gram general: P(palabra_i | n-1 palabras anteriores)
    
    Tokens especiales:
        - <START>: Marca el inicio de una secuencia (versículo)
        - <END>: Marca el final de una secuencia
    """

    def __init__(self, corpus: BibleCorpus, preprocessor: TextPreprocessor):
        """
        Inicializa el generador de n-gramas.
        
        Args:
            corpus: BibleCorpus con versículos
            preprocessor: TextPreprocessor con tokens procesados
        """
        self.corpus = corpus
        self.preprocessor = preprocessor
        
        # Modelos construidos
        self._unigram: Counter = Counter()                # Frecuencias de palabras individuales
        self._ngram_models: dict[int, defaultdict] = {}   # int → {contexto: Counter}
        self._total_unigram_count: int = 0                # Total de palabras
        self._built_ns: set[int] = set()                  # N-valores que se han construido

    # ──────────────────────────────────────────────
    # Construcción de modelos
    # ──────────────────────────────────────────────
    
    def _verse_to_sequence(self, verse_index: int) -> list[str]:
        """
        Convierte un versículo en una secuencia con tokens especiales.
        
        Formato: <START> palabra1 palabra2 ... <END>
        
        Args:
            verse_index: Índice del versículo (en el corpus preprocesado)
            
        Returns:
            Lista de tokens incluyendo delimitadores
        """
        tokens = self.preprocessor.processed_verses[verse_index]
        if not tokens:
            return []
        return [START] + tokens + [END]

    def build(self, n_values: list[int] = None):
        """
        Construye los modelos de n-gramas especificados.
        
        Para cada valor de n, crea un modelo que almacena:
            contexto_de_n-1_palabras → Counter(palabras_que_siguen)
        
        Esto permite muestrear la siguiente palabra dado un contexto.
        
        Args:
            n_values: Lista de valores de n (ej: [1, 2, 3, 4]).
                     Si es None, usa [1, 2, 3, 4]
        """
        if n_values is None:
            n_values = [1, 2, 3, 4]

        print("[NGram] Construyendo modelos de lenguaje...")
        
        # Extraer secuencias de todos los versículos
        sequences = []
        for verse_index in range(len(self.corpus.verses)):
            sequence = self._verse_to_sequence(verse_index)
            if sequence:
                sequences.append(sequence)

        # Construir unigrama (frecuencia de cada palabra)
        for sequence in sequences:
            for word in sequence:
                self._unigram[word] += 1
        self._total_unigram_count = sum(self._unigram.values())

        # Construir n-gramas (donde n ≥ 2)
        for n in n_values:
            if n == 1:
                self._built_ns.add(1)
                continue
            
            ngram_model: defaultdict = defaultdict(Counter)
            
            for sequence in sequences:
                # Iterar sobre todas las n-tuplas en la secuencia
                for position in range(len(sequence) - n + 1):
                    # Contexto: las n-1 palabras anteriores
                    context = tuple(sequence[position : position + n - 1])
                    # Palabra siguiente
                    next_word = sequence[position + n - 1]
                    # Incrementar contador
                    ngram_model[context][next_word] += 1
            
            self._ngram_models[n] = ngram_model
            self._built_ns.add(n)

        print(f"[NGram] Modelos construidos para n = {sorted(self._built_ns)}")
        print(f"[NGram] Vocabulario unigrama: {len(self._unigram):,} tokens")

    # ──────────────────────────────────────────────
    # Muestreo (sampling)
    # ──────────────────────────────────────────────
    
    def _sample_from_unigram(self, exclude: set = None) -> str:
        """
        Muestrea una palabra según su frecuencia en el unigrama.
        
        Las palabras más frecuentes tienen mayor probabilidad de ser seleccionadas.
        
        Args:
            exclude: Conjunto de palabras a no considerar (ej: {<START>, <END>})
            
        Returns:
            Palabra muestreada
        """
        words = list(self._unigram.keys())
        weights = list(self._unigram.values())
        
        if exclude:
            filtered = [(w, weight) for w, weight in zip(words, weights)
                        if w not in exclude]
            if filtered:
                words, weights = zip(*filtered)
        
        return random.choices(words, weights=weights, k=1)[0]

    def _sample_from_counter(self, word_counts: Counter) -> str:
        """
        Muestrea una palabra según su frecuencia en un Counter.
        
        Args:
            word_counts: Counter con palabras y sus frecuencias
            
        Returns:
            Palabra muestreada
        """
        words = list(word_counts.keys())
        weights = list(word_counts.values())
        
        return random.choices(words, weights=weights, k=1)[0]

    # ──────────────────────────────────────────────
    # Generación de texto
    # ──────────────────────────────────────────────
    
    def generate(self, n: int, seed_word: str = None,
                 max_len: int = 30, temperature: float = 1.0) -> str:
        """
        Genera una secuencia de texto usando el modelo n-grama de orden n.
        
        Proceso:
            - Para n=1 (unigrama): Cada palabra es independiente
            - Para n≥2: Usa el contexto anterior para predecir la siguiente palabra
        
        Args:
            n: Orden del modelo (debe estar en los construidos con build())
            seed_word: Palabra inicial (si None, se muestrea aleatoriamente)
            max_len: Longitud máxima de la salida (mínimo 15 palabras)
            temperature: Control de aleatoriedad (unused en esta versión)
            
        Returns:
            Cadena de texto generada (sin tokens <START> y <END>)
        """
        if n not in self._built_ns:
            raise ValueError(f"Modelo n={n} no construido. Usa build([{n}]).")

        max_len = max(max_len, 15)  # Asegurar longitud mínima
        generated_words = []

        if n == 1:
            # Para unigrama: cada palabra es independiente
            if seed_word and seed_word.lower() in self._unigram:
                generated_words.append(seed_word.lower())
            else:
                word = self._sample_from_unigram(exclude={START, END})
                generated_words.append(word)
            
            # Generar resto de palabras
            while len(generated_words) < max_len:
                word = self._sample_from_unigram(exclude={START, END})
                generated_words.append(word)
                if word == END:
                    break
            
            return ' '.join([w for w in generated_words if w not in (START, END)])

        # Para n ≥ 2: usar contexto
        context = [START] * (n - 1)  # Inicializar contexto con START tokens

        # Determinar la primera palabra
        if seed_word and seed_word.lower() in self._unigram:
            first_word = seed_word.lower()
        else:
            context_key = tuple(context)
            if context_key in self._ngram_models[n]:
                word_counts = self._ngram_models[n][context_key]
                first_word = self._sample_from_counter(word_counts)
            else:
                first_word = self._sample_from_unigram(exclude={START, END})

        # Asegurar que la primera palabra no sea especial
        if first_word in (END, START):
            first_word = self._sample_from_unigram(exclude={START, END})
        
        generated_words.append(first_word)
        context = context[1:] + [first_word]

        # Generar resto de la secuencia
        for _ in range(max_len - 1):
            context_key = tuple(context)
            
            if context_key in self._ngram_models[n]:
                word_counts = self._ngram_models[n][context_key]
                next_word = self._sample_from_counter(word_counts)
            else:
                # Si el contexto no existe, usar unigrama (backoff)
                next_word = self._sample_from_unigram(exclude={START, END})

            # Detener si encontramos fin de secuencia
            if next_word == END:
                break
            
            # No permitir START en medio de generación
            if next_word == START:
                next_word = self._sample_from_unigram(exclude={START, END})
            
            generated_words.append(next_word)
            context = context[1:] + [next_word]

        # Retornar solo palabras reales, sin tokens especiales
        return ' '.join(generated_words)

    def compare_models(self, seed_word: str = None,
                       max_len: int = 20) -> dict[str, str]:
        """
        Genera texto usando todos los modelos construidos.
        
        Útil para comparar cómo diferentes órdenes de n-grama generan texto.
        
        Args:
            seed_word: Palabra inicial para la generación
            max_len: Longitud máxima de salida
            
        Returns:
            Diccionario mapeando nombres de modelos a textos generados
        """
        model_names = {1: 'Unigram', 2: 'Bigram', 3: 'Trigram', 4: '4-gram'}
        results = {}
        
        for n in sorted(self._built_ns):
            model_name = model_names.get(n, f'{n}-gram')
            generated_text = self.generate(n, seed_word=seed_word, max_len=max_len)
            results[model_name] = generated_text
        
        return results

    def perplexity(self, n: int, test_sequences: list[list[str]]) -> float:
        """
        Calcula la perplejidad del modelo en secuencias de prueba.
        
        La perplejidad mide qué tan sorprendente encuentra el modelo las secuencias.
        Valores bajos indican mejor predicción.
        
        Fórmula: Perplexity = exp(-mean(log_prob))
        
        Args:
            n: Orden del modelo a evaluar
            test_sequences: Lista de secuencias de tokens para prueba
            
        Returns:
            Valor de perplejidad (float)
        """
        if n not in self._built_ns:
            return float('inf')
        
        log_probability_sum = 0.0
        count_total = 0
        vocabulary_size = len(self._unigram)

        for sequence in test_sequences:
            # Agregar tokens de inicio y fin
            full_sequence = [START] * (n - 1) + sequence + [END]
            
            # Iterar sobre posiciones en la secuencia completa
            for position in range(n - 1, len(full_sequence)):
                # Construir contexto: n-1 palabras anteriores
                context = tuple(full_sequence[position - n + 1 : position])
                word = full_sequence[position]
                
                if n == 1:
                    # Para unigrama: probabilidad simple
                    prob = (self._unigram.get(word, 0) + 1) / (self._total_unigram_count + vocabulary_size)
                else:
                    # Para n-gramas: probabilidad suavizada
                    word_counts = self._ngram_models[n].get(context, Counter())
                    context_total = sum(word_counts.values())
                    prob = (word_counts.get(word, 0) + 1) / (context_total + vocabulary_size) if context_total > 0 else 1 / vocabulary_size
                
                # Acumular log-probabilidad
                log_probability_sum += math.log(prob + 1e-12)
                count_total += 1

        if count_total == 0:
            return float('inf')
        
        # Perplejidad = exp(-media de log-probabilidades)
        return math.exp(-log_probability_sum / count_total)
