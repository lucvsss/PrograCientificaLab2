from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


class VerseClassifier:

    def __init__(self, modelo="logistic", maximo_iteraciones=1000):
        if modelo == "logistic":
            self.modelo = LogisticRegression(max_iter=maximo_iteraciones)
        elif modelo == "linear_svm":
            self.modelo = LinearSVC(max_iter=maximo_iteraciones)
        elif modelo == "naive_bayes":
            self.modelo = MultinomialNB()
        else:
            raise ValueError("modelo debe ser 'logistic' o 'naive_bayes'")

        self.nombre_modelo = modelo
        self.clases = None
        self.X_test = None
        self.y_test = None

    def entrenar(self, X, y, test_size=0.2, random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        self.modelo.fit(X_train, y_train)
        self.clases = self.modelo.classes_
        self.X_test = X_test
        self.y_test = y_test
        return X_train, X_test, y_train, y_test

    def evaluar(self):
        if self.X_test is None:
            raise RuntimeError("Llama a entrenar() antes de evaluar().")

        y_pred = self.modelo.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        matriz_confusion = confusion_matrix(self.y_test, y_pred, labels=self.clases)
        reporte = classification_report(self.y_test, y_pred, labels=self.clases, zero_division=0)
        return {
            "accuracy": accuracy,
            "matriz_confusion": matriz_confusion,
            "clases": self.clases,
            "reporte": reporte,
            "y_test": self.y_test,
            "y_pred": y_pred,
        }

    def predecir(self, X_nuevo):
        return self.modelo.predict(X_nuevo)

    def __str__(self):
        return f"VerseClassifier(modelo={self.nombre_modelo}, clases={self.clases})"
