# perceptron.py
import numpy as np

class Perceptron:
    """
    Perceptrón simple con:
    - pesos (vector)
    - bias (umbral representado como bias; net = w·x + bias)
    """
    def __init__(self, n_inputs):
        self.n_inputs = n_inputs
        self.weights = np.zeros(n_inputs, dtype=float)
        self.bias = 0.0
        # scaler for optional normalization
        self._x_min = None
        self._x_max = None

    # --- Normalización simple (min-max) ---
    def fit_normalizer(self, X):
        self._x_min = X.min(axis=0)
        self._x_max = X.max(axis=0)

    def normalize(self, X):
        if self._x_min is None:
            return X.astype(float)
        denom = (self._x_max - self._x_min).astype(float)
        denom[denom == 0] = 1.0
        return (X - self._x_min) / denom

    # --- Inicialización de pesos ---
    def initialize_weights(self, weights=None, bias=None, random_init=True, seed=None):
        rng = np.random.RandomState(seed)
        if weights is not None:
            w = np.array(weights, dtype=float)
            if w.shape[0] != self.n_inputs:
                raise ValueError("Número de pesos no coincide con número de entradas")
            self.weights = w
        elif random_init:
            self.weights = rng.uniform(-1.0, 1.0, size=self.n_inputs)
        if bias is not None:
            self.bias = float(bias)
        elif random_init:
            self.bias = float(rng.uniform(-1.0,1.0))

    # --- Predicción ---
    def predict_raw(self, X):
        X = np.asarray(X, dtype=float)
        return np.dot(X, self.weights) + self.bias

    def predict(self, X):
        raw = self.predict_raw(X)
        return (raw >= 0).astype(int)

    # --- Entrenamiento: regla clásica del perceptrón (online) ---
    def train_perceptron_rule(self, X, y, eta=0.1, max_iter=100, tol=1e-3, callback=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        errors = []
        n = len(y)
        for it in range(max_iter):
            total_error = 0.0
            for xi, yi in zip(X, y):
                net = float(np.dot(xi, self.weights) + self.bias)
                y_pred = 1 if net >= 0 else 0
                err = yi - y_pred
                self.weights += eta * err * xi
                self.bias += eta * err
                total_error += err**2
            mse = total_error / n
            errors.append(mse)
            if callback:
                callback(it, mse)
            if mse <= tol:
                break
        return errors

    # --- Entrenamiento: regla delta (gradiente MSE) ---
    def train_delta_rule(self, X, y, eta=0.01, max_iter=100, tol=1e-3, callback=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        errors = []
        n = len(y)
        for it in range(max_iter):
            total_error = 0.0
            for xi, yi in zip(X, y):
                net = float(np.dot(xi, self.weights) + self.bias)
                err = yi - net
                self.weights += eta * err * xi
                self.bias += eta * err
                total_error += err**2
            mse = total_error / n
            errors.append(mse)
            if callback:
                callback(it, mse)
            if mse <= tol:
                break
        return errors

    # --- Evaluación (retorna dict con métricas simples) ---
    def evaluate(self, X, y):
        y = np.asarray(y, dtype=int)
        y_pred = self.predict(X)
        acc = (y_pred == y).mean()
        tp = int(((y == 1) & (y_pred == 1)).sum())
        tn = int(((y == 0) & (y_pred == 0)).sum())
        fp = int(((y == 0) & (y_pred == 1)).sum())
        fn = int(((y == 1) & (y_pred == 0)).sum())
        return {"accuracy": acc, "tp": tp, "tn": tn, "fp": fp, "fn": fn}