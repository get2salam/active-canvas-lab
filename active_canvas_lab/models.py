"""From-scratch logistic regression with online SGD and L2 regularisation.

No external libraries — arithmetic uses only Python built-ins and math.
The numerically stable sigmoid avoids overflow for large |z|.
"""
import math
from typing import List


class LogisticRegression:
    """Binary logistic regression trained with stochastic gradient descent."""

    def __init__(self, lr: float = 0.05, n_epochs: int = 100, l2: float = 1e-3):
        self.lr = lr
        self.n_epochs = n_epochs
        self.l2 = l2
        self.w: List[float] = []
        self.b: float = 0.0
        self._fitted = False

    # ------------------------------------------------------------------
    @staticmethod
    def _sigmoid(z: float) -> float:
        """Numerically stable sigmoid."""
        if z >= 0.0:
            return 1.0 / (1.0 + math.exp(-z))
        e = math.exp(z)
        return e / (1.0 + e)

    @staticmethod
    def _dot(w: list, x: list) -> float:
        return sum(a * b for a, b in zip(w, x))

    # ------------------------------------------------------------------
    def fit(self, X: list, y: list) -> "LogisticRegression":
        if not X:
            return self
        d = len(X[0])
        if len(self.w) != d:
            self.w = [0.0] * d
            self.b = 0.0
        lr, l2 = self.lr, self.l2
        for _ in range(self.n_epochs):
            for xi, yi in zip(X, y):
                z = self._dot(self.w, xi) + self.b
                err = self._sigmoid(z) - yi
                for j in range(d):
                    self.w[j] -= lr * (err * xi[j] + l2 * self.w[j])
                self.b -= lr * err
        self._fitted = True
        return self

    def predict_proba(self, X: list) -> List[float]:
        """Return P(y=1) for each sample."""
        if not self._fitted:
            return [0.5] * len(X)
        return [self._sigmoid(self._dot(self.w, xi) + self.b) for xi in X]

    def predict(self, X: list) -> List[int]:
        return [1 if p >= 0.5 else 0 for p in self.predict_proba(X)]

    def score(self, X: list, y: list) -> float:
        if not y:
            return 0.0
        preds = self.predict(X)
        return sum(p == yi for p, yi in zip(preds, y)) / len(y)
