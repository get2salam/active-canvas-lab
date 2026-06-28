"""Active learning query strategies.

Each strategy selects `n_query` indices from the unlabeled pool.
All strategies share the same interface via the Strategy base class.
"""
import random
from typing import List, Optional


def _dist_sq(a: list, b: list) -> float:
    """Squared Euclidean distance between two 2D points."""
    return sum((ai - bi) ** 2 for ai, bi in zip(a, b))


class Strategy:
    name: str = "base"

    def query(
        self,
        model,
        X_pool: list,
        n_query: int,
        X_labeled: Optional[list] = None,
        rng: Optional[random.Random] = None,
    ) -> List[int]:
        """Return sorted list of at most n_query indices into X_pool."""
        raise NotImplementedError


class RandomStrategy(Strategy):
    """Uniform random baseline — sets the floor for comparisons."""

    name = "random"

    def query(self, model, X_pool, n_query, X_labeled=None, rng=None):
        rng = rng or random.Random()
        idx = list(range(len(X_pool)))
        rng.shuffle(idx)
        return sorted(idx[:n_query])


class UncertaintyStrategy(Strategy):
    """Least-confidence sampling — pick points where P(y=1) is closest to 0.5."""

    name = "uncertainty"

    def query(self, model, X_pool, n_query, X_labeled=None, rng=None):
        probs = model.predict_proba(X_pool)
        ranked = sorted(range(len(X_pool)), key=lambda i: abs(probs[i] - 0.5))
        return sorted(ranked[:n_query])


class MarginStrategy(Strategy):
    """Minimum-margin sampling — smallest |2p − 1| for binary classification."""

    name = "margin"

    def query(self, model, X_pool, n_query, X_labeled=None, rng=None):
        probs = model.predict_proba(X_pool)
        ranked = sorted(range(len(X_pool)), key=lambda i: abs(2 * probs[i] - 1))
        return sorted(ranked[:n_query])


class DiversityStrategy(Strategy):
    """Greedy farthest-first traversal — maximise coverage of input space.

    At each step, picks the unlabeled point that is farthest from all
    already-labeled (and already-selected) anchor points. Falls back to
    random when no labeled set is available.
    """

    name = "diversity"

    def query(self, model, X_pool, n_query, X_labeled=None, rng=None):
        rng = rng or random.Random()
        if not X_labeled:
            idx = list(range(len(X_pool)))
            rng.shuffle(idx)
            return sorted(idx[:n_query])

        selected: list[int] = []
        remaining = list(range(len(X_pool)))
        anchor = list(X_labeled)  # grows as we select

        for _ in range(min(n_query, len(remaining))):
            best = max(
                remaining,
                key=lambda i: min(_dist_sq(X_pool[i], a) for a in anchor),
            )
            selected.append(best)
            remaining.remove(best)
            anchor.append(X_pool[best])

        return sorted(selected)


STRATEGIES: dict = {
    "random":      RandomStrategy,
    "uncertainty": UncertaintyStrategy,
    "margin":      MarginStrategy,
    "diversity":   DiversityStrategy,
}


def get_strategy(name: str) -> Strategy:
    """Instantiate a strategy by name."""
    if name not in STRATEGIES:
        raise ValueError(
            f"Unknown strategy '{name}'. Choices: {sorted(STRATEGIES)}"
        )
    return STRATEGIES[name]()
