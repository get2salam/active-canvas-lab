"""Deterministic synthetic 2D binary classification datasets.

All generators use Python's random module with Box-Muller for normal sampling,
so the only dependency is the standard library.
"""
import math
import random


def _gauss(mu: float, sigma: float, rng: random.Random) -> float:
    """Box-Muller normal sample (avoids log(0) by retrying u==0)."""
    while True:
        u = rng.random()
        if u > 0.0:
            break
    v = rng.random()
    z = math.sqrt(-2.0 * math.log(u)) * math.cos(2.0 * math.pi * v)
    return mu + sigma * z


def make_blobs(n_samples: int = 200, seed: int = 0):
    """Two isotropic Gaussian clusters, linearly separable."""
    rng = random.Random(seed)
    X, y = [], []
    for i in range(n_samples):
        label = i % 2
        cx, cy = (-1.5, -1.5) if label == 0 else (1.5, 1.5)
        X.append([_gauss(cx, 0.8, rng), _gauss(cy, 0.8, rng)])
        y.append(label)
    idx = list(range(n_samples))
    rng.shuffle(idx)
    return [X[i] for i in idx], [y[i] for i in idx]


def make_rings(n_samples: int = 200, seed: int = 0):
    """Two concentric rings — not linearly separable."""
    rng = random.Random(seed)
    X, y = [], []
    radii = [1.0, 2.8]
    for i in range(n_samples):
        label = i % 2
        r = radii[label]
        angle = rng.uniform(0.0, 2.0 * math.pi)
        noise = _gauss(0.0, 0.15, rng)
        X.append(
            [(r + noise) * math.cos(angle), (r + noise) * math.sin(angle)]
        )
        y.append(label)
    idx = list(range(n_samples))
    rng.shuffle(idx)
    return [X[i] for i in idx], [y[i] for i in idx]


def make_diagonal(n_samples: int = 200, seed: int = 0):
    """Linearly separable along x+y=0 with a noisy decision band."""
    rng = random.Random(seed)
    X, y = [], []
    for _ in range(n_samples):
        x0 = _gauss(0.0, 1.5, rng)
        x1 = _gauss(0.0, 1.5, rng)
        margin = x0 + x1
        if abs(margin) < 0.3:
            label = rng.randint(0, 1)  # ambiguous band
        else:
            label = 1 if margin > 0 else 0
        X.append([x0, x1])
        y.append(label)
    return X, y


def make_islands(n_samples: int = 200, seed: int = 0):
    """Six scattered micro-clusters — three per class, challenging geometry."""
    rng = random.Random(seed)
    centers = [
        ((-2.5,  2.0), 0),
        (( 2.0, -2.5), 0),
        (( 0.0,  0.0), 0),
        (( 2.5,  2.0), 1),
        ((-2.5, -2.0), 1),
        (( 0.0, -3.0), 1),
    ]
    per_center = n_samples // len(centers)
    X, y = [], []
    for (cx, cy), label in centers:
        for _ in range(per_center):
            X.append([_gauss(cx, 0.45, rng), _gauss(cy, 0.45, rng)])
            y.append(label)
    while len(X) < n_samples:
        X.append([_gauss(0.0, 2.0, rng), _gauss(0.0, 2.0, rng)])
        y.append(rng.randint(0, 1))
    idx = list(range(len(X)))
    rng.shuffle(idx)
    return [X[i] for i in idx], [y[i] for i in idx]


DATASETS: dict = {
    "blobs":    make_blobs,
    "rings":    make_rings,
    "diagonal": make_diagonal,
    "islands":  make_islands,
}


def load_dataset(name: str, n_samples: int = 200, seed: int = 0):
    """Return (X, y) for the named dataset."""
    if name not in DATASETS:
        raise ValueError(
            f"Unknown dataset '{name}'. Choices: {sorted(DATASETS)}"
        )
    return DATASETS[name](n_samples=n_samples, seed=seed)
