"""Пайплайн подготовки числового датасета на чистом NumPy."""
import numpy as np

# у каждого признака свой масштаб — как в реальных данных (возраст, доход, проценты...)
FEATURE_MEANS = np.array([50, 0, 1000, 5, -20, 0.5, 300, 10], dtype=float)
FEATURE_STDS = np.array([10, 1, 250, 2, 5, 0.1, 80, 3], dtype=float)


def generate_data(n_rows: int = 1000, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Синтетический датасет: X (n_rows, 8) с признаками в разных масштабах и целевой вектор y.

    Признак 1 специально зависит от признака 0 — будет что найти корреляцией.
    """
    rng = np.random.default_rng(seed)
    n_features = len(FEATURE_MEANS)
    X = rng.normal(FEATURE_MEANS, FEATURE_STDS, size=(n_rows, n_features))
    X[:, 1] = 0.08 * (X[:, 0] - 50) + rng.normal(0, 0.3, n_rows)
    weights = rng.normal(0, 1, n_features) / FEATURE_STDS
    y = X @ weights + rng.normal(0, 0.5, n_rows)
    return X, y