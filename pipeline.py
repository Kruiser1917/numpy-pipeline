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

def add_dirt(X: np.ndarray, y: np.ndarray, nan_frac: float = 0.02, n_outliers: int = 20,
             n_duplicates: int = 15, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Портит данные, как бывает в жизни: пропуски NaN, выбросы, дубли строк.

    Исходные X и y не меняет — работает с копией.
    """
    rng = np.random.default_rng(seed)
    X = X.copy()                       # без копии испортили бы исходный X снаружи функции
    n_rows, n_features = X.shape
    means, stds = X.mean(axis=0), X.std(axis=0)

    # выбросы: случайные клетки уезжают на 8–12 std от среднего своего столбца
    rows = rng.integers(0, n_rows, n_outliers)
    cols = rng.integers(0, n_features, n_outliers)
    signs = rng.choice([-1, 1], n_outliers)
    X[rows, cols] = means[cols] + signs * rng.uniform(8, 12, n_outliers) * stds[cols]

    # пропуски: каждая клетка с вероятностью nan_frac становится NaN
    X[rng.random(X.shape) < nan_frac] = np.nan

    # дубли: копируем случайные строки целиком, добавляем в конец и перемешиваем всё
    dup = rng.choice(n_rows, n_duplicates, replace=False)
    X = np.vstack([X, X[dup]])
    y = np.concatenate([y, y[dup]])
    order = rng.permutation(len(X))
    return X[order], y[order]