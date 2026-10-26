"""Пайплайн подготовки числового датасета на чистом NumPy."""
from pathlib import Path
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

def drop_duplicates(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Удаляет полностью совпадающие строки X (вместе с их y), оставляя первое вхождение.

    NaN != NaN, поэтому для сравнения пропуски временно заменяем на +inf.
    """
    key = np.where(np.isnan(X), np.inf, X)
    _, first = np.unique(key, axis=0, return_index=True)
    keep = np.sort(first)                  # исходный порядок строк
    return X[keep], y[keep]


def train_test_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.2,
                     seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Перемешивает строки и делит на train и test. Возвращает X_train, X_test, y_train, y_test."""
    if not 0 < test_size < 1:
        raise ValueError("test_size должен быть между 0 и 1")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_test = round(len(X) * test_size)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

# ---------- чистка: отдельные этапы ----------
def fill_missing(X: np.ndarray, medians: np.ndarray) -> np.ndarray:
    """NaN → медиана своего столбца."""
    return np.where(np.isnan(X), medians, X)


def clip_outliers(X: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """Всё, что ниже low или выше high, прижимаем к границе своего столбца."""
    return np.clip(X, low, high)


def scale(X: np.ndarray, center: np.ndarray, spread: np.ndarray) -> np.ndarray:
    """(X − center) / spread: стандартизация или MinMax, смотря что передали."""
    return (X - center) / spread


# ---------- fit на train, transform на чём угодно ----------
def fit_preprocessor(X_train: np.ndarray, method: str = "standard", iqr_k: float = 3.0) -> dict:
    """Считает параметры чистки ТОЛЬКО по train: медианы, границы выбросов, центр и разброс."""
    medians = np.nanmedian(X_train, axis=0)
    X = fill_missing(X_train, medians)
    q1, q3 = np.percentile(X, [25, 75], axis=0)
    iqr = q3 - q1
    low, high = q1 - iqr_k * iqr, q3 + iqr_k * iqr
    X = clip_outliers(X, low, high)
    if method == "standard":
        center, spread = X.mean(axis=0), X.std(axis=0)
    elif method == "minmax":
        center, spread = X.min(axis=0), X.max(axis=0) - X.min(axis=0)
    else:
        raise ValueError("method должен быть 'standard' или 'minmax'")
    spread = np.where(spread == 0, 1.0, spread)       # постоянный столбец: не делим на 0
    return {"method": method, "medians": medians, "low": low, "high": high,
            "center": center, "spread": spread}


def transform(X: np.ndarray, params: dict) -> np.ndarray:
    """Применяет параметры, посчитанные по train, к любым данным: train, test, новым."""
    X = fill_missing(X, params["medians"])
    X = clip_outliers(X, params["low"], params["high"])
    return scale(X, params["center"], params["spread"])

# ---------- проверка и сохранение ----------
def check_train(X_train: np.ndarray, method: str = "standard", atol: float = 1e-6) -> bool:
    """Train после чистки: нет NaN; standard → mean ≈ 0, std ≈ 1; minmax → min ≈ 0, max ≈ 1."""
    if np.isnan(X_train).any():
        return False
    if method == "standard":
        return bool(np.allclose(X_train.mean(axis=0), 0, atol=atol)
                    and np.allclose(X_train.std(axis=0), 1, atol=atol))
    return bool(np.allclose(X_train.min(axis=0), 0, atol=atol)
                and np.allclose(X_train.max(axis=0), 1, atol=atol))


def save_dataset(path: str | Path, X_train: np.ndarray, X_test: np.ndarray,
                 y_train: np.ndarray, y_test: np.ndarray, params: dict) -> Path:
    """Сохраняет выборки и параметры чистки в один .npz — чтобы потом так же обработать новые данные."""
    path = Path(path).with_suffix(".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
             **{f"param_{name}": value for name, value in params.items()})
    return path


def load_dataset(path: str | Path) -> tuple[dict, dict]:
    """Читает .npz обратно: словарь массивов и словарь параметров."""
    with np.load(path) as data:
        arrays = {k: data[k] for k in data.files if not k.startswith("param_")}
        params = {k.removeprefix("param_"): data[k] for k in data.files if k.startswith("param_")}
    params["method"] = str(params["method"])
    return arrays, params

# ---------- дополнительно: корреляции и батчи ----------
def correlation_matrix(X: np.ndarray) -> np.ndarray:
    """Корреляции Пирсона всех признаков со всеми: матрица (n_features, n_features)."""
    return np.corrcoef(X, rowvar=False)


def top_correlated_pairs(corr: np.ndarray, k: int = 3) -> list[tuple[int, int, float]]:
    """k пар признаков с самой сильной связью по модулю: [(i, j, r), ...]."""
    i, j = np.triu_indices(len(corr), k=1)            # пары над диагональю: каждая по одному разу
    order = np.argsort(-np.abs(corr[i, j]))[:k]        # по убыванию |r|
    return [(int(i[o]), int(j[o]), float(corr[i[o], j[o]])) for o in order]


def iter_batches(X: np.ndarray, y: np.ndarray, batch_size: int = 64, shuffle: bool = True,
                 seed: int = 42):
    """Выдаёт данные кусками по batch_size строк; последний кусок может быть меньше."""
    if batch_size <= 0:
        raise ValueError("batch_size должен быть больше нуля")
    idx = np.random.default_rng(seed).permutation(len(X)) if shuffle else np.arange(len(X))
    for start in range(0, len(X), batch_size):
        part = idx[start:start + batch_size]
        yield X[part], y[part]