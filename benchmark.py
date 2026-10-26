"""Время и пиковая память пайплайна на датасетах разного размера: python3 benchmark.py"""
import time
import tracemalloc

from pipeline import add_dirt, drop_duplicates, fit_preprocessor, generate_data, train_test_split, transform


def run_once(n_rows: int) -> None:
    X, y = generate_data(n_rows)
    X, y = drop_duplicates(*add_dirt(X, y))
    X_train, X_test, _, _ = train_test_split(X, y)
    params = fit_preprocessor(X_train)
    transform(X_train, params)
    transform(X_test, params)


def measure(n_rows: int) -> tuple[float, float]:
    """Секунды и пик памяти (МБ) на один прогон."""
    tracemalloc.start()
    start = time.perf_counter()
    run_once(n_rows)
    seconds = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return seconds, peak / 1024**2


if __name__ == "__main__":
    print(f"{'строк':>10} | {'время, с':>9} | {'пик памяти, МБ':>14} | {'сам X, МБ':>9}")
    for n in (1_000, 10_000, 100_000, 1_000_000):
        seconds, peak_mb = measure(n)
        print(f"{n:>10,} | {seconds:>9.3f} | {peak_mb:>14.1f} | {n * 8 * 8 / 1024**2:>9.1f}")