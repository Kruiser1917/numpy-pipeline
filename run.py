"""Пайплайн целиком одной командой.

Примеры:
    python3 run.py
    python3 run.py --method minmax --rows 5000 --out data/minmax.npz
"""
import argparse

import numpy as np

from pipeline import (add_dirt, check_train, drop_duplicates, fit_preprocessor, generate_data,
                      save_dataset, train_test_split, transform)


def run(n_rows: int = 1000, method: str = "standard", seed: int = 42,
        out: str = "data/dataset.npz") -> bool:
    """Проходит все этапы, печатает, что произошло на каждом, и возвращает результат проверки."""
    X, y = generate_data(n_rows, seed)
    print(f"1. generate:        X {X.shape}, y {y.shape}")

    X, y = add_dirt(X, y, seed=seed)
    print(f"2. add_dirt:        строк {len(X)}, NaN {np.isnan(X).sum()}")

    n_before = len(X)
    X, y = drop_duplicates(X, y)
    print(f"3. drop_duplicates: удалено {n_before - len(X)} → строк {len(X)}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, seed=seed)
    print(f"4. split:           train {len(X_train)}, test {len(X_test)}")

    params = fit_preprocessor(X_train, method)
    X_train, X_test = transform(X_train, params), transform(X_test, params)
    print(f"5. fit/transform:   {method}, NaN осталось {np.isnan(X_train).sum() + np.isnan(X_test).sum()}")

    ok = check_train(X_train, method)
    target = "mean ≈ 0, std ≈ 1" if method == "standard" else "min ≈ 0, max ≈ 1"
    print(f"6. check:           train {target} → {'OK' if ok else 'ОШИБКА'}")

    path = save_dataset(out, X_train, X_test, y_train, y_test, params)
    print(f"7. save:            {path} ({path.stat().st_size / 1024:.1f} КБ)")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description="NumPy-пайплайн подготовки данных")
    parser.add_argument("--rows", type=int, default=1000, help="сколько строк сгенерировать")
    parser.add_argument("--method", choices=["standard", "minmax"], default="standard",
                        help="способ масштабирования")
    parser.add_argument("--seed", type=int, default=42, help="seed для воспроизводимости")
    parser.add_argument("--out", default="data/dataset.npz", help="куда сохранить результат")
    args = parser.parse_args()
    ok = run(args.rows, args.method, args.seed, args.out)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()