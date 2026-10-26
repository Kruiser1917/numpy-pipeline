"""Тесты пайплайна: каждый этап на маленьких понятных примерах + прогон целиком."""
import numpy as np
import pytest

from pipeline import (add_dirt, check_train, clip_outliers, correlation_matrix, drop_duplicates,
                      fill_missing, fit_preprocessor, generate_data, iter_batches, load_dataset,
                      save_dataset, top_correlated_pairs, train_test_split, transform)
from run import run


# ---------- данные ----------
def test_generate_data_shape_and_seed():
    X, y = generate_data(500, seed=1)
    assert X.shape == (500, 8) and y.shape == (500,)
    X2, y2 = generate_data(500, seed=1)
    assert np.array_equal(X, X2) and np.array_equal(y, y2)
    assert np.corrcoef(X[:, 0], X[:, 1])[0, 1] > 0.8          # связь, которую мы заложили


def test_add_dirt_does_not_touch_input():
    X, y = generate_data(200)
    X_before = X.copy()
    Xd, yd = add_dirt(X, y, n_duplicates=10)
    assert np.array_equal(X, X_before)                        # исходник цел
    assert Xd.shape == (210, 8) and yd.shape == (210,)
    assert np.isnan(Xd).any()


# ---------- дубли и разбиение ----------
def test_drop_duplicates_handles_nan():
    X = np.array([[1, np.nan], [2, 3], [1, np.nan], [2, 3], [5, 6]])
    y = np.array([10, 20, 10, 20, 50])
    Xu, yu = drop_duplicates(X, y)
    assert len(Xu) == 3
    assert yu.tolist() == [10, 20, 50]                        # пары X↔y и порядок сохранены


def test_split_sizes_and_no_overlap():
    X = np.arange(100, dtype=float).reshape(-1, 1)            # в каждой строке её номер
    y = np.arange(100)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, seed=0)
    assert len(X_tr) == 75 and len(X_te) == 25
    assert sorted(np.concatenate([y_tr, y_te]).tolist()) == list(range(100))   # ничего не потеряли и не задвоили
    assert np.array_equal(X_tr[:, 0], y_tr)                   # пары не разъехались


@pytest.mark.parametrize("bad", [0, 1, -0.1, 1.5])
def test_split_bad_test_size(bad):
    with pytest.raises(ValueError):
        train_test_split(np.zeros((10, 2)), np.zeros(10), test_size=bad)


# ---------- этапы чистки ----------
def test_fill_missing_and_clip():
    X = np.array([[1.0, np.nan], [np.nan, 4.0], [100.0, 5.0]])
    filled = fill_missing(X, medians=np.array([2.0, 3.0]))
    assert filled.tolist() == [[1, 3], [2, 4], [100, 5]]
    clipped = clip_outliers(filled, low=np.array([0, 0]), high=np.array([10, 10]))
    assert clipped[2, 0] == 10


def test_standard_scaling_on_train():
    X, y = generate_data()
    X_tr, X_te, _, _ = train_test_split(*drop_duplicates(*add_dirt(X, y)))
    params = fit_preprocessor(X_tr)
    X_tr_s, X_te_s = transform(X_tr, params), transform(X_te, params)
    assert check_train(X_tr_s, "standard")
    assert not np.isnan(X_te_s).any()
    assert not np.allclose(X_te_s.mean(axis=0), 0, atol=1e-6)  # test не «подогнан» — утечки нет


def test_minmax_scaling():
    X, _ = generate_data()
    params = fit_preprocessor(X, method="minmax")
    assert check_train(transform(X, params), "minmax")


def test_constant_column_does_not_break():
    X = np.column_stack([np.ones(50), np.arange(50.0)])
    Xs = transform(X, fit_preprocessor(X))
    assert not np.isnan(Xs).any()
    assert np.all(Xs[:, 0] == 0)


def test_bad_method():
    with pytest.raises(ValueError):
        fit_preprocessor(np.ones((5, 2)), method="robust")


def test_check_train_catches_problems():
    X, _ = generate_data()
    assert not check_train(X, "standard")                     # сырые данные не стандартизированы
    X_nan = transform(X, fit_preprocessor(X))
    X_nan[0, 0] = np.nan
    assert not check_train(X_nan, "standard")


# ---------- сохранение ----------
def test_save_and_load_roundtrip(tmp_path):
    X, y = generate_data(100)
    params = fit_preprocessor(X)
    path = save_dataset(tmp_path / "out", X, X[:10], y, y[:10], params)
    assert path.suffix == ".npz"
    arrays, loaded = load_dataset(path)
    assert np.array_equal(arrays["X_train"], X)
    assert loaded["method"] == "standard"
    assert np.array_equal(transform(X, loaded), transform(X, params))   # те же параметры → тот же результат


# ---------- корреляции и батчи ----------
def test_top_correlated_pairs_finds_planted_pair():
    rng = np.random.default_rng(0)
    a = rng.normal(size=300)
    X = np.column_stack([a, rng.normal(size=300), -2 * a + 1])
    corr = correlation_matrix(X)
    assert corr.shape == (3, 3)
    i, j, r = top_correlated_pairs(corr, k=1)[0]
    assert (i, j) == (0, 2) and np.isclose(r, -1)


def test_batches_cover_all_rows_once():
    X = np.arange(130, dtype=float).reshape(-1, 1)
    y = np.arange(130)
    batches = list(iter_batches(X, y, batch_size=64))
    assert [len(b[0]) for b in batches] == [64, 64, 2]
    assert sorted(np.concatenate([b[1] for b in batches]).tolist()) == list(range(130))
    first = next(iter_batches(X, y, batch_size=5, shuffle=False))
    assert first[1].tolist() == [0, 1, 2, 3, 4]
    with pytest.raises(ValueError):
        next(iter_batches(X, y, batch_size=0))


# ---------- целиком ----------
def test_run_end_to_end(tmp_path):
    out = tmp_path / "dataset.npz"
    assert run(n_rows=300, method="minmax", out=str(out))
    assert out.exists()