import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.data import (ENG_FEATURES, MODE_COLS, RAW_FEATURES, add_engineered, build_frame, load_raw, verify)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def df():
    return load_raw()


def test_dataset_matches_uci_published_counts(df):
    assert all(ok for _, _, ok in verify(df).values())


def test_no_leakage_columns_in_features(df):
    X, y, _ = build_frame(df)
    leaked = set(MODE_COLS) | {"Machine failure", "UDI", "Product ID"}
    assert not leaked & set(X.columns)
    assert list(X.columns) == RAW_FEATURES


def test_engineered_features_are_finite_and_correct(df):
    X, _, _ = build_frame(df)
    E = add_engineered(X)
    assert list(E.columns) == ENG_FEATURES
    assert np.isfinite(E.values).all()
    assert np.allclose(E["temp_diff"], X["proc_temp"] - X["air_temp"])


def test_saved_model_predicts_valid_probabilities(df):
    bundle = joblib.load(ROOT / "models" / "model.joblib")
    X, _, _ = build_frame(df)
    p = bundle["model"].predict_proba(X.head(200))[:, 1]
    assert p.shape == (200,) and ((p >= 0) & (p <= 1)).all()
    assert 0.05 <= bundle["threshold"] <= 0.95


def test_metrics_file_is_consistent_with_saved_predictions():
    m = json.loads((ROOT / "results" / "metrics.json").read_text())
    p = np.load(ROOT / "results" / "test_probs.npy")
    y = np.load(ROOT / "results" / "test_labels.npy")
    thr = m["test"]["calibrated_at_cost_threshold"]["threshold"]
    pred = p >= thr
    assert int(((pred) & (y == 1)).sum()) == m["test"]["calibrated_at_cost_threshold"]["tp"]
    assert int(((~pred) & (y == 1)).sum()) == m["test"]["calibrated_at_cost_threshold"]["fn"]
    assert m["split"]["test_failures"] == int(y.sum())


def test_test_set_never_used_for_model_selection():
    m = json.loads((ROOT / "results" / "metrics.json").read_text())
    best = max(m["cv_comparison"], key=lambda r: r["pr_auc"])
    assert (best["model"], best["features"]) == (m["selected"]["model"], m["selected"]["features"])
