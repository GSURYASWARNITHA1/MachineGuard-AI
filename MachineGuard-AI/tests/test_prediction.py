import pytest
from pathlib import Path
import pandas as pd
import numpy as np

from src.prediction import (
    ModelManager,
    predict_single,
    predict_batch,
    get_risk_category,
    PRESETS
)
from config import DATA_DIR, DECISION_THRESHOLD


def test_model_manager_loads_successfully():
    bundle = ModelManager.get_bundle()
    assert "model" in bundle
    assert "threshold" in bundle
    assert 0.05 <= bundle["threshold"] <= 0.95


def test_risk_categorization_thresholds():
    cat_high = get_risk_category(0.40, threshold=0.34)
    assert cat_high["level"] == "High Risk"
    assert cat_high["priority"] == "CRITICAL"

    cat_med = get_risk_category(0.25, threshold=0.34)
    assert cat_med["level"] == "Medium Risk"
    assert cat_med["priority"] == "ADVISORY"

    cat_low = get_risk_category(0.05, threshold=0.34)
    assert cat_low["level"] == "Low Risk"
    assert cat_low["priority"] == "NOMINAL"


def test_predict_single_nominal_preset():
    preset = PRESETS["Healthy Nominal Machine"]
    res = predict_single(
        air_temp=preset["air_temp"],
        proc_temp=preset["proc_temp"],
        rpm=preset["rpm"],
        torque=preset["torque"],
        tool_wear=preset["tool_wear"],
        product_type=preset["product_type"]
    )
    assert res["probability"] < DECISION_THRESHOLD
    assert res["risk_level"] == "Low Risk"
    assert res["is_failure"] is False
    assert len(res["recommendations"]) > 0


def test_predict_single_failure_preset():
    preset = PRESETS["Overstrain Failure (OSF)"]
    res = predict_single(
        air_temp=preset["air_temp"],
        proc_temp=preset["proc_temp"],
        rpm=preset["rpm"],
        torque=preset["torque"],
        tool_wear=preset["tool_wear"],
        product_type=preset["product_type"]
    )
    assert res["probability"] >= DECISION_THRESHOLD
    assert res["risk_level"] == "High Risk"
    assert res["is_failure"] is True
    assert any(t["code"] == "OSF" for t in res["triggers"])


def test_predict_batch_sample_fleet():
    sample_path = DATA_DIR / "sample_fleet.csv"
    assert sample_path.exists()
    df = pd.read_csv(sample_path)
    res = predict_batch(df)
    assert res["success"] is True
    assert res["results_df"] is not None
    assert len(res["results_df"]) == len(df)
    assert "Failure_Probability" in res["results_df"].columns
    assert "Risk_Category" in res["results_df"].columns
    assert "Recommended_Action" in res["results_df"].columns
    assert res["summary"]["total_machines"] == 25
    assert res["summary"]["high_risk_count"] > 0
    assert res["summary"]["low_risk_count"] > 0
