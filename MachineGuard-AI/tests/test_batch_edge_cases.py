import pytest
import pandas as pd
import numpy as np

from src.preprocessing import validate_and_clean_batch
from src.prediction import predict_batch


def test_batch_empty_dataframe():
    df = pd.DataFrame()
    res = predict_batch(df)
    assert res["success"] is False
    assert any("empty" in e.lower() for e in res["errors"])


def test_batch_missing_critical_column():
    # Missing torque
    df = pd.DataFrame({
        "Air temperature [K]": [300.0, 301.0],
        "Process temperature [K]": [310.0, 311.0],
        "Rotational speed [rpm]": [1500, 1600],
        "Tool wear [min]": [20, 30]
    })
    res = predict_batch(df)
    assert res["success"] is False
    assert any("torque" in e.lower() for e in res["errors"])


def test_batch_missing_type_defaults_to_L():
    # Omit type column
    df = pd.DataFrame({
        "air_temp": [300.0, 301.0],
        "proc_temp": [310.0, 311.0],
        "rpm": [1500, 1600],
        "torque": [40.0, 42.0],
        "tool_wear": [20, 30]
    })
    ok, errors, warnings, clean_df = validate_and_clean_batch(df)
    assert ok is True
    assert clean_df["type_L"].iloc[0] == 1
    assert any("defaulted to 'l'" in w.lower() for w in warnings)


def test_batch_with_null_and_corrupted_strings():
    # Corrupted rows
    df = pd.DataFrame({
        "air_temp": [300.0, "corrupted", 302.0],
        "proc_temp": [310.0, np.nan, 312.0],
        "rpm": [1500, 1600, 1700],
        "torque": [40.0, 42.0, 45.0],
        "tool_wear": [20, 30, 40],
        "type": ["L", "UNKNOWN_TYPE", "H"]
    })
    res = predict_batch(df)
    assert res["success"] is True
    assert len(res["results_df"]) == 3
    # Check that warnings captured the issues
    assert len(res["warnings"]) > 0
