import pytest
import pandas as pd
import numpy as np

from src.preprocessing import (
    prepare_single_input,
    standardize_columns,
    validate_and_clean_batch,
    compute_physical_indicators,
    celsius_to_kelvin,
    kelvin_to_celsius
)
from src.data import RAW_FEATURES


def test_temperature_unit_conversion():
    c = 25.0
    k = celsius_to_kelvin(c)
    assert np.isclose(k, 298.15)
    assert np.isclose(kelvin_to_celsius(k), 25.0)


def test_prepare_single_input_format():
    df = prepare_single_input(
        air_temp=300.0,
        proc_temp=310.0,
        rpm=1500,
        torque=40.0,
        tool_wear=50,
        product_type="M",
        temp_unit="K"
    )
    assert list(df.columns) == RAW_FEATURES
    assert df["type_M"].iloc[0] == 1
    assert df["type_L"].iloc[0] == 0
    assert df["type_H"].iloc[0] == 0
    assert df["air_temp"].iloc[0] == 300.0


def test_prepare_single_input_celsius_toggle():
    df = prepare_single_input(
        air_temp=25.0,
        proc_temp=35.0,
        rpm=1500,
        torque=40.0,
        tool_wear=50,
        product_type="L",
        temp_unit="°C"
    )
    assert np.isclose(df["air_temp"].iloc[0], 298.15)
    assert np.isclose(df["proc_temp"].iloc[0], 308.15)


def test_standardize_columns_with_aliases():
    raw_df = pd.DataFrame({
        "Air Temperature": [300.0],
        "spindle_speed": [1500],
        "Torque [Nm]": [40.0],
        "Wear": [50],
        "machine_temp": [310.0],
        "Product Type": ["L"]
    })
    std_df, rename_map = standardize_columns(raw_df)
    assert "air_temp" in std_df.columns
    assert "rpm" in std_df.columns
    assert "torque" in std_df.columns
    assert "tool_wear" in std_df.columns
    assert "proc_temp" in std_df.columns
    assert "type" in std_df.columns


def test_physical_indicators_hdf():
    # HDF condition: temp_diff < 8.6 and rpm < 1380
    res = compute_physical_indicators(
        air_temp=303.0,
        proc_temp=310.0,  # diff = 7.0 K
        rpm=1300,
        torque=45.0,
        tool_wear=50,
        product_type="L"
    )
    assert res["hdf_triggered"] is True
    assert any(t["code"] == "HDF" for t in res["triggers"])


def test_physical_indicators_pwf():
    # PWF condition: power < 3500 W or > 9000 W
    # Power = torque * rpm * 2 * pi / 60
    # For torque=40, rpm=2600 -> power = 40 * 2600 * 2 * pi / 60 = 10,890 W > 9000
    res = compute_physical_indicators(
        air_temp=298.0,
        proc_temp=308.0,
        rpm=2600,
        torque=40.0,
        tool_wear=50,
        product_type="L"
    )
    assert res["pwf_triggered"] is True
    assert any(t["code"] == "PWF" for t in res["triggers"])


def test_physical_indicators_osf():
    # OSF condition: strain (tool_wear * torque) > 11,000 for Type L
    # For tool_wear=200, torque=65 -> strain = 13,000 > 11,000
    res = compute_physical_indicators(
        air_temp=298.0,
        proc_temp=308.0,
        rpm=1400,
        torque=65.0,
        tool_wear=200,
        product_type="L"
    )
    assert res["osf_triggered"] is True
    assert any(t["code"] == "OSF" for t in res["triggers"])


def test_validate_and_clean_batch_success():
    df = pd.DataFrame({
        "Air temperature [K]": [298.1, 298.2],
        "Process temperature [K]": [308.6, 308.7],
        "Rotational speed [rpm]": [1551, 1408],
        "Torque [Nm]": [42.8, 46.3],
        "Tool wear [min]": [0, 3],
        "Type": ["M", "L"]
    })
    ok, errors, warnings, clean_df = validate_and_clean_batch(df)
    assert ok is True
    assert len(errors) == 0
    assert len(clean_df) == 2
    assert list(clean_df.columns) == RAW_FEATURES
