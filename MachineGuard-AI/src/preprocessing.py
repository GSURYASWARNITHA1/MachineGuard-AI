"""Data preprocessing, input validation, feature engineering, and format alignment for MachineGuard AI."""
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import pandas as pd

from src.data import RAW_FEATURES, ENG_FEATURES, SHORT, add_engineered

# Standard aliases for flexible batch CSV column recognition
COLUMN_ALIASES = {
    # Air temperature
    "air temperature [k]": "air_temp",
    "air temperature": "air_temp",
    "air_temperature": "air_temp",
    "air_temp": "air_temp",
    "airtemp": "air_temp",
    "air_temp_k": "air_temp",
    "ambient_temp": "air_temp",
    
    # Process temperature
    "process temperature [k]": "proc_temp",
    "process temperature": "proc_temp",
    "process_temperature": "proc_temp",
    "proc_temp": "proc_temp",
    "proctemp": "proc_temp",
    "proc_temp_k": "proc_temp",
    "machine_temp": "proc_temp",

    # Rotational speed
    "rotational speed [rpm]": "rpm",
    "rotational speed": "rpm",
    "rotational_speed": "rpm",
    "rpm": "rpm",
    "speed": "rpm",
    "spindle_speed": "rpm",

    # Torque
    "torque [nm]": "torque",
    "torque": "torque",
    "torque_nm": "torque",
    "motor_torque": "torque",

    # Tool wear
    "tool wear [min]": "tool_wear",
    "tool wear": "tool_wear",
    "tool_wear": "tool_wear",
    "tool_wear_min": "tool_wear",
    "wear": "tool_wear",

    # Type
    "type": "type",
    "product type": "type",
    "product_type": "type",
    "machine_type": "type",
    "quality_type": "type"
}

REQUIRED_NUMERIC = ["air_temp", "proc_temp", "rpm", "torque", "tool_wear"]


def kelvin_to_celsius(k: float) -> float:
    """Convert Kelvin to Celsius."""
    return k - 273.15


def celsius_to_kelvin(c: float) -> float:
    """Convert Celsius to Kelvin."""
    return c + 273.15


def standardize_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Map arbitrary user CSV column names to internal standard names."""
    rename_map = {}
    for col in df.columns:
        norm = str(col).strip().lower()
        if norm in COLUMN_ALIASES:
            rename_map[col] = COLUMN_ALIASES[norm]
    standardized_df = df.rename(columns=rename_map)
    return standardized_df, rename_map


def prepare_single_input(
    air_temp: float,
    proc_temp: float,
    rpm: float,
    torque: float,
    tool_wear: float,
    product_type: str = "L",
    temp_unit: str = "K"
) -> pd.DataFrame:
    """Convert user UI inputs into a model-ready 1-row DataFrame."""
    if temp_unit == "°C":
        air_temp = celsius_to_kelvin(air_temp)
        proc_temp = celsius_to_kelvin(proc_temp)

    row = {
        "air_temp": float(air_temp),
        "proc_temp": float(proc_temp),
        "rpm": float(rpm),
        "torque": float(torque),
        "tool_wear": float(tool_wear),
        "type_L": 1 if product_type == "L" else 0,
        "type_M": 1 if product_type == "M" else 0,
        "type_H": 1 if product_type == "H" else 0,
    }
    df = pd.DataFrame([row])
    # Ensure exact column order matching RAW_FEATURES
    return df[RAW_FEATURES]


def validate_and_clean_batch(df: pd.DataFrame) -> Tuple[bool, List[str], List[str], Optional[pd.DataFrame]]:
    """
    Validate and clean an uploaded CSV for batch prediction.
    Returns: (is_valid, errors, warnings, cleaned_df)
    """
    errors: List[str] = []
    warnings: List[str] = []

    if df.empty:
        errors.append("Uploaded file is empty.")
        return False, errors, warnings, None

    std_df, rename_map = standardize_columns(df)

    # Check for required numeric columns
    missing_cols = [col for col in REQUIRED_NUMERIC if col not in std_df.columns]
    if missing_cols:
        errors.append(f"Missing required sensor columns: {', '.join(missing_cols)}. "
                      f"Expected columns: Air temperature, Process temperature, Rotational speed, Torque, Tool wear.")
        return False, errors, warnings, None

    cleaned_df = std_df.copy()

    # Product Type handling
    if "type" not in cleaned_df.columns:
        warnings.append("No 'Type' column found; defaulted to 'L' (Low quality, 60% of industrial fleet).")
        cleaned_df["type"] = "L"
    else:
        # Standardize type values to L, M, H
        cleaned_df["type"] = cleaned_df["type"].astype(str).str.strip().str.upper()
        invalid_types = ~cleaned_df["type"].isin(["L", "M", "H"])
        if invalid_types.any():
            count = invalid_types.sum()
            warnings.append(f"{count} rows had invalid product types (not L, M, or H). Defaulted these to 'L'.")
            cleaned_df.loc[invalid_types, "type"] = "L"

    # Numeric conversion & validation
    for col in REQUIRED_NUMERIC:
        cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors="coerce")
        null_count = cleaned_df[col].isna().sum()
        if null_count > 0:
            median_val = cleaned_df[col].median()
            if pd.isna(median_val):
                median_val = 300.0 if "temp" in col else (1500.0 if col == "rpm" else 40.0)
            cleaned_df[col] = cleaned_df[col].fillna(median_val)
            warnings.append(f"Column '{col}' contained {null_count} non-numeric/missing values; filled with median ({median_val:.2f}).")

    # Check if temperatures appear to be in Celsius (< 150) and auto-convert
    if (cleaned_df["air_temp"] < 150).all() and (cleaned_df["air_temp"] > -40).all():
        cleaned_df["air_temp"] = celsius_to_kelvin(cleaned_df["air_temp"])
        cleaned_df["proc_temp"] = celsius_to_kelvin(cleaned_df["proc_temp"])
        warnings.append("Detected temperatures in Celsius (<150°C); automatically converted to Kelvin.")

    # One-hot encode product types
    for t in ["L", "M", "H"]:
        cleaned_df[f"type_{t}"] = (cleaned_df["type"] == t).astype(int)

    # Format into RAW_FEATURES
    model_input_df = cleaned_df[RAW_FEATURES].copy()

    return True, errors, warnings, model_input_df


def compute_physical_indicators(
    air_temp: float,
    proc_temp: float,
    rpm: float,
    torque: float,
    tool_wear: float,
    product_type: str = "L"
) -> Dict[str, Any]:
    """Compute physical domain indicators and trigger warnings for AI4I failure modes."""
    temp_diff = proc_temp - air_temp
    power_w = torque * rpm * 2 * np.pi / 60
    strain = tool_wear * torque
    torque_per_rpm = torque / max(rpm, 1e-5)

    # Physical criteria checks based on AI4I specification:
    hdf_triggered = (temp_diff < 8.6) and (rpm < 1380)
    pwf_triggered = (power_w < 3500.0) or (power_w > 9000.0)
    
    strain_limit = {"L": 11000.0, "M": 12000.0, "H": 13000.0}.get(product_type, 11000.0)
    osf_triggered = strain > strain_limit
    twf_triggered = tool_wear >= 200.0

    triggers = []
    if hdf_triggered:
        triggers.append({
            "code": "HDF",
            "name": "Heat Dissipation Failure Trigger",
            "detail": f"ΔT is {temp_diff:.1f} K (< 8.6 K) & Speed is {rpm:.0f} rpm (< 1380 rpm). Inadequate airflow cooling."
        })
    if pwf_triggered:
        detail_msg = f"Mechanical power is {power_w:.0f} W (< 3,500 W lower limit)" if power_w < 3500 else f"Mechanical power is {power_w:.0f} W (> 9,000 W upper limit)"
        triggers.append({
            "code": "PWF",
            "name": "Power Failure Trigger",
            "detail": f"{detail_msg}. Severe electrical/mechanical drive overload."
        })
    if osf_triggered:
        triggers.append({
            "code": "OSF",
            "name": "Overstrain Failure Trigger",
            "detail": f"Overstrain (Wear × Torque) is {strain:.0f} min·Nm (exceeds {strain_limit:.0f} limit for Type {product_type})."
        })
    if twf_triggered:
        triggers.append({
            "code": "TWF",
            "name": "Tool Wear Warning",
            "detail": f"Cutting tool wear has reached {tool_wear:.0f} min (critical threshold: 200 min). Tool replacement due."
        })

    return {
        "temp_diff": temp_diff,
        "power_w": power_w,
        "strain": strain,
        "torque_per_rpm": torque_per_rpm,
        "strain_limit": strain_limit,
        "triggers": triggers,
        "hdf_triggered": hdf_triggered,
        "pwf_triggered": pwf_triggered,
        "osf_triggered": osf_triggered,
        "twf_triggered": twf_triggered
    }
