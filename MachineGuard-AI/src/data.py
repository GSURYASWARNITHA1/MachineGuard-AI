"""Data loading, leakage removal and feature engineering for AI4I 2020."""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "ai4i2020.csv"

# Published UCI figures, used to verify the file before any modelling.
EXPECTED = {"rows": 10000, "failures": 339, "HDF": 115, "PWF": 95, "OSF": 98, "TWF": 46, "RNF": 19}

TARGET = "Machine failure"
# Failure-mode flags are the target decomposed: using them as inputs is leakage.
MODE_COLS = ["TWF", "HDF", "PWF", "OSF", "RNF"]
# Row identifiers carry no physical signal (Product ID encodes Type plus a serial).
ID_COLS = ["UDI", "Product ID"]

RAW_NUMERIC = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
SHORT = {
    "Air temperature [K]": "air_temp",
    "Process temperature [K]": "proc_temp",
    "Rotational speed [rpm]": "rpm",
    "Torque [Nm]": "torque",
    "Tool wear [min]": "tool_wear",
}
RAW_FEATURES = ["type_L", "type_M", "type_H"] + list(SHORT.values())
ENGINEERED = ["temp_diff", "power_w", "strain", "torque_per_rpm"]
ENG_FEATURES = RAW_FEATURES + ENGINEERED


def load_raw(path=DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def verify(df: pd.DataFrame) -> dict:
    """Compare the loaded file against UCI's published counts."""
    got = {"rows": len(df), "failures": int(df[TARGET].sum())}
    got.update({m: int(df[m].sum()) for m in ["HDF", "PWF", "OSF", "TWF", "RNF"]})
    return {k: (got[k], EXPECTED[k], got[k] == EXPECTED[k]) for k in EXPECTED}


def build_frame(df: pd.DataFrame):
    """Return (X_raw, y, type_series). Leakage and ID columns are dropped here."""
    X = df.drop(columns=ID_COLS + MODE_COLS + [TARGET]).rename(columns=SHORT)
    types = X["Type"].copy()
    for t in ["L", "M", "H"]:
        X[f"type_{t}"] = (X["Type"] == t).astype(int)
    X = X.drop(columns="Type")
    return X[RAW_FEATURES], df[TARGET].astype(int), types


def add_engineered(X: pd.DataFrame) -> pd.DataFrame:
    """Physics-motivated features that mirror how the failure modes arise."""
    X = X.copy()
    X["temp_diff"] = X["proc_temp"] - X["air_temp"]  # heat dissipation
    X["power_w"] = X["torque"] * X["rpm"] * 2 * np.pi / 60  # mechanical power
    X["strain"] = X["tool_wear"] * X["torque"]  # overstrain
    X["torque_per_rpm"] = X["torque"] / X["rpm"]
    return X[ENG_FEATURES]


def failure_modes(df: pd.DataFrame) -> pd.DataFrame:
    return df[MODE_COLS]


# Module-level (not lambdas, not defined in __main__) so a saved model can be unpickled anywhere.
def select_raw(X: pd.DataFrame) -> pd.DataFrame:
    return X[RAW_FEATURES]


def select_engineered(X: pd.DataFrame) -> pd.DataFrame:
    return add_engineered(X)[ENG_FEATURES]
