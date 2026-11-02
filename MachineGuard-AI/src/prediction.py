"""Inference engine, risk categorization, actionable recommendations, and explainability for MachineGuard AI."""
from typing import Dict, Any, List, Optional
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from config import MODEL_PATH, DECISION_THRESHOLD, MEDIUM_RISK_THRESHOLD, SENSOR_SPECS
from src.preprocessing import (
    prepare_single_input,
    validate_and_clean_batch,
    compute_physical_indicators,
    kelvin_to_celsius
)
from src.data import RAW_FEATURES, add_engineered

# Nominal industrial operating medians from the 10,000 AI4I dataset
NOMINAL_BASELINES = {
    "air_temp": 300.0,      # ~26.85 °C
    "proc_temp": 310.0,     # ~36.85 °C
    "rpm": 1496.0,          # ~1500 rpm
    "torque": 40.1,         # ~40 Nm
    "tool_wear": 108.0,     # min
    "temp_diff": 10.0,      # K
    "power_w": 6280.0,      # W
    "strain": 4330.0        # min·Nm
}


class ModelManager:
    """Singleton model loader and caching manager."""
    _instance = None
    _bundle = None

    @classmethod
    def get_bundle(cls, path: Path = MODEL_PATH) -> Dict[str, Any]:
        if cls._bundle is None:
            if not path.exists():
                raise FileNotFoundError(f"Model file not found at {path}. Please run training first.")
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                cls._bundle = joblib.load(path)
        return cls._bundle

    @classmethod
    def get_model(cls, path: Path = MODEL_PATH):
        return cls.get_bundle(path)["model"]

    @classmethod
    def get_threshold(cls, path: Path = MODEL_PATH) -> float:
        return float(cls.get_bundle(path).get("threshold", DECISION_THRESHOLD))


def get_risk_category(prob: float, threshold: float = DECISION_THRESHOLD) -> Dict[str, str]:
    """Map calibrated probability to industrial risk category, badge color, and priority."""
    if prob >= threshold:
        return {
            "level": "High Risk",
            "badge_color": "#ef4444",
            "icon": "🚨",
            "priority": "CRITICAL",
            "status": "Action Required"
        }
    elif prob >= MEDIUM_RISK_THRESHOLD:
        return {
            "level": "Medium Risk",
            "badge_color": "#f59e0b",
            "icon": "⚠️",
            "priority": "ADVISORY",
            "status": "Diagnostic Recommended"
        }
    else:
        return {
            "level": "Low Risk",
            "badge_color": "#10b981",
            "icon": "✅",
            "priority": "NOMINAL",
            "status": "Normal Operation"
        }


def generate_recommendations(
    risk_level: str,
    prob: float,
    triggers: List[Dict[str, str]],
    indicators: Dict[str, Any],
    tool_wear: float
) -> List[str]:
    """Generate human-readable engineering recommendations for maintenance technicians."""
    recs = []
    if risk_level == "High Risk":
        recs.append("CRITICAL: Machine failure probability exceeds safety threshold (cost penalty for missed failure is 10x).")
        recs.append("IMMEDIATE ACTION: Dispatch maintenance technician for physical inspection before next production batch.")
        
        # Specific physical trigger suggestions
        for trigger in triggers:
            code = trigger["code"]
            if code == "HDF":
                recs.append("HDF Action: Clear spindle cooling vents, verify ambient air circulation, and check heat exchanger coolant flow.")
            elif code == "PWF":
                recs.append("PWF Action: Check motor drive electrical supply, calibrate spindle inverter, and inspect gear train for binding.")
            elif code == "OSF":
                recs.append("OSF Action: Reduce feed rate or depth of cut immediately. Tool wear combined with excessive torque is stressing tool holder.")
            elif code == "TWF":
                recs.append("TWF Action: Replace cutting insert immediately. Tool cumulative cut time exceeds wear threshold.")
        
        if not triggers:
            recs.append("General Action: Perform full multi-sensor diagnostic; multi-factor strain detected.")
            
    elif risk_level == "Medium Risk":
        recs.append("ADVISORY: Machine exhibits early warning signs of mechanical or thermal degradation.")
        recs.append("MAINTENANCE SCHEDULE: Queue for routine maintenance check during the next scheduled shift changeover.")
        if indicators.get("power_w", 0) > 8000 or indicators.get("power_w", 0) < 4000:
            recs.append("Monitor power output stability on current production run.")
        if tool_wear > 170:
            recs.append(f"Tool wear is at {tool_wear:.0f} min. Prepare replacement tooling ahead of the 200 min threshold.")
    else:
        recs.append("NOMINAL STATUS: Machine operating safely within certified operational boundaries.")
        recs.append("ROUTINE: Continue standard automated monitoring and log sensor telemetry.")

    return recs


def analyze_sensor_deviations(
    air_temp: float,
    proc_temp: float,
    rpm: float,
    torque: float,
    tool_wear: float,
    indicators: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Compare input values against dataset nominal baselines to explain anomaly drivers."""
    deviations = [
        {
            "sensor": "Air Temp",
            "value": f"{air_temp:.1f} K ({kelvin_to_celsius(air_temp):.1f} °C)",
            "nominal": f"{NOMINAL_BASELINES['air_temp']:.1f} K",
            "pct_diff": ((air_temp - NOMINAL_BASELINES['air_temp']) / NOMINAL_BASELINES['air_temp']) * 100,
            "status": "Elevated" if air_temp > 303 else ("Depressed" if air_temp < 297 else "Normal")
        },
        {
            "sensor": "Process Temp",
            "value": f"{proc_temp:.1f} K ({kelvin_to_celsius(proc_temp):.1f} °C)",
            "nominal": f"{NOMINAL_BASELINES['proc_temp']:.1f} K",
            "pct_diff": ((proc_temp - NOMINAL_BASELINES['proc_temp']) / NOMINAL_BASELINES['proc_temp']) * 100,
            "status": "Overheating" if proc_temp > 312 else "Normal"
        },
        {
            "sensor": "Rotational Speed",
            "value": f"{rpm:.0f} rpm",
            "nominal": f"{NOMINAL_BASELINES['rpm']:.0f} rpm",
            "pct_diff": ((rpm - NOMINAL_BASELINES['rpm']) / NOMINAL_BASELINES['rpm']) * 100,
            "status": "Low RPM (HDF risk)" if rpm < 1380 else ("High RPM" if rpm > 2200 else "Normal")
        },
        {
            "sensor": "Torque",
            "value": f"{torque:.1f} Nm",
            "nominal": f"{NOMINAL_BASELINES['torque']:.1f} Nm",
            "pct_diff": ((torque - NOMINAL_BASELINES['torque']) / NOMINAL_BASELINES['torque']) * 100,
            "status": "High Torque" if torque > 55 else ("Low Torque" if torque < 20 else "Normal")
        },
        {
            "sensor": "Tool Wear",
            "value": f"{tool_wear:.0f} min",
            "nominal": f"{NOMINAL_BASELINES['tool_wear']:.0f} min",
            "pct_diff": ((tool_wear - NOMINAL_BASELINES['tool_wear']) / NOMINAL_BASELINES['tool_wear']) * 100,
            "status": "Critical Wear" if tool_wear >= 200 else ("High Wear" if tool_wear >= 160 else "Normal")
        },
        {
            "sensor": "Mechanical Power",
            "value": f"{indicators['power_w']:.0f} W",
            "nominal": f"{NOMINAL_BASELINES['power_w']:.0f} W",
            "pct_diff": ((indicators['power_w'] - NOMINAL_BASELINES['power_w']) / NOMINAL_BASELINES['power_w']) * 100,
            "status": "Out of Spec (<3.5kW or >9kW)" if (indicators['power_w'] < 3500 or indicators['power_w'] > 9000) else "Normal"
        },
        {
            "sensor": "Overstrain Factor",
            "value": f"{indicators['strain']:.0f} min·Nm",
            "nominal": f"{NOMINAL_BASELINES['strain']:.0f} min·Nm",
            "pct_diff": ((indicators['strain'] - NOMINAL_BASELINES['strain']) / NOMINAL_BASELINES['strain']) * 100,
            "status": "Overstrain Alert" if indicators['osf_triggered'] else "Normal"
        }
    ]
    return deviations


def predict_single(
    air_temp: float,
    proc_temp: float,
    rpm: float,
    torque: float,
    tool_wear: float,
    product_type: str = "L",
    temp_unit: str = "K"
) -> Dict[str, Any]:
    """Execute end-to-end prediction for a single machine and generate diagnostic explanation."""
    bundle = ModelManager.get_bundle()
    model = bundle["model"]
    threshold = float(bundle.get("threshold", DECISION_THRESHOLD))

    # Format input
    input_df = prepare_single_input(
        air_temp=air_temp,
        proc_temp=proc_temp,
        rpm=rpm,
        torque=torque,
        tool_wear=tool_wear,
        product_type=product_type,
        temp_unit=temp_unit
    )

    # Actual Kelvin values used
    actual_air_k = float(input_df["air_temp"].iloc[0])
    actual_proc_k = float(input_df["proc_temp"].iloc[0])

    # Calibrated probability
    prob = float(model.predict_proba(input_df)[0, 1])
    is_failure = bool(prob >= threshold)

    # Risk categorization
    risk_info = get_risk_category(prob, threshold)

    # Physical domain indicators
    indicators = compute_physical_indicators(
        air_temp=actual_air_k,
        proc_temp=actual_proc_k,
        rpm=rpm,
        torque=torque,
        tool_wear=tool_wear,
        product_type=product_type
    )

    # Actionable recommendations
    recommendations = generate_recommendations(
        risk_level=risk_info["level"],
        prob=prob,
        triggers=indicators["triggers"],
        indicators=indicators,
        tool_wear=tool_wear
    )

    # Deviation analysis
    deviations = analyze_sensor_deviations(
        air_temp=actual_air_k,
        proc_temp=actual_proc_k,
        rpm=rpm,
        torque=torque,
        tool_wear=tool_wear,
        indicators=indicators
    )

    return {
        "probability": prob,
        "is_failure": is_failure,
        "threshold": threshold,
        "risk_level": risk_info["level"],
        "badge_color": risk_info["badge_color"],
        "icon": risk_info["icon"],
        "priority": risk_info["priority"],
        "status": risk_info["status"],
        "recommendations": recommendations,
        "triggers": indicators["triggers"],
        "indicators": indicators,
        "deviations": deviations,
        "input_features": input_df.to_dict(orient="records")[0]
    }


def predict_batch(raw_df: pd.DataFrame) -> Dict[str, Any]:
    """Execute batch prediction on an uploaded DataFrame."""
    bundle = ModelManager.get_bundle()
    model = bundle["model"]
    threshold = float(bundle.get("threshold", DECISION_THRESHOLD))

    is_valid, errors, warnings, model_input_df = validate_and_clean_batch(raw_df)
    if not is_valid:
        return {
            "success": False,
            "errors": errors,
            "warnings": warnings,
            "results_df": None,
            "summary": None
        }

    probs = model.predict_proba(model_input_df)[:, 1]
    
    results_df = raw_df.copy()
    results_df["Failure_Probability"] = np.round(probs, 4)
    results_df["Risk_Score_Pct"] = np.round(probs * 100, 1)
    
    # Categorization
    conditions = [
        probs >= threshold,
        (probs >= MEDIUM_RISK_THRESHOLD) & (probs < threshold),
        probs < MEDIUM_RISK_THRESHOLD
    ]
    categories = ["High Risk", "Medium Risk", "Low Risk"]
    actions = ["Immediate Inspection", "Schedule Diagnostic", "Normal Operation"]
    
    results_df["Risk_Category"] = np.select(conditions, categories, default="Unknown")
    results_df["Recommended_Action"] = np.select(conditions, actions, default="Routine")
    results_df["Prediction"] = np.where(probs >= threshold, "Failure Predicted", "Healthy")

    # Add quick physical trigger detection for batch rows
    p_air = model_input_df["air_temp"].values
    p_proc = model_input_df["proc_temp"].values
    p_rpm = model_input_df["rpm"].values
    p_torque = model_input_df["torque"].values
    p_wear = model_input_df["tool_wear"].values
    
    hdf_flags = ((p_proc - p_air) < 8.6) & (p_rpm < 1380)
    pwf_flags = ((p_torque * p_rpm * 2 * np.pi / 60) < 3500) | ((p_torque * p_rpm * 2 * np.pi / 60) > 9000)
    twf_flags = p_wear >= 200
    
    suspect_modes = []
    for i in range(len(results_df)):
        modes = []
        if hdf_flags[i]: modes.append("HDF")
        if pwf_flags[i]: modes.append("PWF")
        if twf_flags[i]: modes.append("TWF")
        suspect_modes.append(", ".join(modes) if modes else "None")
    
    results_df["Suspected_Modes"] = suspect_modes

    # Summary Statistics
    total = len(results_df)
    high_count = int((results_df["Risk_Category"] == "High Risk").sum())
    med_count = int((results_df["Risk_Category"] == "Medium Risk").sum())
    low_count = int((results_df["Risk_Category"] == "Low Risk").sum())
    healthy_pct = round((low_count / total) * 100, 1)

    summary = {
        "total_machines": total,
        "high_risk_count": high_count,
        "medium_risk_count": med_count,
        "low_risk_count": low_count,
        "healthy_percentage": healthy_pct,
        "threshold_used": threshold,
        "inspection_needed": high_count + med_count
    }

    return {
        "success": True,
        "errors": errors,
        "warnings": warnings,
        "results_df": results_df,
        "summary": summary
    }


# Presets for instant interactive testing in UI
PRESETS = {
    "Healthy Nominal Machine": {
        "air_temp": 298.2,
        "proc_temp": 308.7,
        "rpm": 1500,
        "torque": 40.0,
        "tool_wear": 45,
        "product_type": "L",
        "description": "Standard nominal working conditions. Low risk."
    },
    "Heat Dissipation Failure (HDF)": {
        "air_temp": 302.5,
        "proc_temp": 310.2,
        "rpm": 1320,
        "torque": 52.0,
        "tool_wear": 95,
        "product_type": "L",
        "description": "ΔT = 7.7 K (< 8.6 K) & RPM = 1320 (< 1380). Insufficient heat dissipation causes thermal failure."
    },
    "Power Failure - Overload (PWF)": {
        "air_temp": 299.0,
        "proc_temp": 309.5,
        "rpm": 2650,
        "torque": 38.0,
        "tool_wear": 30,
        "product_type": "M",
        "description": "Power = 10,545 W (> 9,000 W limit). Extreme electrical/spindle drive power overload."
    },
    "Overstrain Failure (OSF)": {
        "air_temp": 298.8,
        "proc_temp": 309.2,
        "rpm": 1420,
        "torque": 68.0,
        "tool_wear": 195,
        "product_type": "L",
        "description": "Strain = Tool Wear (195 min) × Torque (68 Nm) = 13,260 min·Nm (exceeds 11,000 limit for Type L)."
    },
    "Critical Tool Wear Failure (TWF)": {
        "air_temp": 298.5,
        "proc_temp": 308.8,
        "rpm": 1550,
        "torque": 42.0,
        "tool_wear": 235,
        "product_type": "H",
        "description": "Tool wear reaches 235 minutes (> 200 min critical threshold). High risk of cutting edge fracture."
    }
}
