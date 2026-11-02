"""End-to-end verification script for single-machine and batch fleet prediction workflows."""
import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import numpy as np

from src.prediction import predict_single, predict_batch, PRESETS
from config import DATA_DIR, DECISION_THRESHOLD

def main():
    print("=" * 60)
    print("STEP 1: TESTING SINGLE-MACHINE PREDICTION WORKFLOW")
    print("=" * 60)
    
    # 1. Nominal Machine Test
    nom = PRESETS["Healthy Nominal Machine"]
    res_nom = predict_single(
        air_temp=nom["air_temp"],
        proc_temp=nom["proc_temp"],
        rpm=nom["rpm"],
        torque=nom["torque"],
        tool_wear=nom["tool_wear"],
        product_type=nom["product_type"]
    )
    print(f"[TEST 1 - Nominal Machine]")
    print(f"  • Probability: {res_nom['probability']*100:.2f}%")
    print(f"  • Risk Level: {res_nom['risk_level']} (Cutoff: {res_nom['threshold']*100:.1f}%)")
    print(f"  • Status: {res_nom['status']}")
    print(f"  • Physical Triggers: {len(res_nom['triggers'])} active")
    assert res_nom["probability"] < DECISION_THRESHOLD, "Nominal machine exceeded threshold!"
    print("  --> PASS: Correctly classified as LOW RISK / HEALTHY\n")
    
    # 2. Heat Dissipation Failure (HDF) Test
    hdf = PRESETS["Heat Dissipation Failure (HDF)"]
    res_hdf = predict_single(
        air_temp=hdf["air_temp"],
        proc_temp=hdf["proc_temp"],
        rpm=hdf["rpm"],
        torque=hdf["torque"],
        tool_wear=hdf["tool_wear"],
        product_type=hdf["product_type"]
    )
    print(f"[TEST 2 - Heat Dissipation Failure (HDF)]")
    print(f"  • Probability: {res_hdf['probability']*100:.2f}%")
    print(f"  • Risk Level: {res_hdf['risk_level']}")
    print(f"  • Priority: {res_hdf['priority']}")
    print(f"  • Active Triggers: {[t['code'] for t in res_hdf['triggers']]}")
    print(f"  • Recommendation: {res_hdf['recommendations'][0]}")
    assert res_hdf["probability"] >= DECISION_THRESHOLD, "HDF failure not flagged!"
    assert any(t["code"] == "HDF" for t in res_hdf["triggers"]), "HDF trigger not detected!"
    print("  --> PASS: Correctly flagged as HIGH RISK with HDF Root Trigger\n")

    # 3. Overstrain Failure (OSF) Test
    osf = PRESETS["Overstrain Failure (OSF)"]
    res_osf = predict_single(
        air_temp=osf["air_temp"],
        proc_temp=osf["proc_temp"],
        rpm=osf["rpm"],
        torque=osf["torque"],
        tool_wear=osf["tool_wear"],
        product_type=osf["product_type"]
    )
    print(f"[TEST 3 - Overstrain Failure (OSF)]")
    print(f"  • Probability: {res_osf['probability']*100:.2f}%")
    print(f"  • Risk Level: {res_osf['risk_level']}")
    print(f"  • Active Triggers: {[t['code'] for t in res_osf['triggers']]}")
    assert res_osf["probability"] >= DECISION_THRESHOLD, "OSF failure not flagged!"
    assert any(t["code"] == "OSF" for t in res_osf["triggers"]), "OSF trigger not detected!"
    print("  --> PASS: Correctly flagged as HIGH RISK with OSF Root Trigger\n")

    print("=" * 60)
    print("STEP 2: TESTING BATCH CSV UPLOAD & PREDICTION WORKFLOW")
    print("=" * 60)
    sample_csv = DATA_DIR / "sample_fleet.csv"
    assert sample_csv.exists(), f"Sample fleet not found at {sample_csv}"
    
    fleet_df = pd.read_csv(sample_csv)
    print(f"Loaded sample fleet CSV: {len(fleet_df)} machine records")
    
    batch_res = predict_batch(fleet_df)
    assert batch_res["success"] is True, f"Batch prediction failed: {batch_res['errors']}"
    
    out_df = batch_res["results_df"]
    summary = batch_res["summary"]
    
    print(f"[BATCH RESULTS SUMMARY]")
    print(f"  • Total Machines Scanned: {summary['total_machines']}")
    print(f"  • Critical (High Risk):   {summary['high_risk_count']}")
    print(f"  • Advisory (Medium Risk): {summary['medium_risk_count']}")
    print(f"  • Nominal (Low Risk):     {summary['low_risk_count']}")
    print(f"  • Fleet Health Index:     {summary['healthy_percentage']}%")
    
    # Test CSV export simulation
    export_path = DATA_DIR / "test_inspection_export.csv"
    out_df.to_csv(export_path, index=False)
    assert export_path.exists() and export_path.stat().st_size > 0
    print(f"  • Export CSV generated: {export_path.name} ({export_path.stat().st_size} bytes)")
    export_path.unlink()  # Clean up temporary test file
    print("  --> PASS: Batch prediction & CSV download simulation verified\n")
    print("ALL WORKFLOW TESTS COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
