# MachineGuard AI — Application UI Walkthrough & Views

This directory documents the authentic screens rendered by MachineGuard AI at `http://localhost:8501`.

### 1. Home Dashboard (`🏠 Home Dashboard`)
- **Top KPI Cards:**
  - PR-AUC: `0.9155`
  - ROC-AUC: `0.9853`
  - Failure Recall: `85.3%` (58 of 68 test failures caught)
  - Precision: `95.08%`
  - Cost Cutoff: `0.34` (Cost matrix: FN=10, FP=1)
- **Benchmark Disclosure Banner:** Highlights UCI AI4I 2020 synthetic origin and milling machine failure physics.
- **Dataset Health Overview:** 10,000 machines, 9,661 healthy, 339 failures (3.39% prevalence).

### 2. Single Machine Diagnostic (`⚡ Single Machine Diagnostic`)
- **Quick Preset Selector:** 1-click loading for Nominal, HDF, PWF, OSF, and TWF failure conditions.
- **Telemetry Controls:** Kelvin / Celsius toggle, thermodynamic sliders (Air Temp, Process Temp, Product Variant L/M/H), and mechanical sliders (Rotational Speed rpm, Torque Nm, Tool Wear min).
- **Interactive Gauges & Radar:**
  - Semi-circular speedometer gauge with color-coded risk bands.
  - Telemetry radar chart comparing 5 sensor readings against the certified safe operating envelope.
  - Physical Failure Trigger alert boxes with actionable technician work orders.

### 3. Batch Fleet Inspection (`📁 Batch Fleet Inspection`)
- **CSV Ingestion:** Automatic header mapping (`Air temperature [K]`, `spindle_speed`, `torque_nm`, etc.).
- **Fleet KPI Metrics:** Total Machines Scanned, High Risk count, Medium Risk count, Healthy count, and Fleet Health Index.
- **Risk Distribution Donut Chart:** Visual breakdown of fleet health.
- **Interactive Register:** Filterable machine table with suspected failure modes and 1-click CSV export.

### 4. Model Analytics & Validation (`📊 Model Analytics & Validation`)
- **Dynamic Threshold Slider:** Real-time updates to confusion matrix, operating points on ROC and Precision-Recall curves.
- **Confusion Matrix:** Annotated with financial penalty cost ($10:1$ ratio).
- **CV Model Comparison:** Bar chart across Logistic Regression, Random Forest, and HistGradientBoosting (raw vs engineered).
- **Permutation Feature Importance:** Demonstrates Torque and RPM as primary failure drivers.
- **Stress Testing:** Gaussian sensor noise degradation curve (up to 20% noise).

### 5. System Architecture & About (`ℹ️ System Architecture & About`)
- End-to-end data pipeline flowchart.
- Transparency disclosures on synthetic data limitations and roadmap for vibration accelerometer telemetry.
