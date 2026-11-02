# Neural Storm 2026 — MachineGuard AI Presentation Guide

This guide equips you to deliver a winning hackathon presentation, pitch, and live demonstration for **MachineGuard AI**.

---

## 🏆 Judging Criteria Alignment

| Criteria | Weight | How MachineGuard AI Wins |
| :--- | :---: | :--- |
| **Technical Implementation** | **35%** | • Strict target leakage prevention (purged failure mode flags).<br>• Zero test contamination (80/20 split; tuning solely on CV train).<br>• Repeated 5-fold CV across 3 model families (15 folds).<br>• Platt sigmoid calibration (Brier score 0.00625).<br>• Asymmetric cost optimization (Cost_FN=10, Cost_FP=1).<br>• Physics-informed feature engineering (+19.7% PR-AUC gain).<br>• 23 automated pytest tests passing. |
| **Impact & Usefulness** | **25%** | • Addresses $50,000/hour industrial downtime problem.<br>• Moves from reactive run-to-failure to proactive maintenance.<br>• Generates clear technician work orders and physical trigger diagnostics.<br>• Supports both single machine live telemetry and fleet-wide batch CSV audits. |
| **README & Documentation** | **20%** | • Professional README with Mermaid architecture and sequence diagrams.<br>• Transparent disclosure of synthetic benchmark dataset (UCI AI4I 2020).<br>• Complete tabular evaluation records with 95% bootstrap confidence intervals.<br>• Clean folder structure, clean docstrings, clear setup commands. |
| **Demo & Presentation** | **20%** | • Cohesive industrial dark UI design with cyber-teal highlights.<br>• 1-click failure presets demonstrating HDF, PWF, OSF, TWF.<br>• Interactive dynamic threshold slider updating curves in real-time.<br>• Pre-loaded 25-machine fleet for instant zero-friction demo. |

---

## ⏱️ 3-Minute Live Demo Pitch Script

### [0:00 – 0:35] The Hook & The Problem
> *"Judges, in manufacturing plants and CNC milling facilities, an unexpected machine failure is catastrophic. Unplanned downtime frequently costs industrial plants tens of thousands of dollars per hour according to manufacturing industry surveys. Today, factories are often caught between running machines until they break or replacing expensive cutting tools prematurely based on arbitrary calendar schedules.*
> 
> *We built **MachineGuard AI** — an industrial predictive maintenance system that anticipates machine breakdowns from sensor telemetry, diagnoses the exact physical failure mechanism, and optimizes maintenance schedules based on asymmetric error costs where a missed breakdown is penalized 10× more heavily than an inspection false alarm."*

### [0:35 – 1:30] Live Demonstration: Single Machine Diagnostic & Explainability
> *(Navigate to Single Machine Diagnostic)*  
> *"Let's simulate a machine on the shop floor. I'll load the **Heat Dissipation Failure (HDF)** preset.*
> 
> *(Click preset button)*  
> *Notice our risk gauge immediately shoots to **98% Critical Risk**. But MachineGuard AI doesn't stop at giving a black-box probability. Look at the **Root Physical Triggers** section below:  
> It tells the maintenance engineer that the temperature difference $\Delta T$ is only $7.7\text{ K}$ while spindle speed is below $1,380\text{ rpm}$. The system immediately issues a specific action item: 'Clear spindle cooling vents and inspect heat exchanger coolant flow.'  
> 
> If we switch to **Nominal Machine**, the telemetry radar shows our readings safely nested inside the green certified envelope."*

### [1:30 – 2:15] Batch Fleet Inspection
> *(Navigate to Batch Fleet Inspection)*  
> *"In a real factory, engineers supervise hundreds of machines. With our **Batch Fleet Inspection**, operators can drag and drop raw CSV telemetry.*
> 
> *Our engine automatically cleans and aligns column aliases, handles missing values, and flags abnormal machines across the entire fleet in milliseconds. In this 25-machine test fleet, our Fleet Health Index is **64%**, isolating the 7 critical units that require immediate technician attention before the next production shift. Operators can filter and download the prioritized work order with one click."*

### [2:15 – 3:00] ML Rigor, Cost Matrix, & Closing
> *(Navigate to Model Analytics & Validation)*  
> *"Under the hood, we enforced strict machine learning discipline:*
> 1. *Target leakage guardrails: we stripped all failure-mode columns (`TWF`, `HDF`, etc.) prior to modeling.*
> 2. *Because failures represent only 3.4% of records, accuracy is deceptive. We optimized for **PR-AUC**, reaching **0.9155** on held-out test data.*
> 3. *Most importantly, we introduced an **asymmetric cost penalty matrix**: in maintenance operations, a missed breakdown is far more disruptive than a routine check. We assigned a 10:1 penalty ratio ($Cost_{FN}=10, Cost_{FP}=1$). By tuning our calibrated threshold from 0.50 down to **0.34**, we caught **85.3%** of failures on the test set and reduced total penalty cost by **7.2%**.*
> 
> *With full automated test coverage and physics-informed explainability, MachineGuard AI bridges industrial IoT and machine learning. Thank you!"*

---

## 🎯 Anticipated Judge Questions & Bulletproof Answers

### Q1: "The AI4I dataset is synthetic. How does this transfer to real machines?"
> **Answer:**  
> *"That's a critical distinction, and we explicitly disclose this limitation throughout our UI and documentation. The AI4I 2020 dataset (Matzka, 2020) was generated using mathematical models of thermodynamic heat dissipation, drive power, and overstrain. While it provides a rigorous, reproducible benchmark for predictive maintenance algorithms without target leakage, physical deployment would require domain validation against continuous high-frequency vibration streams and real shop-floor wear telemetry."*

### Q2: "Why didn't you just use Deep Learning or an LSTM?"
> **Answer:**  
> *"For tabular sensor telemetry snapshots without sequential time-series timestamps, Tree-based Gradient Boosting consistently outperforms deep neural networks in both tabular benchmarks and sample efficiency. Furthermore, `HistGradientBoosting` executes in sub-millisecond inference time on lightweight industrial edge hardware without requiring power-hungry GPUs on the factory floor."*

### Q3: "What prevents data leakage in your pipeline?"
> **Answer:**  
> *"The AI4I dataset contains columns like `HDF`, `PWF`, `OSF`, `TWF`, which are sub-components of the target `Machine failure`. If left in, a model trivially peeks at the label. Our `src/data.py` pipeline drops these columns, along with `UDI` and `Product ID`, at step zero. Furthermore, our 80/20 train/test split was established prior to any fitting or threshold tuning, verified by our automated test suite."*

### Q4: "Why is the decision threshold set to 0.34 instead of the standard 0.50?"
> **Answer:**  
> *"In industrial operations, false negatives and false positives have unequal operational impact. A false alarm results in a short inspection check, while a missed breakdown can cause line stoppages and tool damage. To capture this tradeoff without inventing arbitrary monetary amounts, we modeled a 10:1 penalty ratio on out-of-fold training predictions. The optimal threshold of 0.34 captures 58 of 68 test failures (85.3% recall) with only 3 false alarms, reducing the normalized cost score from 111.0 to 103.0."*

### Q5: "Why does Tool Wear Failure (TWF) have lower recall (20%) than the other failure modes (100%)?"
> **Answer:**  
> *"Heat dissipation, power overload, and overstrain are instantaneous physical violations (e.g. power exceeding 9,000 W). In contrast, tool wear failure in the AI4I dataset is modeled as a gradual stochastic degradation between 200 and 240 minutes where the cutting edge fractures randomly. Capturing stochastic tool fracture reliably requires time-series vibration spectra (chatter frequency harmonics) rather than a single static tool wear timer."*
