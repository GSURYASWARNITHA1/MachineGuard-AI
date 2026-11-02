"""Evaluation metrics extraction, curve generation, and diagnostic reporting for MachineGuard AI."""
from typing import Dict, Any, Tuple
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score
)

from config import METRICS_PATH, TEST_PROBS_PATH, TEST_LABELS_PATH, COST_FN, COST_FP, DECISION_THRESHOLD


class EvaluationManager:
    """Manages loading of reproducible test results and generates dynamic curve points."""
    _metrics = None
    _probs = None
    _labels = None

    @classmethod
    def get_metrics(cls) -> Dict[str, Any]:
        if cls._metrics is None:
            if not METRICS_PATH.exists():
                raise FileNotFoundError(f"Metrics file not found at {METRICS_PATH}")
            cls._metrics = json.loads(METRICS_PATH.read_text())
        return cls._metrics

    @classmethod
    def get_test_arrays(cls) -> Tuple[np.ndarray, np.ndarray]:
        if cls._probs is None or cls._labels is None:
            if not TEST_PROBS_PATH.exists() or not TEST_LABELS_PATH.exists():
                raise FileNotFoundError("Test arrays (test_probs.npy, test_labels.npy) not found in results directory.")
            cls._probs = np.load(TEST_PROBS_PATH)
            cls._labels = np.load(TEST_LABELS_PATH)
        return cls._probs, cls._labels

    @classmethod
    def get_roc_data(cls) -> Dict[str, Any]:
        probs, labels = cls.get_test_arrays()
        fpr, tpr, thresholds = roc_curve(labels, probs)
        auc = roc_auc_score(labels, probs)
        return {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": thresholds.tolist(),
            "auc": round(float(auc), 4)
        }

    @classmethod
    def get_pr_data(cls) -> Dict[str, Any]:
        probs, labels = cls.get_test_arrays()
        precision, recall, thresholds = precision_recall_curve(labels, probs)
        auc = average_precision_score(labels, probs)
        baseline = float(labels.mean())
        return {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "thresholds": thresholds.tolist(),
            "auc": round(float(auc), 4),
            "baseline": round(baseline, 4)
        }

    @classmethod
    def evaluate_at_threshold(cls, threshold: float = DECISION_THRESHOLD) -> Dict[str, Any]:
        probs, labels = cls.get_test_arrays()
        preds = (probs >= threshold).astype(int)
        
        tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
        cost = COST_FN * fn + COST_FP * fp
        
        rec = recall_score(labels, preds, zero_division=0)
        prec = precision_score(labels, preds, zero_division=0)
        f1 = f1_score(labels, preds, zero_division=0)
        acc = float((preds == labels).mean())
        
        return {
            "threshold": threshold,
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
            "recall": round(float(rec), 4),
            "precision": round(float(prec), 4),
            "f1": round(float(f1), 4),
            "accuracy": round(acc, 4),
            "cost": float(cost),
            "fpr": round(float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0, 4)
        }

    @classmethod
    def get_model_comparison_table(cls) -> pd.DataFrame:
        m = cls.get_metrics()
        df = pd.DataFrame(m["cv_comparison"])
        df["PR-AUC"] = df.apply(lambda r: f"{r['pr_auc']:.4f} ± {r['pr_auc_std']:.4f}", axis=1)
        df["ROC-AUC"] = df.apply(lambda r: f"{r['roc_auc']:.4f} ± {r['roc_auc_std']:.4f}", axis=1)
        df["Recall"] = df.apply(lambda r: f"{r['recall']:.4f} ± {r['recall_std']:.4f}", axis=1)
        df["Precision"] = df.apply(lambda r: f"{r['precision']:.4f} ± {r['precision_std']:.4f}", axis=1)
        return df[["model", "features", "PR-AUC", "ROC-AUC", "Recall", "Precision"]]

    @classmethod
    def get_noise_robustness_table(cls) -> pd.DataFrame:
        m = cls.get_metrics()
        df = pd.DataFrame(m.get("noise_robustness", []))
        if not df.empty:
            df["Noise %"] = (df["noise_frac_of_std"] * 100).astype(int).astype(str) + "% of Feature Std"
            df["PR-AUC"] = df["pr_auc"]
            df["Recall"] = df["recall"]
            return df[["Noise %", "PR-AUC", "Recall"]]
        return pd.DataFrame()

    @classmethod
    def get_lotto_table(cls) -> pd.DataFrame:
        m = cls.get_metrics()
        df = pd.DataFrame(m.get("leave_one_type_out", []))
        if not df.empty:
            name_map = {"L": "Type L (Low Quality - 6,000 machines)",
                        "M": "Type M (Medium Quality - 2,997 machines)",
                        "H": "Type H (High Quality - 1,003 machines)"}
            df["Held-Out Variant"] = df["held_out_type"].map(name_map)
            df["Total Test Samples"] = df["n"]
            df["Failures in Test"] = df["failures"]
            df["PR-AUC"] = df["pr_auc"]
            df["Recall @ 0.5"] = df["recall_at_0.5"]
            df["Precision @ 0.5"] = df["precision_at_0.5"]
            return df[["Held-Out Variant", "Total Test Samples", "Failures in Test", "PR-AUC", "Recall @ 0.5", "Precision @ 0.5"]]
        return pd.DataFrame()

    @classmethod
    def get_failure_mode_breakdown(cls) -> pd.DataFrame:
        m = cls.get_metrics()
        breakdown = m.get("per_mode_recall", {})
        rows = []
        name_map = {
            "HDF": "Heat Dissipation Failure (HDF)",
            "PWF": "Power Failure (PWF)",
            "OSF": "Overstrain Failure (OSF)",
            "TWF": "Tool Wear Failure (TWF)"
        }
        for code, info in breakdown.items():
            rows.append({
                "Failure Mode": name_map.get(code, code),
                "Total in Test Set": info["failures"],
                "Correctly Caught": info["caught"],
                "Recall Rate": f"{info['recall'] * 100:.1f}%",
                "Recall_Num": info["recall"]
            })
        return pd.DataFrame(rows)
