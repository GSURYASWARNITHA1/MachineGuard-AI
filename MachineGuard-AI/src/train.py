"""Train and rigorously evaluate the failure model.

Run:  python -m src.train
Writes results/metrics.json and models/model.joblib.

Protocol (every choice is made WITHOUT looking at the test set):
  1. Stratified 80/20 split, done before any fitting.
  2. Repeated stratified 5-fold CV on the training set compares 3 model families
     on raw vs engineered features (metric: PR-AUC, because failures are 3.4%).
  3. The best CV configuration is calibrated and a cost-based decision threshold
     is chosen from out-of-fold training predictions.
  4. The held-out test set is scored once, with bootstrap confidence intervals.
  5. Extra checks: sensor-noise robustness, leave-one-product-type-out, per-mode recall.
"""
import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold,
                                     cross_val_predict, cross_validate, train_test_split)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer

from src.data import (MODE_COLS, RAW_FEATURES, build_frame, failure_modes, load_raw,
                      select_engineered, select_raw, verify)

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
SEED = 42
COST_FN, COST_FP = 10.0, 1.0  # a missed failure is assumed 10x worse than a false alarm


def make_model(name: str, engineered: bool):
    prep = FunctionTransformer(select_engineered if engineered else select_raw)
    if name == "LogisticRegression":
        est = LogisticRegression(class_weight="balanced", max_iter=2000)
        return make_pipeline(prep, StandardScaler(), est)
    if name == "RandomForest":
        est = RandomForestClassifier(n_estimators=200, min_samples_leaf=2, class_weight="balanced_subsample",
                                     n_jobs=2, random_state=SEED)
    else:
        est = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=200,
                                             class_weight="balanced", random_state=SEED)
    return make_pipeline(prep, est)


def ci(values, lo=2.5, hi=97.5):
    return [round(float(np.percentile(values, lo)), 4), round(float(np.percentile(values, hi)), 4)]


def bootstrap(y, p, thr, n=1000):
    rng = np.random.default_rng(SEED)
    rows = {"recall": [], "precision": [], "pr_auc": [], "roc_auc": []}
    y = np.asarray(y); p = np.asarray(p)
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].sum() == 0:
            continue
        pred = (p[i] >= thr).astype(int)
        rows["recall"].append(recall_score(y[i], pred, zero_division=0))
        rows["precision"].append(precision_score(y[i], pred, zero_division=0))
        rows["pr_auc"].append(average_precision_score(y[i], p[i]))
        rows["roc_auc"].append(roc_auc_score(y[i], p[i]))
    return {k: ci(v) for k, v in rows.items()}


def cost(y, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return COST_FN * fn + COST_FP * fp, int(tn), int(fp), int(fn), int(tp)


def point_metrics(y, p, thr):
    pred = (p >= thr).astype(int)
    c, tn, fp, fn, tp = cost(y, pred)
    return {"threshold": round(float(thr), 4), "recall": round(recall_score(y, pred), 4),
            "precision": round(precision_score(y, pred, zero_division=0), 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn, "cost": c}


def main():
    t0 = time.time()
    df = load_raw()
    check = verify(df)
    assert all(ok for _, _, ok in check.values()), f"dataset does not match UCI figures: {check}"
    X, y, types = build_frame(df)
    modes = failure_modes(df)
    out = {"dataset_check": {k: {"got": g, "uci": e, "match": ok} for k, (g, e, ok) in check.items()},
           "seed": SEED, "cost_fn": COST_FN, "cost_fp": COST_FP}

    # 1. split first
    idx = np.arange(len(X))
    tr, te = train_test_split(idx, test_size=0.2, stratify=y, random_state=SEED)
    Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]
    out["split"] = {"train_rows": len(tr), "test_rows": len(te), "train_failures": int(ytr.sum()),
                    "test_failures": int(yte.sum())}
    print("split", out["split"], flush=True)

    # 2. repeated CV model comparison on TRAIN only
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
    scoring = {"pr_auc": "average_precision", "roc_auc": "roc_auc", "recall": "recall", "precision": "precision"}
    rows, best = [], None
    for name in ["LogisticRegression", "RandomForest", "HistGradientBoosting"]:
        for eng in (False, True):
            r = cross_validate(make_model(name, eng), Xtr, ytr, cv=cv, scoring=scoring, n_jobs=1)
            row = {"model": name, "features": "engineered" if eng else "raw"}
            for m in scoring:
                v = r[f"test_{m}"]
                row[m] = round(float(v.mean()), 4)
                row[m + "_std"] = round(float(v.std()), 4)
            rows.append(row)
            print(row, f"{time.time()-t0:.0f}s", flush=True)
            if best is None or row["pr_auc"] > best[0]["pr_auc"]:
                best = (row, name, eng)
    out["cv_comparison"] = rows
    out["selected"] = {"model": best[1], "features": best[0]["features"], "selected_by": "mean CV PR-AUC on training set"}
    sel_name, sel_eng = best[1], best[2]

    # 3. calibrate + choose cost threshold from OUT-OF-FOLD train predictions
    def calibrated():
        return CalibratedClassifierCV(make_model(sel_name, sel_eng), method="sigmoid", cv=5)

    oof = cross_val_predict(calibrated(), Xtr, ytr, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                            method="predict_proba")[:, 1]
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
    costs = [cost(ytr, (oof >= t).astype(int))[0] for t in grid]
    thr = float(grid[int(np.argmin(costs))])
    out["threshold_selection"] = {"chosen": thr, "oof_cost_at_chosen": float(min(costs)),
                                  "oof_cost_at_0.5": float(cost(ytr, (oof >= 0.5).astype(int))[0])}

    # 4. final fit + single test evaluation
    raw_model = make_model(sel_name, sel_eng).fit(Xtr, ytr)
    cal_model = calibrated().fit(Xtr, ytr)
    p_raw = raw_model.predict_proba(Xte)[:, 1]
    p_cal = cal_model.predict_proba(Xte)[:, 1]
    out["test"] = {
        "uncalibrated_at_0.5": point_metrics(yte, p_raw, 0.5),
        "calibrated_at_0.5": point_metrics(yte, p_cal, 0.5),
        "calibrated_at_cost_threshold": point_metrics(yte, p_cal, thr),
        "pr_auc": round(average_precision_score(yte, p_cal), 4),
        "roc_auc": round(roc_auc_score(yte, p_cal), 4),
        "brier_uncalibrated": round(brier_score_loss(yte, p_raw), 5),
        "brier_calibrated": round(brier_score_loss(yte, p_cal), 5),
        "bootstrap_95ci_at_cost_threshold": bootstrap(yte, p_cal, thr),
        "baseline_always_no_failure": {"accuracy": round(1 - float(yte.mean()), 4), "recall": 0.0},
    }
    for k in ("uncalibrated_at_0.5", "calibrated_at_0.5", "calibrated_at_cost_threshold"):
        print("TEST", k, out["test"][k], flush=True)

    # per-failure-mode recall (test rows only, at chosen threshold)
    pred = (p_cal >= thr).astype(int)
    mt = modes.iloc[te]
    # Only rows that actually failed count (the dataset has RNF flags on rows with no recorded failure).
    failed = yte.values == 1
    out["per_mode_recall"] = {}
    for m in MODE_COLS:
        sel = failed & (mt[m].values == 1)
        if sel.sum() > 0:
            out["per_mode_recall"][m] = {"failures": int(sel.sum()), "caught": int((pred[sel] == 1).sum()),
                                         "recall": round(float((pred[sel] == 1).mean()), 3)}
    unlabeled = (yte.values == 1) & (mt.max(axis=1).values == 0)
    out["unlabeled_failures_in_test"] = {"n": int(unlabeled.sum()), "caught": int((pred[unlabeled] == 1).sum())}

    # 5a. sensor-noise robustness: Gaussian noise = fraction of each feature's train std, test set only
    num = ["air_temp", "proc_temp", "rpm", "torque", "tool_wear"]
    sd = Xtr[num].std()
    rng = np.random.default_rng(SEED)
    noise = []
    for frac in [0.0, 0.02, 0.05, 0.10, 0.20]:
        rs = []
        for _ in range(10 if frac else 1):
            Xn = Xte.copy()
            for c in num:
                Xn[c] = Xn[c] + rng.normal(0, frac * sd[c], len(Xn))
            pn = cal_model.predict_proba(Xn)[:, 1]
            rs.append((average_precision_score(yte, pn), recall_score(yte, (pn >= thr).astype(int))))
        a = np.array(rs)
        noise.append({"noise_frac_of_std": frac, "pr_auc": round(float(a[:, 0].mean()), 4),
                      "recall": round(float(a[:, 1].mean()), 4)})
    out["noise_robustness"] = noise
    print("noise", noise, flush=True)

    # 5b. leave-one-product-type-out (train on two types, test on the unseen one)
    lotto = []
    for held in ["L", "M", "H"]:
        trm, tem = (types != held).values, (types == held).values
        m = make_model(sel_name, sel_eng).fit(X[trm], y[trm])
        pp = m.predict_proba(X[tem])[:, 1]
        pr = (pp >= 0.5).astype(int)
        lotto.append({"held_out_type": held, "n": int(tem.sum()), "failures": int(y[tem].sum()),
                      "pr_auc": round(average_precision_score(y[tem], pp), 4),
                      "recall_at_0.5": round(recall_score(y[tem], pr), 4),
                      "precision_at_0.5": round(precision_score(y[tem], pr, zero_division=0), 4)})
    out["leave_one_type_out"] = lotto
    print("lotto", lotto, flush=True)

    # feature importance via permutation on the test set (no extra dependency)
    from sklearn.inspection import permutation_importance
    pi = permutation_importance(cal_model, Xte, yte, scoring="average_precision", n_repeats=10, random_state=SEED)
    imp = sorted(zip(RAW_FEATURES, pi.importances_mean, pi.importances_std), key=lambda t: -t[2 - 1])
    out["permutation_importance_pr_auc_drop"] = [{"feature": f, "drop": round(float(m), 4), "std": round(float(s), 4)}
                                                  for f, m, s in imp]

    # save
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump({"model": cal_model, "threshold": thr, "features": RAW_FEATURES, "engineered": sel_eng,
                 "name": sel_name, "cost_fn": COST_FN, "cost_fp": COST_FP}, ROOT / "models" / "model.joblib")
    np.save(ROOT / "results" / "test_probs.npy", p_cal)
    np.save(ROOT / "results" / "test_labels.npy", yte.values)
    (ROOT / "results" / "metrics.json").write_text(json.dumps(out, indent=2))
    print(f"done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
