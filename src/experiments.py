"""Every experiment reported in the paper. Results are written to results/."""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_recall_fscore_support,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, train_test_split

import models
import unsw_loader
from config import FLOWS, RESULTS, SEED

ETA_EXCLUDE = {
    "flow_id", "scenario", "label", "ai_bound", "obfuscation", "evasion",
    "n_requests", "app_uploaded_bytes", "app_downloaded_bytes", "plaintext_chars",
}


def load_ai() -> tuple[pd.DataFrame, np.ndarray]:
    frame = pd.read_csv(FLOWS / "ai_endpoint_flows.csv")
    seq = np.load(FLOWS / "ai_endpoint_sequences.npy")
    return frame, seq


def eta_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns
            if c not in ETA_EXCLUDE and not c.startswith("dlp_")]


def _metrics(y_true, y_pred, scores=None, positive: int | None = None) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, zero_division=0, labels=sorted(set(y_true)))
    out = {
        "accuracy": float((np.asarray(y_true) == np.asarray(y_pred)).mean()),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class": {int(c): {"precision": float(p), "recall": float(r),
                               "f1": float(f), "support": int(s)}
                      for c, p, r, f, s in zip(sorted(set(y_true)), precision,
                                               recall, f1, support)},
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    if scores is not None and positive is not None:
        column = scores[:, positive] if scores.ndim > 1 else scores
        binary = (np.asarray(y_true) == positive).astype(int)
        out["roc_auc"] = float(roc_auc_score(binary, column))
        out["pr_auc"] = float(average_precision_score(binary, column))
        tn, fp, fn, tp = confusion_matrix(binary, (np.asarray(y_pred) == positive).astype(int)).ravel()
        out["false_positive_rate"] = float(fp / max(1, fp + tn))
        out["false_negative_rate"] = float(fn / max(1, fn + tp))
    return out


# ---------------------------------------------------------------- task A
def task_a(ai: pd.DataFrame, per_class: int = 10000) -> dict:
    raw = unsw_loader.load_unsw()
    common = unsw_loader.to_common(raw)
    common["label"] = np.where(raw["label"].astype(int) == 0, 0, 1)
    common["attack_cat"] = raw["attack_cat"].astype(str).values
    rng = np.random.RandomState(SEED)
    normal = common[common["label"] == 0].sample(
        min(per_class, int((common["label"] == 0).sum())), random_state=SEED)
    attack = common[common["label"] == 1].sample(
        min(per_class, int((common["label"] == 1).sum())), random_state=SEED)

    ai_common = unsw_loader.ai_to_common(ai)
    ai_common["label"] = np.where(ai["label"].values == 2, 2, 0)
    ai_common["attack_cat"] = ai["scenario"].values

    table = pd.concat([normal, attack, ai_common], ignore_index=True)
    table = table.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X = table[unsw_loader.COMMON].values
    y = table["label"].values.astype(int)

    x_tr, x_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=SEED)
    report = {"composition": {int(k): int(v) for k, v in
                              pd.Series(y).value_counts().sort_index().items()},
              "models": {}}
    for name, model in models.tabular_models(3).items():
        model.fit(x_tr, y_tr)
        pred = model.predict(x_te)
        proba = model.predict_proba(x_te)
        report["models"][name] = _metrics(y_te, pred, proba, positive=2)
    return report


# ---------------------------------------------------------------- task B
def task_b(ai: pd.DataFrame, seq: np.ndarray) -> dict:
    cols = eta_columns(ai)
    X = ai[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
    y = (ai["label"].values == 2).astype(int)

    report = {"n_flows": int(len(ai)), "n_features": len(cols), "features": cols,
              "class_counts": {int(k): int(v) for k, v in
                               pd.Series(y).value_counts().sort_index().items()},
              "cross_validation": {}, "holdout": {}, "roc": {}, "pr": {}}

    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for name in models.tabular_models(2):
        scores = {"macro_f1": [], "roc_auc": [], "pr_auc": [], "recall": [],
                  "precision": [], "fpr": []}
        for train_idx, test_idx in folds.split(X, y):
            model = models.tabular_models(2)[name]
            model.fit(X[train_idx], y[train_idx])
            pred = model.predict(X[test_idx])
            proba = model.predict_proba(X[test_idx])[:, 1]
            m = _metrics(y[test_idx], pred, proba.reshape(-1, 1), positive=None)
            scores["macro_f1"].append(m["macro_f1"])
            scores["roc_auc"].append(float(roc_auc_score(y[test_idx], proba)))
            scores["pr_auc"].append(float(average_precision_score(y[test_idx], proba)))
            p, r, f, _ = precision_recall_fscore_support(
                y[test_idx], pred, average="binary", zero_division=0)
            scores["precision"].append(float(p))
            scores["recall"].append(float(r))
            tn, fp, fn, tp = confusion_matrix(y[test_idx], pred).ravel()
            scores["fpr"].append(float(fp / max(1, fp + tn)))
        report["cross_validation"][name] = {
            k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
            for k, v in scores.items()}

    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.3, stratify=y, random_state=SEED)
    for name in models.tabular_models(2):
        model = models.tabular_models(2)[name]
        model.fit(X[tr], y[tr])
        proba = model.predict_proba(X[te])[:, 1]
        pred = (proba >= 0.5).astype(int)
        report["holdout"][name] = _metrics(y[te], pred, proba.reshape(-1, 1))
        report["holdout"][name]["roc_auc"] = float(roc_auc_score(y[te], proba))
        report["holdout"][name]["pr_auc"] = float(average_precision_score(y[te], proba))
        tn, fp, fn, tp = confusion_matrix(y[te], pred).ravel()
        report["holdout"][name]["false_positive_rate"] = float(fp / max(1, fp + tn))
        fpr, tpr, _ = roc_curve(y[te], proba)
        pr, rc, _ = precision_recall_curve(y[te], proba)
        report["roc"][name] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
        report["pr"][name] = {"precision": pr.tolist(), "recall": rc.tolist()}

    # Sequence model on the same split.
    x_tr, x_te = seq[tr], seq[te]
    inner_tr, inner_val = train_test_split(
        np.arange(len(tr)), test_size=0.2, stratify=y[tr], random_state=SEED)
    net, history = models.train_bilstm(
        x_tr[inner_tr], y[tr][inner_tr], x_tr[inner_val], y[tr][inner_val])
    proba = models.bilstm_scores(net, x_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    report["holdout"]["Bidirectional LSTM"] = _metrics(y[te], pred)
    report["holdout"]["Bidirectional LSTM"]["roc_auc"] = float(roc_auc_score(y[te], proba))
    report["holdout"]["Bidirectional LSTM"]["pr_auc"] = float(average_precision_score(y[te], proba))
    tn, fp, fn, tp = confusion_matrix(y[te], pred).ravel()
    report["holdout"]["Bidirectional LSTM"]["false_positive_rate"] = float(fp / max(1, fp + tn))
    fpr, tpr, _ = roc_curve(y[te], proba)
    pr, rc, _ = precision_recall_curve(y[te], proba)
    report["roc"]["Bidirectional LSTM"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
    report["pr"]["Bidirectional LSTM"] = {"precision": pr.tolist(), "recall": rc.tolist()}
    report["bilstm_history"] = history
    report["split"] = {"train": tr.tolist(), "test": te.tolist()}

    # Permutation importance of the strongest tabular model.
    best = models.tabular_models(2)["Random Forest"]
    best.fit(X[tr], y[tr])
    imp = permutation_importance(best, X[te], y[te], n_repeats=12,
                                 random_state=SEED, scoring="f1", n_jobs=1)
    report["permutation_importance"] = sorted(
        [{"feature": c, "mean": float(m), "std": float(s)}
         for c, m, s in zip(cols, imp.importances_mean, imp.importances_std)],
        key=lambda d: -d["mean"])

    # Detection latency of the deployed model on single flows.
    timings = []
    for row in X[te][:400]:
        start = time.perf_counter()
        best.predict(row.reshape(1, -1))
        timings.append((time.perf_counter() - start) * 1000.0)
    report["inference_latency_ms"] = {
        "mean": float(np.mean(timings)), "p50": float(np.percentile(timings, 50)),
        "p95": float(np.percentile(timings, 95)), "p99": float(np.percentile(timings, 99))}
    return report


# ---------------------------------------------------------------- task C
def task_c_dlp(ai: pd.DataFrame, report_b: dict) -> dict:
    y = (ai["label"].values == 2).astype(int)
    te = np.array(report_b["split"]["test"])
    dlp = ai["dlp_dlp_alert"].values.astype(int)
    out = {"whole_set": _metrics(y, dlp), "test_split": _metrics(y[te], dlp[te])}
    out["rule_activation"] = {
        c.replace("dlp_", ""): int((ai[c] > 0).sum())
        for c in ai.columns if c.startswith("dlp_") and c != "dlp_dlp_alert"}
    out["alerts_on_benign"] = int(dlp[y == 0].sum())
    out["alerts_on_exfil"] = int(dlp[y == 1].sum())
    return out


# ---------------------------------------------------------------- task D
def task_d_generalisation(ai: pd.DataFrame) -> dict:
    cols = eta_columns(ai)
    X = ai[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
    y = (ai["label"].values == 2).astype(int)
    scenarios = ai["scenario"].values
    out = {}
    for held in ["exfil_code_chunked", "exfil_doc_bulk", "agentic_scrape"]:
        train_mask = ~((scenarios == held) & (y == 1))
        test_mask = (scenarios == held) & (y == 1)
        benign_idx = np.where(y == 0)[0]
        rng = np.random.RandomState(SEED)
        reserve = min(300, max(1, benign_idx.size // 3))
        benign_test = rng.choice(benign_idx, size=reserve, replace=False)
        train_mask[benign_test] = False
        if np.unique(y[train_mask]).size < 2 or test_mask.sum() == 0:
            continue
        model = models.tabular_models(2)["Random Forest"]
        model.fit(X[train_mask], y[train_mask])
        idx = np.concatenate([np.where(test_mask)[0], benign_test])
        proba = model.predict_proba(X[idx])[:, 1]
        pred = (proba >= 0.5).astype(int)
        out[held] = _metrics(y[idx], pred)
        out[held]["roc_auc"] = float(roc_auc_score(y[idx], proba))
        out[held]["recall_on_held_out"] = float(pred[:int(test_mask.sum())].mean())
    return out


# ---------------------------------------------------------------- task E
def task_e_evasion(ai: pd.DataFrame) -> dict:
    cols = eta_columns(ai)
    X = ai[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
    y = (ai["label"].values == 2).astype(int)
    obf = ai["obfuscation"].values
    eva = ai["evasion"].values

    clean = (obf == "none") & (eva == "none")
    rng = np.random.RandomState(SEED)
    benign_idx = np.where(y == 0)[0]
    reserve = min(320, max(1, benign_idx.size // 3))
    benign_test = rng.choice(benign_idx, size=reserve, replace=False)
    # A quarter of the non evasive exfiltration sessions is held back so that
    # the evasive families can be compared against a like for like reference.
    clean_exfil_idx = np.where(clean & (y == 1))[0]
    clean_holdout = rng.choice(clean_exfil_idx,
                               size=max(1, clean_exfil_idx.size // 4), replace=False)
    train_mask = clean.copy()
    train_mask[benign_test] = False
    train_mask[clean_holdout] = False
    model = models.tabular_models(2)["Random Forest"]
    model.fit(X[train_mask], y[train_mask])

    out = {"train_size": int(train_mask.sum())}
    for label, mask in [
        ("clean_exfil", None),
        ("obfuscated_payload", (obf != "none") & (eva == "none") & (y == 1)),
        ("timing_jitter", (eva == "jitter") & (y == 1)),
        ("chunk_morphing", (eva == "morph") & (y == 1)),
    ]:
        sel = clean_holdout if label == "clean_exfil" else np.where(mask)[0]
        if sel.size == 0:
            continue
        idx = np.concatenate([sel, benign_test])
        proba = model.predict_proba(X[idx])[:, 1]
        pred = (proba >= 0.5).astype(int)
        out[label] = {
            "n_exfil": int(sel.size),
            "recall": float(pred[:sel.size].mean()),
            "roc_auc": float(roc_auc_score(y[idx], proba)),
            "mean_score": float(proba[:sel.size].mean()),
        }
    out["benign_reference_fpr"] = float(
        (model.predict_proba(X[benign_test])[:, 1] >= 0.5).mean())
    return out


def scenario_profile(ai: pd.DataFrame) -> pd.DataFrame:
    keep = ["dur", "sbytes", "dbytes", "upload_share", "srecords", "drecords",
            "srec_mean", "sinpkt_mean", "macro_gap_cv", "up_burst_count",
            "max_uplink_run", "byte_asymmetry"]
    grouped = ai.groupby("scenario")[keep].median().round(4)
    grouped["n"] = ai.groupby("scenario").size()
    return grouped.reset_index()
