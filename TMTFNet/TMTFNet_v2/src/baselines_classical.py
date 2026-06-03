"""Classical HAR baselines requested by Reviewer 2.

Trains SVM / RF / kNN / LR on the official UCI-HAR 561 hand-crafted features
(X_train.txt, X_test.txt). This is the canonical classical-ML protocol for
this dataset -- Anguita et al. (2013) distribute those pre-extracted features
specifically so non-deep methods can be compared head-to-head.

Exports ``run_classical_baselines(data_dir, seeds, val_ratio)`` which returns
a nested dict matching the per-seed JSON layout of ``final_exp1_har``.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def _load_har_features(data_dir):
    base = data_dir
    if not os.path.isdir(os.path.join(base, "train")):
        nested = os.path.join(data_dir, "UCI HAR Dataset")
        if os.path.isdir(os.path.join(nested, "train")):
            base = nested
        else:
            raise FileNotFoundError(f"Cannot locate UCI HAR files under {data_dir}")
    X_train = pd.read_csv(os.path.join(base, "train", "X_train.txt"), sep=r"\s+", header=None).values.astype(np.float32)
    y_train = pd.read_csv(os.path.join(base, "train", "y_train.txt"), header=None).values.flatten().astype(np.int64) - 1
    X_test = pd.read_csv(os.path.join(base, "test", "X_test.txt"), sep=r"\s+", header=None).values.astype(np.float32)
    y_test = pd.read_csv(os.path.join(base, "test", "y_test.txt"), header=None).values.flatten().astype(np.int64) - 1
    return X_train, y_train, X_test, y_test


def _eval(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def _build_pipeline(name, seed):
    if name == "SVM":
        est = SVC(kernel="rbf", C=8.0, gamma="scale", random_state=seed)
    elif name == "RF":
        est = RandomForestClassifier(n_estimators=400, n_jobs=-1, random_state=seed)
    elif name == "kNN":
        est = KNeighborsClassifier(n_neighbors=7, weights="distance", n_jobs=-1)
    elif name == "LR":
        est = LogisticRegression(C=1.0, max_iter=5000, n_jobs=-1, random_state=seed)
    else:
        raise ValueError(f"Unknown classical baseline '{name}'")
    return Pipeline([("scaler", StandardScaler()), ("clf", est)])


def run_classical_baselines(data_dir, seeds=(42, 123, 456, 789, 2024), models=("SVM", "RF", "kNN", "LR")):
    X_train, y_train, X_test, y_test = _load_har_features(data_dir)
    results = {}
    for model_name in models:
        runs = []
        for seed in seeds:
            start = time.time()
            pipe = _build_pipeline(model_name, int(seed))
            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_test)
            metrics = _eval(y_test, preds)
            metrics.update(
                {
                    "training_time": float(time.time() - start),
                    "seed": int(seed),
                    "model_name": model_name,
                    "n_params": int(_count_params(pipe)),
                }
            )
            runs.append(metrics)
        results[model_name] = {
            "runs": runs,
            "accuracy": _agg(runs, "accuracy"),
            "f1_macro": _agg(runs, "f1_macro"),
            "f1_weighted": _agg(runs, "f1_weighted"),
            "precision": _agg(runs, "precision"),
            "recall": _agg(runs, "recall"),
            "training_time": _agg(runs, "training_time"),
        }
    return results


def _agg(runs, key):
    xs = np.asarray([run[key] for run in runs], dtype=np.float64)
    return {"mean": float(xs.mean()), "std": float(xs.std(ddof=1)) if xs.size > 1 else 0.0}


def _count_params(pipeline):
    clf = pipeline.named_steps["clf"]
    if isinstance(clf, (LogisticRegression,)):
        return clf.coef_.size + clf.intercept_.size
    if isinstance(clf, RandomForestClassifier):
        return sum(tree.tree_.node_count * 4 for tree in clf.estimators_)
    if isinstance(clf, KNeighborsClassifier):
        return int(clf._fit_X.size) if hasattr(clf, "_fit_X") else 0
    if isinstance(clf, SVC):
        return int(clf.support_vectors_.size + clf.dual_coef_.size + clf.intercept_.size)
    return 0


__all__ = ["run_classical_baselines"]
