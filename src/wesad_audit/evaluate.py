"""Three validation protocols on the same feature table.

random_kfold      windows shuffled across folds: the protocol many notebooks use.
                  Overlapping windows from the same subject end up in train and test.
loso              leave-one-subject-out: the model never sees the test person.
loso_subject_norm LOSO after per-subject feature standardization. This assumes the
                  device can collect an unlabelled calibration period per user; it is a
                  deployment assumption, not a free lunch, and is reported as such.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 0


def models() -> dict:
    return {
        "logreg": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                LogisticRegression(max_iter=2000, class_weight="balanced")),
        "random_forest": make_pipeline(SimpleImputer(strategy="median"),
                                       RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                                              random_state=SEED, n_jobs=-1)),
    }


def subject_standardize(X: pd.DataFrame, groups: np.ndarray) -> pd.DataFrame:
    Xn = X.copy()
    for g in np.unique(groups):
        m = groups == g
        mu, sd = X[m].mean(), X[m].std().replace(0, 1.0)
        Xn.loc[m] = (X[m] - mu) / sd
    return Xn


def overlap_leak_rate(meta: pd.DataFrame, train_idx, test_idx, win_s: float) -> float:
    """Share of test windows that overlap in time with a train window of the same subject."""
    tr = meta.iloc[train_idx]
    leaked = 0
    for _, row in meta.iloc[test_idx].iterrows():
        same = tr[tr.subject == row.subject]
        leaked += bool(np.any(np.abs(same.t_start.values - row.t_start) < win_s))
    return leaked / max(len(test_idx), 1)


def _scores(y, p) -> dict:
    return dict(balanced_accuracy=balanced_accuracy_score(y, p), macro_f1=f1_score(y, p, average="macro"))


def run_protocols(X: pd.DataFrame, y: np.ndarray, meta: pd.DataFrame, win_s: float) -> tuple[dict, pd.DataFrame]:
    groups = meta.subject.values
    results, per_subject = {}, []
    for mname, model in models().items():
        # A: random window-level k-fold
        pred = np.empty_like(y)
        leak = []
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y):
            pred[te] = clone(model).fit(X.iloc[tr], y[tr]).predict(X.iloc[te])
            leak.append(overlap_leak_rate(meta, tr, te, win_s))
        results[(mname, "random_kfold")] = {**_scores(y, pred), "test_windows_overlapping_train": float(np.mean(leak))}

        # B and C: leave-one-subject-out, raw and subject-standardized
        for proto, Xp in (("loso", X), ("loso_subject_norm", subject_standardize(X, groups))):
            pred = np.empty_like(y)
            for tr, te in LeaveOneGroupOut().split(Xp, y, groups):
                pred[te] = clone(model).fit(Xp.iloc[tr], y[tr]).predict(Xp.iloc[te])
                s = groups[te][0]
                per_subject.append(dict(model=mname, protocol=proto, subject=s,
                                        **_scores(y[te], pred[te]) if len(np.unique(y[te])) > 1
                                        else dict(balanced_accuracy=np.nan, macro_f1=np.nan)))
            results[(mname, proto)] = {**_scores(y, pred), "test_windows_overlapping_train": 0.0}
    return results, pd.DataFrame(per_subject)


def loso_permutation_importance(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray,
                                n_repeats: int = 5) -> pd.Series:
    """Permutation importance measured on held-out subjects only (random forest)."""
    model = models()["random_forest"]
    imps = []
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        if len(np.unique(y[te])) < 2:
            continue
        fitted = clone(model).fit(X.iloc[tr], y[tr])
        r = permutation_importance(fitted, X.iloc[te], y[te], n_repeats=n_repeats,
                                   random_state=SEED, scoring="balanced_accuracy")
        imps.append(r.importances_mean)
    return pd.Series(np.mean(imps, axis=0), index=X.columns).sort_values(ascending=False)
