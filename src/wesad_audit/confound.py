"""Time-confound check: is the model detecting stress, or how long the watch has been on?

In the WESAD protocol the baseline condition is recorded before the stress condition for
every subject, and skin temperature drifts as the wrist sensor warms up. If elapsed time
alone predicts the label well under leave-one-subject-out, any feature that tracks elapsed
time can score well without measuring stress at all.

Run after the main audit:  python -m wesad_audit.confound --data /path/to/WESAD --out results
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import LeaveOneGroupOut

from .data import LABEL_FS, find_subjects, load_subject
from .evaluate import models
from .run import build_table


def condition_order(root: Path) -> pd.DataFrame:
    rows = []
    for p in find_subjects(root):
        _, labels, fs = load_subject(p)
        first = {c: np.argmax(labels == c) / fs / 60 for c in (1, 2) if np.any(labels == c)}
        rows.append(dict(subject=p.stem, baseline_start_min=round(first.get(1, np.nan), 1),
                         stress_start_min=round(first.get(2, np.nan), 1)))
    return pd.DataFrame(rows)


def loso(X: pd.DataFrame, y, groups, model) -> float:
    pred = np.empty_like(y)
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        pred[te] = clone(model).fit(X.iloc[tr], y[tr]).predict(X.iloc[te])
    return balanced_accuracy_score(y, pred)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--out", default=Path("results"), type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    order = condition_order(a.data)
    X, y, meta = build_table(a.data, 60.0, 5.0)
    g = meta.subject.values
    temp_cols = [c for c in X.columns if c.startswith("temp_")]
    variants = {
        "all 29 features": X,
        "without skin temperature": X.drop(columns=temp_cols),
        "elapsed time only": meta[["t_start"]].rename(columns={"t_start": "elapsed_s"}),
    }
    res = []
    for vname, Xv in variants.items():
        for mname, m in models().items():
            res.append(dict(features=vname, model=mname, loso_balanced_accuracy=loso(Xv, y, g, m)))
    res = pd.DataFrame(res)
    print(order.to_string(index=False))
    print(res.round(3).to_string(index=False))
    order.to_csv(a.out / "condition_order.csv", index=False)
    res.to_csv(a.out / "confound_check.csv", index=False)
    late = late_calm_test(a.data, a.out)
    print(late.round(3).to_string(index=False))
    json.dump({"baseline_before_stress_all_subjects":
               bool((order.baseline_start_min < order.stress_start_min).all())},
              open(a.out / "confound_summary.json", "w"), indent=2)



def late_calm_test(root: Path, out: Path, win_s: float = 60.0, step_s: float = 5.0) -> pd.DataFrame:
    """Train baseline-vs-stress on 14 subjects, then score the held-out subject's meditation
    windows. Meditation (label 4) is a calm condition recorded after stress. A model that
    reads physiology should call it calm; a model that reads elapsed time will call it stress."""
    from .data import BASELINE, STRESS, FS
    from .features import window_features
    rows, meta = [], []
    for p in find_subjects(root):
        wrist, labels, fs = load_subject(p)
        dur = min([len(labels) / fs] + [len(x) / FS[n] for n, x in wrist.items()])
        t = 0.0
        while t + win_s <= dur:
            lab = labels[int(t * fs): int((t + win_s) * fs)]
            if lab[0] in (BASELINE, STRESS, 4) and np.all(lab == lab[0]):
                sig = {n: x[int(t * FS[n]): int(t * FS[n]) + int(win_s * FS[n])] for n, x in wrist.items()}
                rows.append(window_features(sig))
                meta.append(dict(subject=p.stem, t_start=t, cond=int(lab[0])))
            t += step_s
    X = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
    meta = pd.DataFrame(meta)
    temp_cols = [c for c in X.columns if c.startswith("temp_")]
    res = []
    X["elapsed_s"] = meta.t_start.values
    feats = [c for c in X.columns if c != "elapsed_s"]
    for vname, cols in (("all 29 features", feats),
                        ("without skin temperature", [c for c in feats if c not in temp_cols]),
                        ("elapsed time only", ["elapsed_s"])):
        for mname, m in models().items():
            share = []
            for s in meta.subject.unique():
                tr = (meta.subject != s) & meta.cond.isin([BASELINE, STRESS])
                te = (meta.subject == s) & (meta.cond == 4)
                if te.sum() == 0:
                    continue
                fitted = clone(m).fit(X.loc[tr, cols], (meta.loc[tr, "cond"] == STRESS).astype(int))
                share.append(fitted.predict(X.loc[te, cols]).mean())
            res.append(dict(features=vname, model=mname,
                            meditation_windows_called_stress=float(np.mean(share))))
    res = pd.DataFrame(res)
    res.to_csv(out / "late_calm_test.csv", index=False)
    return res

if __name__ == "__main__":
    main()
