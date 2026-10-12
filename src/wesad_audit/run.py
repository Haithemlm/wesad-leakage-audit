"""End-to-end audit: python -m wesad_audit.run --data /path/to/WESAD --out results"""
from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn

from .data import cut_windows, find_subjects, load_subject
from .evaluate import loso_permutation_importance, run_protocols
from .features import window_features

PROTO_LABEL = {"random_kfold": "Random window k-fold\n(leaky)",
               "loso": "Leave-one-subject-out",
               "loso_subject_norm": "LOSO + per-subject\nstandardization"}


def build_table(root: Path, win_s: float, step_s: float):
    rows, meta = [], []
    for p in find_subjects(root):
        wrist, labels, label_fs = load_subject(p)
        for w in cut_windows(wrist, labels, p.stem, win_s, step_s, label_fs):
            rows.append(window_features(w.signals))
            meta.append(dict(subject=w.subject, t_start=w.t_start, label=w.label))
        print(f"  {p.stem}: {sum(m['subject'] == p.stem for m in meta)} windows")
    X = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
    meta = pd.DataFrame(meta)
    return X, meta.label.values, meta


def plots(res: pd.DataFrame, per_subj: pd.DataFrame, imp: pd.Series, out: Path):
    fig, ax = plt.subplots(figsize=(7, 4))
    piv = res.pivot(index="protocol", columns="model", values="balanced_accuracy").loc[list(PROTO_LABEL)]
    x = np.arange(len(piv))
    for i, m in enumerate(piv.columns):
        bars = ax.bar(x + (i - 0.5) * 0.38, piv[m], 0.38, label=m)
        ax.bar_label(bars, fmt="%.2f", fontsize=8)
    ax.axhline(0.5, color="grey", lw=0.8, ls="--")
    ax.set_xticks(x, [PROTO_LABEL[p] for p in piv.index], fontsize=8)
    ax.set_ylim(0, 1.05); ax.set_ylabel("Balanced accuracy"); ax.legend(frameon=False, fontsize=8)
    ax.set_title("Same features, same model: only the validation changes", fontsize=10)
    fig.tight_layout(); fig.savefig(out / "fig_protocols.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    d = per_subj[(per_subj.model == "random_forest") & (per_subj.protocol == "loso")].sort_values("balanced_accuracy")
    ax.bar(d.subject, d.balanced_accuracy, color="#5b7bb2")
    ax.axhline(0.5, color="grey", lw=0.8, ls="--")
    ax.set_ylim(0, 1.05); ax.set_ylabel("Balanced accuracy"); ax.tick_params(axis="x", labelsize=8)
    ax.set_title("Per-subject LOSO accuracy (random forest): the spread a single average hides", fontsize=10)
    fig.tight_layout(); fig.savefig(out / "fig_per_subject.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    top = imp.head(12)[::-1]
    ax.barh(top.index, top.values, color="#5b7bb2"); ax.tick_params(axis="y", labelsize=8)
    ax.set_xlabel("Drop in balanced accuracy on held-out subjects")
    ax.set_title("Permutation importance (LOSO, random forest)", fontsize=10)
    fig.tight_layout(); fig.savefig(out / "fig_importance.png", dpi=160); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path, help="WESAD root folder containing S2..S17")
    ap.add_argument("--out", default=Path("results"), type=Path)
    ap.add_argument("--win", type=float, default=60.0, help="window length, seconds")
    ap.add_argument("--step", type=float, default=5.0, help="window step, seconds")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    synthetic = (a.data / "SYNTHETIC_DATA_NOT_WESAD").exists()

    print("Extracting features...")
    X, y, meta = build_table(a.data, a.win, a.step)
    print(f"{len(X)} windows, {X.shape[1]} features, {meta.subject.nunique()} subjects, stress share {y.mean():.2f}")

    print("Running protocols...")
    results, per_subj = run_protocols(X, y, meta, a.win)
    res = pd.DataFrame([dict(model=m, protocol=p, **v) for (m, p), v in results.items()])
    print(res.round(3).to_string(index=False))

    print("Permutation importance (LOSO)...")
    imp = loso_permutation_importance(X, y, meta.subject.values)

    plots(res, per_subj, imp, a.out)
    res.to_csv(a.out / "protocol_scores.csv", index=False)
    per_subj.to_csv(a.out / "per_subject_scores.csv", index=False)
    imp.to_csv(a.out / "loso_permutation_importance.csv", header=["importance"])
    json.dump(dict(dataset="SYNTHETIC (not WESAD) - do not publish" if synthetic else "WESAD wrist (Empatica E4)",
                   n_windows=int(len(X)), n_features=int(X.shape[1]), n_subjects=int(meta.subject.nunique()),
                   window_s=a.win, step_s=a.step, stress_share=float(y.mean()),
                   python=platform.python_version(), sklearn=sklearn.__version__,
                   run_utc=datetime.now(timezone.utc).isoformat(timespec="seconds")),
              open(a.out / "run_info.json", "w"), indent=2)
    print(f"Done. Outputs in {a.out}/")


if __name__ == "__main__":
    main()
