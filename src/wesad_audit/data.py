"""Load WESAD wrist (Empatica E4) signals and cut labelled windows.

WESAD layout: <root>/S<k>/S<k>.pkl, a pickle (latin1) with
  data['signal']['wrist'] = {'ACC': (n,3) @32 Hz, 'BVP': (n,1) @64 Hz,
                             'EDA': (n,1) @4 Hz,  'TEMP': (n,1) @4 Hz}
  data['label'] = (n,) @700 Hz, 0 undefined, 1 baseline, 2 stress, 3 amusement, 4 meditation
Subjects S2..S17 (S1 and S12 do not exist): 15 subjects.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FS = {"ACC": 32, "BVP": 64, "EDA": 4, "TEMP": 4}
LABEL_FS = 700
BASELINE, STRESS = 1, 2


@dataclass
class Window:
    subject: str
    t_start: float  # seconds from recording start
    label: int  # 0 = baseline, 1 = stress
    signals: dict  # name -> np.ndarray for this window


def load_subject(path: Path) -> tuple[dict, np.ndarray]:
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="latin1")
    wrist = {k: np.asarray(v, dtype=float) for k, v in d["signal"]["wrist"].items()}
    return wrist, np.asarray(d["label"]).ravel()


def find_subjects(root: Path) -> list[Path]:
    paths = sorted(root.glob("S*/S*.pkl"), key=lambda p: int(p.stem[1:]))
    if not paths:
        raise FileNotFoundError(f"No S*/S*.pkl files under {root}")
    return paths


def cut_windows(wrist: dict, labels: np.ndarray, subject: str,
                win_s: float = 60.0, step_s: float = 5.0) -> list[Window]:
    """Windows whose label is constant (pure baseline or pure stress)."""
    dur = len(labels) / LABEL_FS
    for name, x in wrist.items():
        dur = min(dur, len(x) / FS[name])
    out: list[Window] = []
    t = 0.0
    while t + win_s <= dur:
        lab = labels[int(t * LABEL_FS): int((t + win_s) * LABEL_FS)]
        first = lab[0]
        if first in (BASELINE, STRESS) and np.all(lab == first):
            sig = {n: x[int(t * FS[n]): int(t * FS[n]) + int(win_s * FS[n])] for n, x in wrist.items()}
            out.append(Window(subject, t, int(first == STRESS), sig))
        t += step_s
    return out
