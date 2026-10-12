"""Shrink WESAD (~2 GB) to the wrist signals + labels the audit needs (~40-60 MB).

Usage (only needs numpy):
    python tools/extract_wrist.py /path/to/WESAD wesad_wrist_compact.zip

Keeps: wrist ACC (32 Hz), BVP (64 Hz), EDA (4 Hz), TEMP (4 Hz) as float32,
and the condition labels downsampled from 700 Hz to 4 Hz.
Drops: all chest (RespiBAN) signals, questionnaires, raw E4 CSVs.
"""
import io
import pickle
import sys
import zipfile
from pathlib import Path

import numpy as np

LABEL_FS_IN, LABEL_FS_OUT = 700, 4


def main(root: Path, out: Path):
    paths = sorted(root.glob("S*/S*.pkl"), key=lambda p: int(p.stem[1:]))
    if not paths:
        sys.exit(f"No S*/S*.pkl under {root}. Point this at the unzipped WESAD folder.")
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in paths:
            with open(p, "rb") as f:
                d = pickle.load(f, encoding="latin1")
            wrist = {k: np.asarray(v, dtype=np.float32) for k, v in d["signal"]["wrist"].items()}
            label = np.asarray(d["label"]).ravel()[:: LABEL_FS_IN // LABEL_FS_OUT].astype(np.int8)
            buf = io.BytesIO()
            pickle.dump({"signal": {"wrist": wrist}, "label": label,
                         "label_fs": LABEL_FS_OUT, "subject": p.stem}, buf, protocol=4)
            z.writestr(f"{p.stem}/{p.stem}.pkl", buf.getvalue())
            print(f"  {p.stem} done")
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.1f} MB). Upload this file to the chat.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(Path(sys.argv[1]), Path(sys.argv[2]))
