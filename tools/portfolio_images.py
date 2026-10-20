"""Portfolio images (4:3, large type, safe margins) built from results/*.csv.

python tools/portfolio_images.py results
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a64", "#d9d8d2", "#fcfcfb"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": GRID,
                     "axes.labelcolor": MUTED, "xtick.color": INK, "ytick.color": MUTED})

res = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
scores = pd.read_csv(res / "protocol_scores.csv").set_index(["model", "protocol"]).balanced_accuracy
per = pd.read_csv(res / "per_subject_scores.csv")


def frame(title, subtitle):
    fig = plt.figure(figsize=(8, 6), dpi=200, facecolor=SURFACE)
    fig.text(0.07, 0.92, title, fontsize=21, fontweight="bold", color=INK, va="top")
    fig.text(0.07, 0.835, subtitle, fontsize=12.5, color=MUTED, va="top")
    ax = fig.add_axes([0.10, 0.17, 0.84, 0.56], facecolor=SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, labelsize=12)
    fig.text(0.07, 0.04, "WESAD wrist signals, 15 subjects  |  github.com/Haithemlm/wesad-leakage-audit",
             fontsize=9.5, color=MUTED)
    return fig, ax


# 1. Cover: shuffled vs unseen people
fig, ax = frame("Shuffled validation: 1.00.\nUnseen people: 0.91.",
                "Same features, same model. Only the validation protocol changes.")
fig.texts[0].set_position((0.07, 0.95))
fig.texts[1].set_position((0.07, 0.815))
protos = [("random_kfold", "Shuffled windows\n(leaky)"), ("loso", "New, unseen\npeople")]
w = 0.34
for i, (m, name, c) in enumerate((("logreg", "Logistic regression", BLUE),
                                  ("random_forest", "Random forest", ORANGE))):
    for j, (p, _) in enumerate(protos):
        v = scores[(m, p)]
        x = j + (i - 0.5) * (w + 0.03)
        ax.bar(x, v, w, color=c, label=name if j == 0 else None)
        ax.text(x, v + 0.02, f"{v:.2f}", ha="center", fontsize=13, color=INK, fontweight="bold")
ax.set_xticks([0, 1], [n for _, n in protos], fontsize=13)
ax.set_xlim(-0.6, 1.6)
ax.set_ylabel("Balanced accuracy", fontsize=11)
ax.legend(frameon=False, fontsize=11, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.10))
ax.set_ylim(0, 1.14)
fig.savefig(res / "portfolio_cover.png", facecolor=SURFACE)
plt.close(fig)

# 2. Per-subject spread
d = per[(per.model == "random_forest") & (per.protocol == "loso")].sort_values("balanced_accuracy")
mean = d.balanced_accuracy.mean()
fig, ax = frame(f"Average {mean:.2f}. One person at chance.",
                "Random forest, tested on each person it never saw (leave-one-subject-out).")
colors = [ORANGE if v <= 0.55 else BLUE for v in d.balanced_accuracy]
ax.bar(range(len(d)), d.balanced_accuracy, 0.75, color=colors)
ax.axhline(0.5, color=MUTED, lw=1, ls="--")
ax.set_xlim(-0.6, len(d) + 0.6)
ax.text(len(d) - 0.35, 0.515, "chance", ha="left", va="bottom", fontsize=10, color=MUTED)
low = d.iloc[0]
ax.annotate(f"{low.subject}: {low.balanced_accuracy:.2f}", (0, low.balanced_accuracy),
            xytext=(-0.3, 0.97), fontsize=12, color=INK, fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=INK, lw=1))
ax.set_xticks(range(len(d)), d.subject, fontsize=9.5)
ax.set_ylabel("Balanced accuracy", fontsize=11)
fig.savefig(res / "portfolio_per_subject.png", facecolor=SURFACE)
plt.close(fig)
print("wrote", res / "portfolio_cover.png", res / "portfolio_per_subject.png")
