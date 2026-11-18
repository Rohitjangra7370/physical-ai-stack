"""Plot the ACT training curves from the CSVs in this folder.

    python3 plot_training.py        # writes training_curve.png

The CSVs were parsed from LeRobot's console log for the run: training loss every 200 steps,
held-out loss every 5000. The training loss is L1 + 10 * KL (kl_weight = 10). The held-out
loss is L1 only, because the VAE branch is off in eval mode, so the two levels are not
comparable.
"""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

HERE = Path(__file__).resolve().parent

SURFACE, INK, INK_2, GRID, SERIES = "#fcfcfb", "#0b0b0b", "#52514e", "#e7e6e2", "#2a78d6"
LINE = 1.44  # pt; 2 CSS px when the 200 dpi image is shown 900 px wide


def read(name, *cols):
    with open(HERE / name, newline="") as f:
        rows = list(csv.DictReader(f))
    return [[float(r[c]) for r in rows] for c in cols]


step, loss = read("train_loss.csv", "step", "loss")
h_step, h_loss = read("heldout_loss.csv", "step", "heldout_loss")

plt.rcParams.update({
    "font.size": 9,
    "text.color": INK,
    "axes.labelcolor": INK_2,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.4), facecolor=SURFACE)
for ax in (ax1, ax2):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0)
    ax.set_xlim(0, 20000)
    ax.xaxis.set_major_locator(FixedLocator([0, 5000, 10000, 15000, 20000]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v / 1000)}k" if v else "0"))
    ax.set_xlabel("training step")

# Left: training loss, log scale because it falls by two orders of magnitude.
ax1.plot(step, loss, color=SERIES, linewidth=LINE, solid_capstyle="round")
ax1.set_yscale("log")
ax1.set_ylim(0.04, 12)
ax1.yaxis.set_major_locator(FixedLocator([0.1, 1, 10]))
ax1.yaxis.set_minor_locator(NullLocator())
ax1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
ax1.set_title("Training loss (L1 + 10 × KL), log scale", loc="left", fontsize=10)
ax1.annotate(f"{loss[0]:.3f}", (step[0], loss[0]), xytext=(8, -1), textcoords="offset points",
             color=INK_2, va="center")
ax1.annotate(f"{loss[-1]:.3f}", (step[-1], loss[-1]), xytext=(0, 10), textcoords="offset points",
             color=INK_2, ha="right")

# Right: held-out loss, four points. Axis starts at zero so the drop is shown at its true size.
ax2.plot(h_step, h_loss, color=SERIES, linewidth=LINE, marker="o", markersize=6,
         markerfacecolor=SERIES, markeredgecolor=SURFACE, markeredgewidth=LINE)
ax2.set_ylim(0, 0.1)
ax2.yaxis.set_major_locator(FixedLocator([0, 0.025, 0.05, 0.075, 0.1]))
ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
ax2.set_title("Held-out loss (L1), 10 episodes", loc="left", fontsize=10)
ax2.annotate(f"{h_loss[0]:.4f}", (h_step[0], h_loss[0]), xytext=(0, 9), textcoords="offset points",
             color=INK_2, ha="center")
ax2.annotate(f"{h_loss[-1]:.4f}", (h_step[-1], h_loss[-1]), xytext=(0, 9), textcoords="offset points",
             color=INK_2, ha="right")

fig.tight_layout(w_pad=3)
fig.savefig(HERE / "training_curve.png", dpi=200, facecolor=SURFACE)
print("wrote", HERE / "training_curve.png")
