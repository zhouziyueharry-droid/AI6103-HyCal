"""Figures 2 and 3 for the HyCal pooled-shrinkage report.

Numbers are copied from partC/results/param_search/ablation_shrink.md (alpha sweep,
test split, lambda=1e-4, n_min=2, scale=1) and from the OrganAMNIST per-class
3-seed means (partC/results/paired_transfer/organamnist/predictions.csv).
Run:  python make_plots.py <output_dir>
"""
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
OUT.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 7.5,
    "axes.titlesize": 7.5,
    "axes.labelsize": 7.5,
    "xtick.labelsize": 6.8,
    "ytick.labelsize": 6.8,
    "legend.fontsize": 6.5,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "pdf.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

INK = "#222222"
MUTED = "#8a8a8a"
GRID = "#e6e6e6"
BLUE = "#1f5fa8"      # EuroSAT / shrinkage
BLUE_L = "#7fa8d6"
ORANGE = "#c8641c"    # Galaxy10
ORANGE_L = "#eaa673"
RED = "#b8322a"
COL_W = 3.31          # AAAI column width (in)

# ---------------------------------------------------------------- Figure 2: alpha sweep
alphas = [0, 0.25, 0.5, 0.75, 1.0]
acc = {  # test accuracy (%), mean over seeds 0/1/42
    ("EuroSAT", 5): [76.60, 79.77, 79.65, 79.53, 79.53],
    ("EuroSAT", 10): [83.49, 86.10, 86.24, 86.40, 86.12],
    ("Galaxy10", 5): [37.99, 40.02, 40.23, 39.70, 39.38],
    ("Galaxy10", 10): [39.74, 43.14, 43.63, 43.74, 44.19],
}
paired = {("EuroSAT", 5): 2.93, ("EuroSAT", 10): 2.92,  # paired gains at alpha=0.75 (Table 5)
          ("Galaxy10", 5): 1.71, ("Galaxy10", 10): 4.00}
style = {
    ("EuroSAT", 5): dict(color=BLUE_L, ls=(0, (3, 1.6)), marker="o", mfc="white"),
    ("EuroSAT", 10): dict(color=BLUE, ls="-", marker="o", mfc=BLUE),
    ("Galaxy10", 5): dict(color=ORANGE_L, ls=(0, (3, 1.6)), marker="s", mfc="white"),
    ("Galaxy10", 10): dict(color=ORANGE, ls="-", marker="s", mfc=ORANGE),
}

fig, ax = plt.subplots(figsize=(COL_W, 1.70))
ax.axvspan(0.70, 0.80, color="#eef3fa", zorder=0, lw=0)
ax.axhline(0, color=MUTED, lw=0.6, zorder=1)
for key, ys in acc.items():
    gains = [y - ys[0] for y in ys]
    s = style[key]
    ax.plot(alphas, gains, color=s["color"], ls=s["ls"], lw=1.25, marker=s["marker"],
            ms=3.6, mfc=s["mfc"], mec=s["color"], mew=0.9, zorder=3,
            label=f"{key[0]}, $K{{=}}{key[1]}$")
    below = key in (("Galaxy10", 5), ("EuroSAT", 10))
    ax.annotate(f"{paired[key]:+.2f}", (0.75, gains[3]), xytext=(-9, 3.2 if not below else -8.5),
                textcoords="offset points", ha="center", fontsize=6.0, color=s["color"],
                bbox=dict(boxstyle="square,pad=0.05", fc="white", ec="none", alpha=0.85),
                zorder=4)
ax.text(0.75, -0.62, r"selected $\alpha$", ha="center", va="center", fontsize=6.2,
        color=BLUE, style="italic")
ax.text(0.015, -0.62, r"Tuned HyCal ($\alpha{=}0$)", ha="left", va="center", fontsize=6.2,
        color=MUTED)
ax.set_xlim(-0.04, 1.04)
ax.set_ylim(-1.0, 6.3)
ax.set_xticks(alphas)
ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
ax.set_yticks([0, 1, 2, 3, 4, 5])
ax.spines["left"].set_bounds(-1.0, 5.0)
ax.set_xlabel(r"shrinkage coefficient $\alpha$", labelpad=1.5)
ax.set_ylabel(r"gain over $\alpha{=}0$ (pp)", labelpad=2)
ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
ax.legend(loc="upper left", ncol=2, frameon=False, handlelength=2.4, columnspacing=1.0,
          borderaxespad=0.1, handletextpad=0.5)
fig.tight_layout(pad=0.25)
fig.savefig(OUT / "fig_alpha.pdf")
fig.savefig(OUT / "fig_alpha.png", dpi=300)
plt.close(fig)

# ---------------------------------------------------------------- Figure 3: OrganAMNIST per class
organ = [  # name, 5-shot a0, a0.75, 10-shot a0, a0.75  (sorted by 5-shot gain)
    ("Pancreas", 62.5, 80.2, 78.9, 88.2, 17.7, 9.3),
    ("Left lung", 68.6, 75.9, 64.1, 78.7, 7.3, 14.7),
    ("Left femur", 52.0, 58.3, 56.3, 58.6, 6.3, 2.3),
    ("Spleen", 45.5, 51.7, 60.2, 71.7, 6.2, 11.5),
    ("Right lung", 81.5, 87.5, 89.5, 92.7, 6.0, 3.2),
    ("Left kidney", 30.4, 35.3, 32.1, 33.0, 4.8, 0.9),
    ("Right kidney", 31.5, 35.0, 37.7, 49.1, 3.5, 11.4),
    ("Bladder", 41.0, 43.9, 52.1, 60.2, 2.9, 8.0),
    ("Heart", 66.3, 68.7, 75.0, 75.9, 2.4, 0.9),
    ("Liver", 93.6, 93.2, 96.3, 96.9, -0.4, 0.6),
    ("Right femur", 42.2, 39.9, 46.8, 49.1, -2.2, 2.3),
]
overall = {5: (59.37, 64.30), 10: (65.61, 71.63)}
n = len(organ)
fig, axes = plt.subplots(1, 2, figsize=(COL_W, 2.38), sharey=True)
for j, (ax, K) in enumerate(zip(axes, (5, 10))):
    for i, row in enumerate(organ):
        y = n - 1 - i
        a0, a1 = (row[1], row[2]) if K == 5 else (row[3], row[4])
        d = row[5] if K == 5 else row[6]   # 3-seed mean of per-seed deltas
        c = BLUE if d >= 0 else RED
        ax.plot([a0, a1], [y, y], color=c, lw=1.6 if abs(d) > 0.8 else 1.0, alpha=0.55,
                solid_capstyle="butt", zorder=2)
        ax.plot(a0, y, "o", ms=3.3, mfc="white", mec=MUTED, mew=0.9, zorder=3)
        ax.plot(a1, y, "o", ms=3.5, mfc=c, mec=c, mew=0.6, zorder=4)
        ax.text(101.5, y, f"{d:+.1f}".replace("-", "−"), va="center", ha="left", fontsize=6.0, color=c)
    # overall row
    y = -1.35
    a0, a1 = overall[K]
    ax.plot([a0, a1], [y, y], color=BLUE, lw=1.8, alpha=0.55, zorder=2)
    ax.plot(a0, y, "D", ms=3.3, mfc="white", mec=INK, mew=0.9, zorder=3)
    ax.plot(a1, y, "D", ms=3.5, mfc=INK, mec=INK, zorder=4)
    ax.text(101.5, y, f"{(4.92 if K == 5 else 6.02):+.2f}", va="center", ha="left", fontsize=6.0, color=INK,
            weight="bold")
    ax.axhline(-0.68, color=GRID, lw=0.7)
    ax.set_xlim(25, 100)
    ax.set_xticks([30, 50, 70, 90])
    ax.set_ylim(-2.0, n - 0.4)
    ax.grid(axis="x", color=GRID, lw=0.5, zorder=0)
    ax.set_title(f"{K}-shot", pad=2.5, fontsize=7.2)
    ax.set_xlabel("accuracy (%)", labelpad=1.0)
    ax.spines["left"].set_visible(j == 0)
    if j:
        ax.tick_params(axis="y", length=0)
axes[0].set_yticks(list(range(n - 1, -1, -1)) + [-1.35])
axes[0].set_yticklabels([r[0] for r in organ] + ["Overall"])
axes[0].get_yticklabels()[-1].set_fontweight("bold")
handles = [
    Line2D([], [], marker="o", ls="", mfc="white", mec=MUTED, ms=3.6, label=r"$\alpha=0$"),
    Line2D([], [], marker="o", ls="", mfc=BLUE, mec=BLUE, ms=3.6, label=r"$\alpha=0.75$"),
    Line2D([], [], marker="o", ls="", mfc=RED, mec=RED, ms=3.6, label="drop"),
]
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
           bbox_to_anchor=(0.6, -0.01), handletextpad=0.2, columnspacing=1.2)
fig.tight_layout(pad=0.25, w_pad=1.6, rect=(0, 0.06, 1, 1))
fig.savefig(OUT / "fig_organ.pdf")
fig.savefig(OUT / "fig_organ.png", dpi=300)
plt.close(fig)
print("ok")
