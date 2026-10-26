"""
Charts for project_explanation.md (section A14) and for the PDF.

All the numbers are the ones MEASURED and reported in the .md (the section they come from is written
next to every value): nothing new is computed here, we only draw.

Usage (from documentation/charts/):  python make_charts.py   -> writes the .png files in this folder
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

# validated palette (slots 1 and 2 of the reference palette, light theme)
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.dpi": 200,
})


def num(v, fmt):
    """Number formatted as in the text (decimal point)."""
    return format(v, fmt)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, name), bbox_inches="tight")
    plt.close(fig)


def bar_labels(ax, bars, fmt):
    for b in bars:
        ax.annotate(fmt(b.get_height()), (b.get_x() + b.get_width() / 2, b.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", color=INK, fontsize=9)


def two_series(ax, cats, a, b, la, lb, fmt):
    x = range(len(cats))
    w = 0.36
    ba = ax.bar([i - w / 2 - 0.01 for i in x], a, w, color=BLUE, label=la)
    bb = ax.bar([i + w / 2 + 0.01 for i in x], b, w, color=ORANGE, label=lb)
    ax.set_xticks(list(x), cats)
    bar_labels(ax, ba, fmt)
    bar_labels(ax, bb, fmt)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, 1.12), ncol=2)
    ax.grid(axis="x", visible=False)


def two_series_dots(ax, cats, a, b, la, lb, fmt):
    """Two series as dots (no bars: the axis does not start from zero, bars would exaggerate)."""
    x = list(range(len(cats)))
    off = 0.12
    ax.scatter([i - off for i in x], a, s=70, color=BLUE, label=la, zorder=3, edgecolors=SURFACE, linewidths=1.5)
    ax.scatter([i + off for i in x], b, s=70, color=ORANGE, label=lb, zorder=3, edgecolors=SURFACE, linewidths=1.5)
    for i in x:
        ax.annotate(fmt(a[i]), (i - off, a[i]), xytext=(-8, 0), textcoords="offset points", ha="right", va="center",
                    color=INK, fontsize=9)
        ax.annotate(fmt(b[i]), (i + off, b[i]), xytext=(8, 0), textcoords="offset points", ha="left", va="center",
                    color=INK, fontsize=9)
    ax.set_xticks(x, cats)
    ax.set_xlim(-0.6, len(cats) - 0.4)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, 1.12), ncol=2)
    ax.grid(axis="x", visible=False)


# 1. ROC thresholds per kind of H0 (§A7.2)
fig, ax = plt.subplots(figsize=(6.4, 3.2))
kinds = ["H0-R\nrandom watermarks", "H0-A\nattacked original", "H0-B\nother group attacked"]
vals = [0.0835, 0.0841, 0.0867]
ax.scatter(kinds, vals, s=80, color=BLUE, zorder=3, edgecolors=SURFACE, linewidths=1.5)
for k, v in zip(kinds, vals):
    ax.annotate(num(v, ".4f"), (k, v), xytext=(10, 0), textcoords="offset points", va="center", color=INK, fontsize=9)
ax.set_xlim(-0.5, 2.6)
ax.axhline(0.087, color=INK, lw=1.2, ls="--")
ax.annotate("chosen τ = 0.087", (-0.35, 0.087), xytext=(0, 4), textcoords="offset points", ha="left", color=INK)
ax.set_ylim(0.080, 0.090)
ax.set_ylabel("threshold for FPR ≤ 0.1%")
ax.set_title("Full ROC: the highest threshold (H0-B) decides τ", loc="left", color=INK)
ax.grid(axis="x", visible=False)
save(fig, "g1_roc_thresholds.png")

# 2. FPR at tau 0.0867: calibration against test (§A7.2)
fig, ax = plt.subplots(figsize=(6.4, 3.2))
two_series(ax, ["H0-R", "H0-A", "H0-B"], [0.046, 0.035, 0.070], [0.058, 0.0, 0.0],
           "calibration (71 images)", "test (30 images never used)", lambda v: num(v, ".3f") + "%")
ax.axhline(0.1, color=INK, lw=1.2, ls="--")
ax.annotate("limit we set ourselves: 0.1%", (2.45, 0.1), xytext=(0, 4), textcoords="offset points", ha="right",
            color=INK)
ax.set_ylim(0, 0.125)
ax.set_ylabel("false positives (%)")
ax.set_title("False positives at the threshold 0.0867", loc="left", color=INK, pad=24)
save(fig, "g2_false_positives.png")

# 3. robustness: broken images, v1.0 against v2.0 (§A8.2, 104 images, 62.05 dB)
advs = ["fixed battery", "adaptive\nglobal only", "adaptive\n+ zones"]
fig, ax = plt.subplots(figsize=(6.4, 3.2))
two_series(ax, advs, [72, 73, 73], [75, 70, 70], "v1.0 (DCT only)", "v2.0 hybrid (submitted)", lambda v: f"{v:.0f}%")
ax.set_ylim(0, 100)
ax.set_ylabel("broken images (%)")
ax.set_title("How many images get broken (62.05 dB, 104 images)", loc="left", color=INK, pad=24)
save(fig, "g3_robustness_broken.png")

# 4. robustness: mean WPSNR of the successful attacks (§A8.2)
fig, ax = plt.subplots(figsize=(6.4, 3.2))
two_series_dots(ax, advs, [37.10, 37.00, 37.24], [36.92, 36.76, 37.01], "v1.0 (DCT only)", "v2.0 hybrid (submitted)",
                lambda v: num(v, ".2f"))
ax.axhspan(35, 38, color=GRID, alpha=0.6, zorder=0)
ax.annotate("35–38 dB band = 6 robustness points", (-0.55, 37.9), color=INK2, fontsize=9, va="top")
ax.set_ylim(35, 39)
ax.set_ylabel("mean WPSNR of successful attacks (dB)")
ax.set_title("How much whoever breaks us has to ruin the image", loc="left", color=INK, pad=24)
save(fig, "g4_robustness_wpsnr.png")

# 5. expected defense points, 62.05 against 58.05 (§A8.3, 20,000 simulated triples)
fig, ax = plt.subplots(figsize=(6.4, 3.2))
two_series_dots(ax, ["target 62.05 dB", "target 58.05 dB"], [10.79, 11.04], [10.88, 11.10], "v1.0 (DCT only)",
                "v2.0 hybrid (submitted)", lambda v: num(v, ".2f"))
ax.set_ylim(10.5, 11.3)
ax.set_ylabel("expected defense points")
ax.set_title("Expected points (quality + robustness + bonus), adaptive attacker", loc="left", color=INK, pad=24)
save(fig, "g5_expected_points.png")

# 6. which attack breaks us (§A8.2, v2.0, adaptive, 73 broken images)
fams = [("JPEG + blur 0.8", 21), ("blur 0.6 + JPEG", 12), ("median 3 + JPEG", 11), ("noise + JPEG", 11),
        ("blur 1.0 + JPEG", 8), ("resize 0.5 + JPEG", 5), ("JPEG alone", 4), ("noise + blur", 1)]
fig, ax = plt.subplots(figsize=(6.4, 3.4))
names, counts = [f[0] for f in fams][::-1], [f[1] for f in fams][::-1]
bars = ax.barh(names, counts, 0.6, color=BLUE)
for b in bars:
    ax.annotate(f"{int(b.get_width())}", (b.get_width(), b.get_y() + b.get_height() / 2), xytext=(4, 0),
                textcoords="offset points", va="center", color=INK, fontsize=9)
ax.set_xlabel("images broken with that attack")
ax.set_title("The enemy is JPEG: 72 breaks out of 73 contain it", loc="left", color=INK)
ax.grid(axis="y", visible=False)
save(fig, "g6_winning_attacks.png")

# 7. simulation of the attack phase: WPSNR of the 18 images broken out of 27 (§A11)
broken = [37.82, 35.51, 35.51, 36.22, 37.99, 37.34, 36.36, 36.75, 37.26,
          36.16, 35.62, 37.00, 36.00, 35.57, 35.61, 37.46, 36.14, 36.34]
fig, ax = plt.subplots(figsize=(6.4, 2.6))
xs = sorted(broken)
ys = []                                       # almost equal values stacked, so they do not cover each other
for i, x in enumerate(xs):
    ys.append(0.16 * sum(1 for p in xs[:i] if abs(p - x) < 0.06))
ys = [y - 0.08 for y in ys]
ax.scatter(xs, ys, s=60, color=BLUE, edgecolors=SURFACE, linewidths=1.5, zorder=3)
ax.axvline(sum(broken) / len(broken), color=INK, lw=1.2, ls="--")
ax.annotate(f"mean {num(sum(broken) / len(broken), '.2f')} dB", (sum(broken) / len(broken), 0.32), xytext=(4, 0),
            textcoords="offset points", color=INK)
ax.set_xlim(35, 38.5)
ax.set_ylim(-0.5, 0.5)
ax.set_yticks([])
ax.set_xlabel("WPSNR of the successful attack (dB)")
ax.set_title("Simulation: 18 images broken out of 27 in 32 minutes", loc="left", color=INK)
ax.grid(axis="y", visible=False)
save(fig, "g7_attack_simulation.png")

# 8. detection time against the 5 s limit (§A1.1, A12)
fig, ax = plt.subplots(figsize=(6.4, 3.0))
cases = ["free CPU\n(demo)", "busy CPU\n(13 processes, 19/10)", "busy CPU\n(mean, 14/10)",
         "busy CPU\n(maximum, 14/10)"]
vals = [0.56, 1.7, 2.2, 3.1]
bars = ax.bar(cases, vals, 0.5, color=BLUE)
bar_labels(ax, bars, lambda v: num(v, ".2f") + " s")
ax.axhline(5, color=INK, lw=1.2, ls="--")
ax.annotate("limit of the rules: 5 s", (3.3, 5), xytext=(0, 4), textcoords="offset points", ha="right", color=INK)
ax.set_ylim(0, 5.8)
ax.set_ylabel("seconds per call")
ax.set_title("Detection time", loc="left", color=INK)
ax.grid(axis="x", visible=False)
save(fig, "g8_detection_times.png")

print("charts written in", HERE)
