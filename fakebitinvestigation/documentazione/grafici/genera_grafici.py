"""
Grafici per spiegazione_progetto.md (sezione A15) e per il PDF.

Tutti i numeri sono quelli MISURATI e riportati nel .md (la sezione di provenienza e' indicata
accanto a ogni dato): qui non si calcola niente di nuovo, si disegna e basta.

Uso (da documentazione/grafici/):  python genera_grafici.py   -> scrive i .png in questa cartella
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, ScalarFormatter  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

# palette validata (slot 1 e 2 della palette di riferimento, tema chiaro)
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.dpi": 200,
})


def it(v, fmt):
    """Numero con la virgola decimale (stile italiano, come nel testo)."""
    return format(v, fmt).replace(".", ",")


COMMA = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))


def save(fig, name):
    for ax in fig.axes:                       # virgola decimale solo sugli assi numerici (non sulle categorie)
        for axis in (ax.xaxis, ax.yaxis):
            if isinstance(axis.get_major_formatter(), ScalarFormatter):
                axis.set_major_formatter(COMMA)
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
    """Due serie come punti (niente barre: l'asse non parte da zero, le barre esagererebbero)."""
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


# 1. soglie della ROC per tipo di H0 (§A7.2)
fig, ax = plt.subplots(figsize=(6.4, 3.2))
kinds = ["H0-R\nwatermark casuali", "H0-A\noriginale attaccata", "H0-B\naltro gruppo attaccato"]
vals = [0.0835, 0.0841, 0.0867]
ax.scatter(kinds, vals, s=80, color=BLUE, zorder=3, edgecolors=SURFACE, linewidths=1.5)
for k, v in zip(kinds, vals):
    ax.annotate(it(v, ".4f"), (k, v), xytext=(10, 0), textcoords="offset points", va="center", color=INK, fontsize=9)
ax.set_xlim(-0.5, 2.6)
ax.axhline(0.087, color=INK, lw=1.2, ls="--")
ax.annotate("τ scelta = 0,087", (-0.35, 0.087), xytext=(0, 4), textcoords="offset points", ha="left", color=INK)
ax.set_ylim(0.080, 0.090)
ax.set_ylabel("soglia per FPR ≤ 0,1%")
ax.set_title("ROC completa: la soglia più alta (H0-B) decide τ", loc="left", color=INK)
ax.grid(axis="x", visible=False)
save(fig, "g1_soglie_roc.png")

# 2. FPR alla tau 0.0867: calibrazione contro test (§A7.2)
fig, ax = plt.subplots(figsize=(6.4, 3.2))
two_series(ax, ["H0-R", "H0-A", "H0-B"], [0.046, 0.035, 0.070], [0.058, 0.0, 0.0],
           "calibrazione (71 immagini)", "test (30 immagini mai usate)", lambda v: it(v, ".3f") + "%")
ax.axhline(0.1, color=INK, lw=1.2, ls="--")
ax.annotate("limite che ci siamo dati: 0,1%", (2.45, 0.1), xytext=(0, 4), textcoords="offset points", ha="right",
            color=INK)
ax.set_ylim(0, 0.125)
ax.set_ylabel("falsi positivi (%)")
ax.set_title("Falsi positivi alla soglia 0,0867", loc="left", color=INK, pad=24)
save(fig, "g2_falsi_positivi.png")

# 3. robustezza: immagini rotte, v1.0 contro v2.0 (§A8.2, 104 immagini, 62,05 dB)
advs = ["batteria fissa", "adattivo\nsolo globale", "adattivo\n+ zone"]
fig, ax = plt.subplots(figsize=(6.4, 3.2))
two_series(ax, advs, [72, 73, 73], [75, 70, 70], "v1.0 (solo DCT)", "v2.0 ibrida (consegnata)", lambda v: f"{v:.0f}%")
ax.set_ylim(0, 100)
ax.set_ylabel("immagini rotte (%)")
ax.set_title("Quante immagini ci rompono (62,05 dB, 104 immagini)", loc="left", color=INK, pad=24)
save(fig, "g3_robustezza_rotte.png")

# 4. robustezza: WPSNR medio degli attacchi riusciti (§A8.2)
fig, ax = plt.subplots(figsize=(6.4, 3.2))
two_series_dots(ax, advs, [37.10, 37.00, 37.24], [36.92, 36.76, 37.01], "v1.0 (solo DCT)", "v2.0 ibrida (consegnata)",
           lambda v: it(v, ".2f"))
ax.axhspan(35, 38, color=GRID, alpha=0.6, zorder=0)
ax.annotate("fascia 35–38 dB = 6 punti di robustezza", (-0.55, 37.9), color=INK2, fontsize=9, va="top")
ax.set_ylim(35, 39)
ax.set_ylabel("WPSNR medio attacchi riusciti (dB)")
ax.set_title("Quanto deve rovinare l'immagine chi ci rompe", loc="left", color=INK, pad=24)
save(fig, "g4_robustezza_wpsnr.png")

# 5. punti attesi di difesa, 62,05 contro 58,05 (§A8.3, 20.000 terne simulate)
fig, ax = plt.subplots(figsize=(6.4, 3.2))
two_series_dots(ax, ["target 62,05 dB", "target 58,05 dB"], [10.79, 11.04], [10.88, 11.10], "v1.0 (solo DCT)",
           "v2.0 ibrida (consegnata)", lambda v: it(v, ".2f"))
ax.set_ylim(10.5, 11.3)
ax.set_ylabel("punti di difesa attesi")
ax.set_title("Punti attesi (qualità + robustezza + bonus), attaccante adattivo", loc="left", color=INK, pad=24)
save(fig, "g5_punti_attesi.png")

# 6. con quale attacco ci rompono (§A8.2, v2.0, adattivo, 73 immagini rotte)
fams = [("JPEG + blur 0,8", 21), ("blur 0,6 + JPEG", 12), ("mediano 3 + JPEG", 11), ("rumore + JPEG", 11),
        ("blur 1,0 + JPEG", 8), ("resize 0,5 + JPEG", 5), ("JPEG da solo", 4), ("rumore + blur", 1)]
fig, ax = plt.subplots(figsize=(6.4, 3.4))
names, counts = [f[0] for f in fams][::-1], [f[1] for f in fams][::-1]
bars = ax.barh(names, counts, 0.6, color=BLUE)
for b in bars:
    ax.annotate(f"{int(b.get_width())}", (b.get_width(), b.get_y() + b.get_height() / 2), xytext=(4, 0),
                textcoords="offset points", va="center", color=INK, fontsize=9)
ax.set_xlabel("immagini rotte con quell'attacco")
ax.set_title("Il nemico è il JPEG: 72 rotture su 73 lo contengono", loc="left", color=INK)
ax.grid(axis="y", visible=False)
save(fig, "g6_attacchi_vincenti.png")

# 7. simulazione della fase di attacco: WPSNR delle 18 immagini rotte su 27 (§A13)
broken = [37.82, 35.51, 35.51, 36.22, 37.99, 37.34, 36.36, 36.75, 37.26,
          36.16, 35.62, 37.00, 36.00, 35.57, 35.61, 37.46, 36.14, 36.34]
fig, ax = plt.subplots(figsize=(6.4, 2.6))
xs = sorted(broken)
ys = []                                       # valori quasi uguali impilati, per non coprirsi
for i, x in enumerate(xs):
    ys.append(0.16 * sum(1 for p in xs[:i] if abs(p - x) < 0.06))
ys = [y - 0.08 for y in ys]
ax.scatter(xs, ys, s=60, color=BLUE, edgecolors=SURFACE, linewidths=1.5, zorder=3)
ax.axvline(sum(broken) / len(broken), color=INK, lw=1.2, ls="--")
ax.annotate(f"media {it(sum(broken) / len(broken), '.2f')} dB", (sum(broken) / len(broken), 0.32), xytext=(4, 0),
            textcoords="offset points", color=INK)
ax.set_xlim(35, 38.5)
ax.set_ylim(-0.5, 0.5)
ax.set_yticks([])
ax.set_xlabel("WPSNR dell'attacco riuscito (dB)")
ax.set_title("Simulazione: 18 immagini rotte su 27 in 32 minuti", loc="left", color=INK)
ax.grid(axis="y", visible=False)
save(fig, "g7_simulazione_attacco.png")

# 8. tempo della detection contro il limite di 5 s (§A1.1, A12)
fig, ax = plt.subplots(figsize=(6.4, 3.0))
cases = ["CPU libera\n(demo)", "CPU occupata\n(13 processi, 19/10)", "CPU occupata\n(media, 14/10)",
         "CPU occupata\n(massimo, 14/10)"]
vals = [0.56, 1.7, 2.2, 3.1]
bars = ax.bar(cases, vals, 0.5, color=BLUE)
bar_labels(ax, bars, lambda v: it(v, ".2f") + " s")
ax.axhline(5, color=INK, lw=1.2, ls="--")
ax.annotate("limite del regolamento: 5 s", (3.3, 5), xytext=(0, 4), textcoords="offset points", ha="right", color=INK)
ax.set_ylim(0, 5.8)
ax.set_ylabel("secondi per chiamata")
ax.set_title("Tempo della detection", loc="left", color=INK)
ax.grid(axis="x", visible=False)
save(fig, "g8_tempi_detection.png")

print("grafici scritti in", HERE)
