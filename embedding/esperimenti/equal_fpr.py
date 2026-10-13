"""
Confronto tra configurazioni a PARITA' di falsi positivi (non a parita' di tau).
Per ogni configurazione: tau = quantile (1 - FPR) della H0 realistica (h0.py: watermark di un altro
gruppo + attacco, e originale + attacco, insieme), poi robustezza (A+B) a quella tau.
Uso: python equal_fpr.py
"""
import pickle
import numpy as np
from analyze import load, best_attacks, simval

CONFIGS = {  # nome: (file h0, [run batteria], detector)
    "g0  max(cos,sq)":    ("h0_v2", ["eA_base", "eB_base"], "max(cos|sq:16:0.5)"),
    "g0  max(rs,sq)":     ("h0_v2", ["eA_base", "eB_base"], "max(rs_cos|sq:16:0.5)"),
    "g05 max(cos,sq)":    ("h0_g05", ["eA_g05", "eB_g05"], "max(cos|sq:16:0.5)"),
    "g05 max(rs,sq)":     ("h0_g05", ["eA_g05", "eB_g05"], "max(rs_cos|sq:16:0.5)"),
}


def h0_scores(h, det):
    ds = det[4:-1].split("|") if det.startswith("max(") else [det]
    x = [np.maximum.reduce([h[c][d] for d in ds]) for c in ("altro_wm", "orig_attaccata")]
    o = np.maximum.reduce([h["orig"][d] for d in ds])
    return np.concatenate(x), o


for fpr in (0.001, 0.002):
    print(f"\n=== FPR realistica = {fpr}")
    for name, (hf, runs, det) in CONFIGS.items():
        h = pickle.load(open(f"risultati/{hf}.pkl", "rb"))
        x, o = h0_scores(h, det)
        tau = float(np.quantile(x, 1 - fpr))
        hit, n = [], 0
        for r in runs:
            b = best_attacks(load(r), det, tau)
            n += len(b)
            hit += [v["wpsnr"] for v in b.values() if v]
        print(f"  {name:18s} tau={tau:.4f}  originale max={o.max():.3f}  rotte {len(hit):3d}/{n} ({100*len(hit)/n:4.1f}%)  "
              f"WPSNR medio {np.mean(hit):.2f}  mediana {np.median(hit):.2f}")
