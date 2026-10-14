"""
Terza iterazione: confronto dei detector a PARITA' di FPR realistica (h0_v3) sulla batteria v3.
Uso: python equal_fpr_v3.py <tag batteria> [<tag batteria> ...]
"""
import collections, pickle, sys
import numpy as np
from analyze import load, best_attacks
from equal_fpr import h0_scores

DETS = {
    "attuale: max(rs_cos, sq)":            "max(rs_cos|sq:16:0.5)",
    "solo pr":                              "pr",
    "max(rs_cos, sq, pr)":                 "max(rs_cos|sq:16:0.5|pr)",
    "max(rs_cos, sq, pr, prsq)":           "max(rs_cos|sq:16:0.5|pr|prsq:16:0.5)",
    "max(rs_cos, prsq)":                   "max(rs_cos|prsq:16:0.5)",
}
h = pickle.load(open("risultati/h0_v3.pkl", "rb"))
runs = sys.argv[1:]
for fpr in (0.001,):
    print(f"=== FPR realistica {fpr}  (batterie: {runs})")
    for name, det in DETS.items():
        x, o = h0_scores(h, det)
        tau = float(np.quantile(x, 1 - fpr))
        for t in (tau, 0.11):
            hit, n, fam = [], 0, collections.Counter()
            for r in runs:
                b = best_attacks(load(r), det, t)
                n += len(b)
                hit += [v["wpsnr"] for v in b.values() if v]
                fam.update(v["family"] for v in b.values() if v)
            lab = "tau FPR" if t == tau else "tau 0.11"
            print(f"  {name:28s} {lab} {t:.4f} | H0 q99.9 {np.quantile(x, .999):.3f} max {x.max():.3f} orig max {o.max():.3f} | "
                  f"rotte {len(hit):3d}/{n} ({100*len(hit)/n:4.1f}%) WPSNR medio {np.mean(hit) if hit else float('nan'):.2f} max {max(hit) if hit else float('nan'):.1f}")
            if t == tau:
                print("       famiglie:", dict(fam.most_common(6)))
