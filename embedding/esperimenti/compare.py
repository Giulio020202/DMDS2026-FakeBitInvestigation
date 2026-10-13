"""Tabella di confronto tra varianti. Uso: python compare.py <detector> tag1 tag2 ..."""
import os, sys
import numpy as np
from analyze import load, best_attacks


def rob_points(m):
    for thr, p in [(53, 0), (50, 1), (47, 2), (44, 3), (41, 4), (38, 5), (35, 6)]:
        if m >= thr:
            return p
    return 6


det = sys.argv[1]
print(f"detector: {det}")
print(f"{'variante':14s} | {'tau':>4s} | {'rotte':>11s} | {'WPSNR medio':>11s} | {'mediana':>7s} | {'max':>5s} | punti rob.")
for tag in sys.argv[2:]:
    if not os.path.exists(os.path.join(os.path.dirname(__file__), "risultati", tag + ".pkl")):
        print(f"{tag:14s} | (manca)"); continue
    d = load(tag)
    for tau in (0.10, 0.12):
        b = best_attacks(d, det, tau)
        hit = [x["wpsnr"] for x in b.values() if x]
        n = len(b)
        if hit:
            print(f"{tag:14s} | {tau:.2f} | {len(hit):3d}/{n} {100*len(hit)/n:4.0f}% | {np.mean(hit):11.2f} | {np.median(hit):7.2f} | {max(hit):5.1f} | {rob_points(np.mean(hit))}")
        else:
            print(f"{tag:14s} | {tau:.2f} | {0:3d}/{n}    0% |           - |       - |     - | mai rotte")
