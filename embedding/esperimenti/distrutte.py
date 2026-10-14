"""
Due controlli sulla detection (14/10, regolamento aggiornato):

1) IMMAGINI DISTRUTTE: la detection non puo' piu' forzare output1 col WPSNR, quindi misuriamo quante
   volte dice "presente" su attaccate con WPSNR <= 25 dB, per ciascun detector (il massimo dei
   detector indicati con --dets, come farebbe la detection).
2) TEMPI: tempo di una chiamata al detector combinato, in un solo processo (da lanciare a CPU libera).

Uso (da embedding/): python esperimenti/distrutte.py <tau> [--dets cos,rs_cos,sq:16:0.5,pr,prsq:16:0.5] [--step 8]
"""
import argparse, os, sys, time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import core, detectors, battery as B          # noqa: E402
from run_battery import all_images, ROOT       # noqa: E402

DESTROY = {}
for s in (60, 80, 120):
    DESTROY[f"awgn {s}"] = lambda x, s=s: B.awgn(x, s, 3)
for q in (1, 2):
    DESTROY[f"jpeg {q} + blur 3"] = lambda x, q=q: B.blur(B.jpeg(x, q), 3)
DESTROY["jpeg 1 + awgn 40"] = lambda x: B.awgn(B.jpeg(x, 1), 40, 3)
for s in (5, 8):
    DESTROY[f"blur {s}"] = lambda x, s=s: B.blur(x, s)
for k in (15, 21):
    DESTROY[f"median {k}"] = lambda x, k=k: B.median_nd(x, k)
DESTROY["resize 0.1"] = lambda x: B.resizing(x, 0.1)
DESTROY["sharp 10,3"] = lambda x: B.sharpening(x, 10, 3)
DESTROY["sharp 4,8"] = lambda x: B.sharpening(x, 4, 8)
DESTROY["nero"] = lambda x: np.zeros_like(x)
DESTROY["bianco"] = lambda x: np.full_like(x, 255)
DESTROY["negativo"] = lambda x: 255 - x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tau", type=float)
    ap.add_argument("--dets", default="cos,rs_cos,sq:16:0.5,pr,prsq:16:0.5")
    ap.add_argument("--step", type=int, default=8, help="una immagine ogni STEP")
    a = ap.parse_args()
    dets = a.dets.split(",")
    w = 2.0 * np.load(os.path.join(ROOT, "fakebitinvestigation.npy")).astype(float) - 1
    imgs = all_images()[::a.step]
    hits = {d: 0 for d in dets + ["max attuale", "max con predizione"]}
    tot, times, per_att = 0, [], {}
    for n_img, p in enumerate(imgs, 1):
        o = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        wm, _ = core.embed(o, w)
        D = detectors.build(o, wm, dict(rmin=32.0, rmax=128.0, dets=dets))
        for name, f in DESTROY.items():
            att = f(wm)
            if core.wpsnr_fast(wm, att) > 25:
                continue
            t = time.perf_counter()
            s = {d: D[d](att) for d in dets}
            times.append(time.perf_counter() - t)
            tot += 1
            cur = max(s.get("rs_cos", 0), s.get("sq:16:0.5", 0))
            new = max([cur] + [s[d] for d in dets if d.startswith("pr")])
            for d in dets:
                hits[d] += s[d] >= a.tau
            hits["max attuale"] += cur >= a.tau
            hits["max con predizione"] += new >= a.tau
            per_att.setdefault(name, []).append((cur, new))
        sys.stderr.write(f"\r[{'#' * (30 * n_img // len(imgs)):30s}] {n_img}/{len(imgs)} immagini ")
    sys.stderr.write("\n")
    print(f"== immagini distrutte (WPSNR <= 25): {tot} casi su {len(imgs)} immagini, tau {a.tau}")
    for d, h in hits.items():
        print(f"   {d:22s} 'presente' {h}/{tot}")
    print("   per attacco (max attuale -> max con predizione):")
    for name, v in per_att.items():
        v = np.array(v)
        print(f"     {name:20s} presente {int((v[:, 0] >= a.tau).sum())}/{len(v)} -> {int((v[:, 1] >= a.tau).sum())}/{len(v)}"
              f"   sim max {v[:, 0].max():.3f} -> {v[:, 1].max():.3f}")
    t = np.array(times)
    print(f"== tempo di una chiamata a tutti i detector ({', '.join(dets)}), un processo: "
          f"media {t.mean():.2f} s, mediana {np.median(t):.2f} s, max {t.max():.2f} s")


if __name__ == "__main__":
    main()
