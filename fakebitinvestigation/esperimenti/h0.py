"""
Falsi positivi (ipotesi H0) per ciascun detector.
Per ogni immagine: K watermark casuali +-1 (un "altro gruppo"), ognuno con un attacco casuale,
e anche l'originale non watermarkata con un attacco casuale. La detection riceve
(originale, NOSTRA watermarkata, immagine H0) e non deve dire "presente".

Uso: python h0.py <tag> <dets separati da virgola> [K] [gamma]
Output: risultati/h0_<tag>.pkl = {det: array di similarita'} + casi
"""
import os, pickle, sys
from multiprocessing import Pool
import cv2, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import core, detectors, battery as B          # noqa: E402
from progress import pmap                    # noqa: E402
from run_battery import all_images, ROOT       # noqa: E402


DEMOD = os.environ.get("DEMOD", "0") == "1"   # detection demodulata (solo con gamma > 0)


def random_attack(img, rng):
    k = rng.integers(9)
    if k == 0: return B.jpeg(img, int(rng.integers(3, 90)))
    if k == 1: return B.awgn(img, rng.uniform(2, 30), int(rng.integers(1e6)))
    if k == 2: return B.blur(img, rng.uniform(0.3, 3))
    if k == 3: return B.sharpening(img, rng.choice([0.5, 1, 2, 4, 10]), rng.uniform(0.2, 3))
    if k == 4: return B.median_nd(img, int(rng.choice([3, 5, 7, 9])))
    if k == 5: return B.resizing(img, round(rng.uniform(0.2, 0.95), 2))
    if k == 6: return B.blur(B.awgn(img, rng.uniform(5, 30), int(rng.integers(1e6))), rng.uniform(0.5, 2))
    if k == 7:
        m = list(B.masks(img).values())[int(rng.integers(8))]
        return B.localize(img, B.jpeg(img, int(rng.integers(2, 20))), m)
    return img.copy()                                   # nessun attacco


def run_one(args):
    path, dets, K, seed, gamma = args
    rng = np.random.default_rng(seed)
    o = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    w = 2.0 * np.load(os.path.join(ROOT, "fakebitinvestigation.npy")).astype(float) - 1
    wm, _ = core.embed(o, w, gamma=gamma)
    D = detectors.build(o, wm, dict(rmin=32.0, rmax=128.0, dets=dets, demod=gamma if DEMOD else 0.0))
    out = {"altro_wm": {d: [] for d in dets}, "orig_attaccata": {d: [] for d in dets},
           "orig": {d: D[d](o) for d in dets}}
    for _ in range(K):
        w2 = rng.choice([-1.0, 1.0], 1024)
        other, _ = core.embed(o, w2, gamma=gamma)
        h = random_attack(other, rng)
        if core.wpsnr_fast(wm, h) > 25:
            for d in dets: out["altro_wm"][d].append(D[d](h))
        h = random_attack(o, rng)
        if core.wpsnr_fast(wm, h) > 25:
            for d in dets: out["orig_attaccata"][d].append(D[d](h))
    return out


def main():
    tag, dets = sys.argv[1], sys.argv[2].split(",")
    K = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    gamma = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
    imgs = all_images()
    with Pool(13) as p:
        res = pmap(p, run_one, [(im, dets, K, 1000 + i, gamma) for i, im in enumerate(imgs)], "h0 " + tag)
    agg = {}
    for case in ("altro_wm", "orig_attaccata"):
        agg[case] = {d: np.concatenate([r[case][d] for r in res]) for d in dets}
    agg["orig"] = {d: np.array([r["orig"][d] for r in res]) for d in dets}
    pickle.dump(agg, open(os.path.join(HERE, "risultati", f"h0_{tag}.pkl"), "wb"))
    for case in ("altro_wm", "orig_attaccata", "orig"):
        print(f"--- {case}")
        for d in dets:
            x = agg[case][d]
            print(f"  {d:16s} n={len(x):5d} media={x.mean():+.4f} std={x.std():.4f} "
                  f"q99.9={np.quantile(x, .999):+.3f} max={x.max():+.3f} "
                  f"P(>=0.08)={np.mean(x >= .08):.4f} P(>=0.10)={np.mean(x >= .10):.4f}")


if __name__ == "__main__":
    main()
