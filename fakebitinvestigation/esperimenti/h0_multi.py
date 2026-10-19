"""
Falsi positivi REALISTICI per gli embedding multi-parte (multi.py), come h0.py:
per ogni immagine K casi "altro gruppo" (stesso schema, watermark casuale, attacco casuale),
K casi "originale + attacco casuale" e l'originale esatta. Detector: coseno concatenato
(massimo sulle varianti normale / de-stirate / pesata), come in multi.py.

Uso (da fakebitinvestigation/): python esperimenti/h0_multi.py <config> [K]
"""
import os, pickle, sys
from multiprocessing import Pool

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import core, multi                              # noqa: E402
from progress import pmap                    # noqa: E402
from h0 import random_attack                    # noqa: E402
from run_battery import all_images, ROOT        # noqa: E402


def run_one(args):
    path, name, K, seed = args
    rng = np.random.default_rng(seed)
    o = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    of = o.astype(np.float64)
    w = 2.0 * np.load(os.path.join(ROOT, "fakebitinvestigation.npy")).astype(float) - 1
    parts = multi.make_parts(o, multi.CONFIGS[name])
    wm, _ = multi.embed(o, w, parts)
    ref = [p(wm.astype(np.float64))[None, :] for p, _ in parts]

    def sim(img):
        """(coseno concatenato, punteggio normalizzato)"""
        V = multi.variants(of, img)
        ys = [np.stack([p(v) for v in V]) for p, _ in parts]
        c, _, z = multi.scores(ref, ys)
        return float(c[0]), float(z[0])

    out = {"altro_wm": [], "orig_attaccata": [], "orig": [sim(o)]}
    for _ in range(K):
        other, _ = multi.embed(o, rng.choice([-1.0, 1.0], 1024), parts)
        h = random_attack(other, rng)
        if core.wpsnr_fast(wm, h) > 25:
            out["altro_wm"].append(sim(h))
        h = random_attack(o, rng)
        if core.wpsnr_fast(wm, h) > 25:
            out["orig_attaccata"].append(sim(h))
    return out


def main():
    name = sys.argv[1]
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    with Pool(13) as p:
        res = pmap(p, run_one, [(im, name, K, 1000 + i) for i, im in enumerate(all_images())], "h0 " + name)
    # agg[caso] = array (n, 2): colonna 0 coseno concatenato, colonna 1 punteggio normalizzato
    agg = {c: np.array([x for r in res for x in r[c]]).reshape(-1, 2) for c in ("altro_wm", "orig_attaccata", "orig")}
    pickle.dump(agg, open(os.path.join(HERE, "risultati", f"h0multi_{name}.pkl"), "wb"))
    for col, lab in ((0, "coseno"), (1, "normalizzato")):
        allx = np.concatenate([agg["altro_wm"][:, col], agg["orig_attaccata"][:, col]])
        print(f"== H0 realistica {name} [{lab}]: tau (q99.9 di altro_wm + orig_attaccata) = {np.quantile(allx, .999):.4f}")
        for c in ("altro_wm", "orig_attaccata", "orig"):
            x = agg[c][:, col]
            print(f"  {c:15s} n={len(x):5d} media {x.mean():+.4f} std {x.std():.4f} q99.9 {np.quantile(x, .999):+.3f} max {x.max():+.3f}")


if __name__ == "__main__":
    main()
