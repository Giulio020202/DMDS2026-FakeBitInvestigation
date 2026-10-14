"""
Terza iterazione (14/10): embedding MULTI-PARTE (multi-banda / ibrido DCT + wavelet).

Ogni "parte" e' uno schema a coppie differenziali gia' noto (DCT globale in una banda radiale,
oppure DWT db4 in alcuni livelli). Lo STESSO watermark w va in tutte le parti, con ampiezze
relative k_p fissate; un'unica scala s viene calibrata con la bisezione al WPSNR target
(stesso WPSNR del lab, versione FFT).

Detection: per ogni parte si estrae il vettore come nella detection attuale (normale, 4 de-stirate,
pesatura spaziale sq 16/0.5). Due modi di decidere:
  concat : coseno sul vettore concatenato di tutte le parti (massimo sulle varianti);
  maxpart: massimo tra concat e il coseno di ogni singola parte.
H0 come nella ROC del regolamento (p.4): W_random +-1 confrontato con i vettori estratti
dall'attaccata, con la stessa funzione di similarita' -> tau a FPR 0.1% per OGNI configurazione
(confronto a parita' di falsi positivi, non a parita' di tau).

Uso (da embedding/): python esperimenti/multi.py <config> [--subset A] [--procs 13]
"""
import argparse, os, pickle, sys, time
from multiprocessing import Pool

import cv2
import numpy as np
from scipy.ndimage import uniform_filter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import core, core_dwt                         # noqa: E402
from progress import pmap                    # noqa: E402
from battery import battery                   # noqa: E402
from detectors import destretch, STRETCH      # noqa: E402
from run_battery import all_images, ROOT      # noqa: E402

# nome -> lista di parti: ("dct", rmin, rmax, k) oppure ("dwt", livelli, k)
CONFIGS = {
    "base":      [("dct", 32, 128, 1.0)],
    "hi3":       [("dct", 32, 128, 1.0), ("dct", 200, 320, 3.0)],
    "hi6":       [("dct", 32, 128, 1.0), ("dct", 200, 320, 6.0)],
    "lo_mid":    [("dct", 8, 56, 1.0), ("dct", 56, 140, 1.0)],
    "dct_dwt34": [("dct", 32, 128, 1.0), ("dwt", (3, 4), 1.0)],
    "dct_dwt2":  [("dct", 32, 128, 1.0), ("dwt", (2,), 3.0)],
}
N_RANDOM = 10
TAU_FIXED = 0.11


class DctPart:
    def __init__(self, o, rmin, rmax):
        self.ex = core.Extractor(o, rmin, rmax)
        self.c = self.ex.c0

    def pattern(self, w):
        ex = self.ex
        P = np.zeros_like(self.c)
        P[ex.uu, ex.vv] = ex.s1 * w
        P[ex.uu, ex.vv + 1] = -ex.s2 * w
        return core.idct2(P)

    def __call__(self, img):
        return self.ex(img)


class DwtPart:
    def __init__(self, o, levels):
        self.ex = core_dwt.Extractor(o, levels)
        self.arr, self.sl = self.ex.c0, self.ex.sl

    def pattern(self, w):
        ex = self.ex
        P = np.zeros_like(self.arr)
        s1 = np.where(ex.a1 < 0, -1.0, 1.0)
        s2 = np.where(ex.a2 < 0, -1.0, 1.0)
        P[ex.ii, ex.jj] = s1 * w
        P[ex.ii, ex.jj + 1] = -s2 * w
        return core_dwt.inv(P, self.sl)

    def __call__(self, img):
        return self.ex(img)


def make_parts(o, cfg):
    out = []
    for p in cfg:
        if p[0] == "dct":
            out.append((DctPart(o, p[1], p[2]), p[3]))
        else:
            out.append((DwtPart(o, p[1]), p[2]))
    return out


def embed(o, w, parts, target=62.05):
    pat = sum(k * part.pattern(w) for part, k in parts)
    base = o.astype(np.float64)

    def mark(s):
        return np.clip(np.rint(base + s * pat), 0, 255).astype(np.uint8)

    lo, hi = 0.2, 40.0
    best, best_s = mark(lo), lo
    for _ in range(14):
        mid = np.sqrt(lo * hi)
        cand = mark(mid)
        if core.wpsnr_fast(o, cand) >= target:
            lo, best, best_s = mid, cand, mid
        else:
            hi = mid
    return best, best_s


def variants(o, att):
    """Le immagini da cui estrarre (come la detection attuale): normale, 4 de-stirate, pesata."""
    a = att.astype(np.float64)
    out = [a] + [destretch(att, m).astype(np.float64) for m in STRETCH]
    d = a - o
    rho = 1.0 / (uniform_filter(d * d, 16) + 0.5)
    out.append(o + rho * d)
    return out


def cos_rows(R, Y):
    """R: (n_ref, N) riferimenti, Y: (n_var, N) vettori estratti -> (n_ref, n_var) coseni."""
    nr = np.linalg.norm(R, axis=1, keepdims=True)
    ny = np.linalg.norm(Y, axis=1, keepdims=True)
    nr[nr == 0] = np.inf
    ny[ny == 0] = np.inf
    return (R / nr) @ (Y / ny).T


def scores(refs, ys):
    """refs: lista per parte di (n_ref, 1024); ys: lista per parte di (n_var, 1024).
    Restituisce (concat, maxpart, norm) per ogni riferimento.
    norm = SOGLIA NORMALIZZATA (Lab 4): per ogni variante <x,y> / sqrt(sum (x_i y_i)^2), cioe' la
    similarita' divisa per la sua deviazione standard stimata su QUESTA immagine e QUESTO attacco
    (sotto H0 vale circa N(0,1) qualunque sia l'immagine); poi massimo sulle varianti."""
    R = np.concatenate(refs, axis=1)
    Y = np.concatenate(ys, axis=1)
    concat = cos_rows(R, Y).max(axis=1)
    mp = concat.copy()
    if len(refs) > 1:
        for r, y in zip(refs, ys):
            mp = np.maximum(mp, cos_rows(r, y).max(axis=1))
    num = R @ Y.T
    den = np.sqrt((R * R) @ (Y * Y).T)
    den[den == 0] = np.inf
    norm = (num / den).max(axis=1)
    return concat, mp, norm


def run_one(args):
    path, name, seed = args
    rng = np.random.default_rng(seed)
    o = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    of = o.astype(np.float64)
    w = 2.0 * np.load(os.path.join(ROOT, "fakebitinvestigation.npy")).astype(float) - 1
    parts = make_parts(o, CONFIGS[name])
    wm, s = embed(o, w, parts)
    wext = [p(wm.astype(np.float64)) for p, _ in parts]
    norms = [np.linalg.norm(x) / np.sqrt(x.size) for x in wext]
    rows = []
    for fam, an, att in battery(wm):
        q = core.wpsnr_fast(wm, att)
        V = variants(of, att)
        ys = [np.stack([p(v) for v in V]) for p, _ in parts]
        Wr = rng.choice([-1.0, 1.0], (N_RANDOM, 1024))
        refs = [np.vstack([x[None, :], Wr * nm]) for x, nm in zip(wext, norms)]
        c, m, z = scores(refs, ys)
        rows.append(dict(family=fam, attack=an, wpsnr=q, concat=c[0], maxpart=m[0], norm=z[0],
                         h0_concat=c[1:], h0_maxpart=m[1:], h0_norm=z[1:]))
    # originale come attaccata: deve dare "assente"
    V = variants(of, o)
    ys = [np.stack([p(v) for v in V]) for p, _ in parts]
    c, m, z = scores([x[None, :] for x in wext], ys)
    return dict(img=os.path.splitext(os.path.basename(path))[0], scale=s,
                q_emb=core.wpsnr_fast(o, wm), rows=rows, orig=(c[0], m[0], z[0]))


def analyze(d):
    modes = ("concat", "maxpart", "norm") if "norm" in d["res"][0]["rows"][0] else ("concat", "maxpart")
    for mode in modes:
        h0 = np.concatenate([[x for row in r["rows"] for x in row["h0_" + mode]] for r in d["res"]])
        tau = float(np.quantile(h0, 0.999))
        print(f"  [{mode}] H0 n={h0.size} media {h0.mean():+.4f} std {h0.std():.4f} q99.9 {tau:.4f} max {h0.max():.3f}; "
              f"originale max {max(r['orig'][modes.index(mode)] for r in d['res']):.3f}")
        for t, lab in ((tau, "tau FPR0.1%"),) if mode == "norm" else ((tau, "tau FPR0.1%"), (TAU_FIXED, "tau 0.11")):
            hit = []
            for r in d["res"]:
                b = [row["wpsnr"] for row in r["rows"] if row[mode] < t and row["wpsnr"] >= 35]
                if b:
                    hit.append(max(b))
            n = len(d["res"])
            print(f"     {lab:12s} {t:.4f}: rotte {len(hit)}/{n} ({100 * len(hit) / n:.0f}%)  "
                  f"WPSNR medio {np.mean(hit) if hit else float('nan'):.2f}  max {max(hit) if hit else float('nan'):.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--subset", default="A", choices=["all", "A", "B"])
    ap.add_argument("--procs", type=int, default=13)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    imgs = all_images()
    if a.subset != "all":
        from analyze import split
        A, B = split([os.path.splitext(os.path.basename(i))[0] for i in imgs])
        keep = A if a.subset == "A" else B
        imgs = [i for i in imgs if os.path.splitext(os.path.basename(i))[0] in keep]
    if a.limit:
        imgs = imgs[:a.limit]
    t = time.time()
    with Pool(a.procs) as p:
        res = pmap(p, run_one, [(im, a.config, 5000 + i) for i, im in enumerate(imgs)], a.config)
    d = dict(config=a.config, parts=CONFIGS[a.config], res=res)
    out = os.path.join(HERE, "risultati", f"multi_{a.config}_{a.subset}.pkl")
    pickle.dump(d, open(out, "wb"))
    q = [r["q_emb"] for r in res]
    print(f"== {a.config} {CONFIGS[a.config]}  {len(imgs)} img, {time.time() - t:.0f}s, WPSNR emb {min(q):.3f}-{max(q):.3f}")
    analyze(d)


if __name__ == "__main__":
    main()
