"""
ATTACCANTE ADATTIVO (terza iterazione, 14/10).

La batteria fissa (battery.py) prova sempre gli stessi parametri. Un avversario vero, che ha la
nostra detection, CERCA per ogni immagine l'attacco piu' leggero che toglie il watermark.
Qui simuliamo quell'avversario, usando la detection come oracolo (presente / assente):

  fase 1 - ricerca della forza minima: per ogni famiglia di attacchi permessi (anche catene)
           si cerca il parametro piu' leggero che rompe (scansione grossolana + raffinamento);
  fase 2 - attacco a zone (greedy): partendo dai migliori attacchi globali, si "restituiscono"
           all'immagine watermarkata i blocchi che non servono (prima 64x64, poi 32x32),
           tenendo la modifica solo se il watermark resta assente. Meno danno -> WPSNR piu' alto.

Attacco riuscito: sim < tau e WPSNR >= 35 (p.5). Per ogni immagine si tiene il riuscito con il
WPSNR piu' alto (quello che un avversario caricherebbe).

NOTA REGOLAMENTO: la fase 2 sceglie le zone interrogando la detection (che usa l'originale).
Per valutare la NOSTRA difesa assumiamo il caso peggiore; per i nostri attacchi in gara va
chiarito con i TA se e' ammesso (p.3/p.6: vietato usare l'originale per localizzare).

Difese: le configurazioni di multi.py ('base' = versione 1.0, 'dct_dwt34' = versione 2.0 ibrida),
detection = massimo dei coseni sulle varianti (normale, de-stirate, pesata), ciascuna con la sua tau.

Uso (da embedding/):
  python esperimenti/adaptive.py <config> <tau> [--target 62.05] [--subset A|B|all] [--procs 13] [--limit N] [--tag nome]
Output: esperimenti/risultati/adapt_<tag>.pkl
"""
import argparse, os, pickle, sys, time
from multiprocessing import Pool

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import core, multi                              # noqa: E402
import battery as B                             # noqa: E402
from progress import pmap                       # noqa: E402
from run_battery import all_images, ROOT        # noqa: E402

# famiglie: nome -> (funzione(img, p), griglia grossolana dal piu' LEGGERO al piu' FORTE, tipo di raffinamento)
FAMILIES = {
    "jpeg":        (lambda x, p: B.jpeg(x, p), [95, 90, 80, 70, 60, 50, 40, 30, 25, 20, 16, 13, 10, 8, 6, 5, 4, 3, 2, 1], "int"),
    "blur":        (lambda x, p: B.blur(x, p), [0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.6, 2.0, 2.5, 3.0, 4.0], "float"),
    "awgn":        (lambda x, p: B.awgn(x, p, 7), [2, 4, 6, 8, 10, 13, 16, 20, 25, 30, 40], "float"),
    "median":      (lambda x, p: B.median_nd(x, p), [3, 5, 7, 9, 11, 13], "none"),
    "resize":      (lambda x, p: B.resizing(x, p), [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4,
                                                     0.35, 0.3, 0.25, 0.2, 0.15], "none"),
    "blur0.6+jpeg": (lambda x, p: B.jpeg(B.blur(x, 0.6), p), [90, 70, 50, 40, 30, 25, 20, 16, 13, 10, 8, 6, 5, 4, 3, 2], "int"),
    "blur1.0+jpeg": (lambda x, p: B.jpeg(B.blur(x, 1.0), p), [90, 70, 50, 40, 30, 25, 20, 16, 13, 10, 8, 6, 5, 4, 3, 2], "int"),
    "jpeg+blur0.8": (lambda x, p: B.blur(B.jpeg(x, p), 0.8), [90, 70, 50, 40, 30, 25, 20, 16, 13, 10, 8, 6, 5, 4, 3, 2], "int"),
    "median3+jpeg": (lambda x, p: B.jpeg(B.median_nd(x, 3), p), [90, 70, 50, 40, 30, 25, 20, 16, 13, 10, 8, 6, 5, 4, 3, 2], "int"),
    "awgn3+jpeg":  (lambda x, p: B.jpeg(B.awgn(x, 3, 7), p), [90, 70, 50, 40, 30, 25, 20, 16, 13, 10, 8, 6, 5, 4, 3, 2], "int"),
    "resize0.5+jpeg": (lambda x, p: B.jpeg(B.resizing(x, 0.5), p), [90, 70, 50, 40, 30, 20, 13, 8, 5, 3], "int"),
    "blur+sharp":  (lambda x, p: B.sharpening(B.blur(x, p), 4, 1.5), [0.5, 0.8, 1.0, 1.3, 1.6, 2.0, 2.5, 3.0, 4.0], "float"),
    "awgn+blur":   (lambda x, p: B.blur(B.awgn(x, p, 7), 1.0), [4, 8, 12, 16, 20, 25, 30, 40], "float"),
}
BLOCKS = (64, 32)        # lati dei blocchi della fase 2
TOP_GLOBAL = 3           # quanti attacchi globali raffinare a zone


class Defense:
    """Embedding + detection di una configurazione di multi.py, con la sua soglia."""

    def __init__(self, o, w, config, tau, target=62.05):
        self.o = o
        self.of = o.astype(np.float64)
        self.parts = multi.make_parts(o, multi.CONFIGS[config])
        self.wm, _ = multi.embed(o, w, self.parts, target)
        self.ref = [p(self.wm.astype(np.float64))[None, :] for p, _ in self.parts]
        self.tau = tau
        self.calls = 0

    def sim(self, att):
        self.calls += 1
        V = multi.variants(self.of, att)
        ys = [np.stack([p(v) for v in V]) for p, _ in self.parts]
        return float(multi.scores(self.ref, ys)[0][0])

    def broken(self, att):
        """(rotto?, wpsnr, sim). Rotto = sim < tau e WPSNR >= 35."""
        q = core.wpsnr_fast(self.wm, att)
        if q < 35:
            return False, q, None
        s = self.sim(att)
        return s < self.tau, q, s


def search_family(D, name):
    """Fase 1: parametro piu' leggero che rompe. Restituisce lista di (wpsnr, descrizione, immagine)."""
    f, grid, kind = FAMILIES[name]
    found, prev_ok = [], None
    for p in grid:
        att = f(D.wm, p)
        ok, q, _ = D.broken(att)
        if q < 35:
            break                                   # da qui in poi solo piu' forte: inutile
        if ok:
            found.append((q, f"{name} {p}", att))
            # raffinamento tra il parametro precedente (non rompe) e questo (rompe)
            if prev_ok is not None and kind != "none":
                lo, hi = prev_ok, p                 # lo = piu' leggero (non rompe), hi = rompe
                for _ in range(5 if kind == "float" else 7):
                    if kind == "int":
                        mid = (lo + hi) // 2 if lo > hi else (lo + hi + 1) // 2
                        if mid in (lo, hi):
                            break
                    else:
                        mid = (lo + hi) / 2
                    a2 = f(D.wm, mid)
                    ok2, q2, _ = D.broken(a2)
                    if ok2:
                        hi = mid
                        found.append((q2, f"{name} {mid:.3g}" if kind == "float" else f"{name} {mid}", a2))
                    else:
                        lo = mid
            break
        prev_ok = p
    return found


def localize_greedy(D, att, desc):
    """Fase 2: restituisce alla watermarkata i blocchi che non servono a rompere."""
    cur = att.copy()
    for b in BLOCKS:
        n = 512 // b
        # prima i blocchi dove l'attacco ha fatto piu' danno (restituirli fa guadagnare di piu')
        err = ((cur.astype(float) - D.wm) ** 2).reshape(n, b, n, b).sum(axis=(1, 3))
        order = np.argsort(-err, axis=None)
        for idx in order:
            i, j = divmod(int(idx), n)
            if err[i, j] == 0:
                continue
            cand = cur.copy()
            cand[i * b:(i + 1) * b, j * b:(j + 1) * b] = D.wm[i * b:(i + 1) * b, j * b:(j + 1) * b]
            ok, _, _ = D.broken(cand)
            if ok:
                cur = cand
    q = core.wpsnr_fast(D.wm, cur)
    return q, f"{desc} localizzato (blocchi {BLOCKS})", cur


def run_one(args):
    path, config, tau, target = args
    o = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    w = 2.0 * np.load(os.path.join(ROOT, "fakebitinvestigation.npy")).astype(float) - 1
    D = Defense(o, w, config, tau, target)
    t0 = time.time()
    glob_found = []
    per_family = {}
    for name in FAMILIES:
        r = search_family(D, name)
        glob_found += r
        per_family[name] = max((x[0] for x in r), default=None)
    calls_glob = D.calls
    glob_found.sort(key=lambda x: -x[0])
    best_glob = glob_found[0][:2] if glob_found else None
    loc = []
    # per ogni famiglia diversa, il miglior attacco globale; poi a zone
    seen = set()
    for q, desc, att in glob_found:
        fam = desc.rsplit(" ", 1)[0]
        if fam in seen:
            continue
        seen.add(fam)
        loc.append(localize_greedy(D, att, desc)[:2])
        if len(seen) >= TOP_GLOBAL:
            break
    best_all = max([best_glob] + loc, key=lambda x: x[0]) if best_glob else None
    return dict(img=os.path.splitext(os.path.basename(path))[0], best_global=best_glob, best_local=loc,
                best=best_all, per_family=per_family, calls=(calls_glob, D.calls), secs=time.time() - t0)


def rob_points(m):
    for thr, p in [(53, 0), (50, 1), (47, 2), (44, 3), (41, 4), (38, 5), (35, 6)]:
        if m >= thr:
            return p
    return 6


def report(d):
    res = d["res"]
    n = len(res)
    for key, lab in (("best_global", "solo attacchi globali (fase 1)"), ("best", "globali + a zone (fase 2)")):
        hit = [r[key][0] for r in res if r[key]]
        m = np.mean(hit) if hit else float("nan")
        print(f"  {lab:32s} rotte {len(hit)}/{n} ({100 * len(hit) / n:.0f}%)  WPSNR medio {m:.2f}  "
              f"mediana {np.median(hit) if hit else float('nan'):.2f}  max {max(hit) if hit else float('nan'):.1f}  "
              f"-> punti robustezza {rob_points(m) if hit else 'mai rotte'}")
    fam = {}
    for r in res:
        if r["best"]:
            k = r["best"][1].split(" ")[0] + (" (zone)" if "localizzato" in r["best"][1] else "")
            fam[k] = fam.get(k, 0) + 1
    print("  miglior attacco per immagine:", dict(sorted(fam.items(), key=lambda x: -x[1])))
    print(f"  chiamate alla detection per immagine: media {np.mean([r['calls'][1] for r in res]):.0f}, "
          f"tempo medio {np.mean([r['secs'] for r in res]):.0f} s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("tau", type=float)
    ap.add_argument("--subset", default="all", choices=["all", "A", "B"])
    ap.add_argument("--procs", type=int, default=13)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--target", type=float, default=62.05, help="WPSNR dell'embedding (fine-tuning: 58.05 / 62.05 / 66.05)")
    a = ap.parse_args()
    imgs = all_images()
    if a.subset != "all":
        from analyze import split
        A, Bset = split([os.path.splitext(os.path.basename(i))[0] for i in imgs])
        keep = A if a.subset == "A" else Bset
        imgs = [i for i in imgs if os.path.splitext(os.path.basename(i))[0] in keep]
    if a.limit:
        imgs = imgs[:a.limit]
    tag = a.tag or f"{a.config}_{a.subset}"
    t = time.time()
    with Pool(a.procs) as p:
        res = pmap(p, run_one, [(im, a.config, a.tau, a.target) for im in imgs], "adattivo " + tag)
    d = dict(config=a.config, tau=a.tau, target=a.target, res=res)
    pickle.dump(d, open(os.path.join(HERE, "risultati", f"adapt_{tag}.pkl"), "wb"))
    print(f"== attaccante adattivo contro {a.config} a {a.target} dB (tau {a.tau}), {len(imgs)} immagini, {time.time() - t:.0f} s")
    report(d)


if __name__ == "__main__":
    main()
