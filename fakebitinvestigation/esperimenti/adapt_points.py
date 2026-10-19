"""
Punti attesi della difesa dai risultati dell'attaccante adattivo (risultati/adapt_<tag>.pkl).

Regole (regolamento + risposte del TA del 14/10):
  - qualita': media del WPSNR delle 3 immagini (qui tutte allo stesso target) -> 62.05 = 5 punti, 58.05 = 4;
  - robustezza: WPSNR medio degli attacchi riusciti sulle nostre immagini (tabella p.7);
  - mai rotti: 6 punti robustezza + 2 di bonus.
Simulazione: 3 immagini estratte a caso dalle 104 (N volte), per ogni immagine il miglior attacco
riuscito (quello che un avversario caricherebbe). Incertezza: bootstrap sulle immagini.

Uso (da fakebitinvestigation/esperimenti): python adapt_points.py v1 v2 v1_t58 v2_t58 [--key best|best_global]
"""
import argparse, pickle
import numpy as np
from scipy.stats import binomtest

QUALITY = {62.05: 5, 58.05: 4, 66.05: 6}


def rob_points(m):
    for thr, p in [(53, 0), (50, 1), (47, 2), (44, 3), (41, 4), (38, 5), (35, 6)]:
        if m >= thr:
            return p
    return 6


def load(tag, key):
    d = pickle.load(open(f"risultati/adapt_{tag}.pkl", "rb"))
    best = {r["img"]: (r[key][0] if r[key] else None) for r in d["res"]}
    return d.get("target", 62.05), best


def expected(best, imgs, target, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    q = QUALITY[target]
    v = [best[i] for i in imgs]
    tot = np.empty(n)
    for k in range(n):
        s = [v[j] for j in rng.choice(len(v), 3, replace=False)]
        hit = [x for x in s if x is not None]
        tot[k] = q + (8 if not hit else rob_points(np.mean(hit)))
    return tot.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--key", default="best", choices=["best", "best_global"])
    ap.add_argument("--boot", type=int, default=200)
    a = ap.parse_args()
    data = {t: load(t, a.key) for t in a.tags}
    imgs = sorted(next(iter(data.values()))[1])
    print(f"attacchi considerati: {'globali + a zone' if a.key == 'best' else 'solo globali'}")
    pts = {}
    for t, (target, best) in data.items():
        hit = [x for x in best.values() if x is not None]
        pts[t] = expected(best, imgs, target)
        print(f"  {t:8s} target {target}: rotte {len(hit)}/{len(best)} ({100 * len(hit) / len(best):.0f}%), "
              f"WPSNR medio {np.mean(hit):.2f}, punti attesi {pts[t]:.2f}")
    # confronti a coppie: stesso target, prima contro seconda configurazione
    rng = np.random.default_rng(1)
    for t1, t2 in [(a.tags[i], a.tags[i + 1]) for i in range(0, len(a.tags) - 1, 2)]:
        b1, b2 = data[t1][1], data[t2][1]
        only1 = sum(1 for i in imgs if b1[i] is not None and b2[i] is None)
        only2 = sum(1 for i in imgs if b2[i] is not None and b1[i] is None)
        p = binomtest(only1, only1 + only2).pvalue if only1 + only2 else 1.0
        diffs = []
        for _ in range(a.boot):
            s = list(rng.choice(imgs, len(imgs), replace=True))
            bb1 = {f"{i}#{k}": b1[i] for k, i in enumerate(s)}      # nomi ripetuti resi distinti
            bb2 = {f"{i}#{k}": b2[i] for k, i in enumerate(s)}
            ks = list(bb1)
            diffs.append(expected(bb2, ks, data[t2][0], 2000, 7) - expected(bb1, ks, data[t1][0], 2000, 7))
        lo, hi = np.percentile(diffs, [5, 95])
        print(f"  {t2} - {t1}: punti attesi {pts[t2] - pts[t1]:+.2f} (intervallo 90% bootstrap {lo:+.2f} .. {hi:+.2f}); "
              f"rotte solo da {t1}: {only1}, solo da {t2}: {only2} (test del segno p={p:.2f})")


if __name__ == "__main__":
    main()
