"""
Analisi dei risultati di run_battery.py.
Attacco riuscito (p.5): sim < tau  e  WPSNR >= 35. Per ogni immagine teniamo il riuscito
con WPSNR piu' alto (quello che un attaccante caricherebbe).

Split fisso: le 104 immagini sono divise a caso (seme 2026) in A (52, per decidere)
e B (52, per verificare che le decisioni non siano overfitting).

Uso: python analyze.py <tag> [detector] [tau ...]
"""
import collections, os, pickle, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load(tag):
    with open(os.path.join(HERE, "risultati", tag + ".pkl"), "rb") as f:
        return pickle.load(f)


def split(names):
    names = sorted(names)
    perm = np.random.default_rng(2026).permutation(len(names))
    A = {names[i] for i in perm[: len(names) // 2]}
    return A, set(names) - A


def simval(row, det):
    """det puo' essere un nome o 'max(a,b,...)' = massimo tra piu' detector."""
    if det.startswith("max(") and det.endswith(")"):
        return max(row["sim"][d] for d in det[4:-1].split("|"))
    return row["sim"][det]


def best_attacks(data, det, tau, exclude_families=()):
    out = {}
    for r in data["res"]:
        best = None
        for row in r["rows"]:
            if row["family"] in exclude_families:
                continue
            if simval(row, det) < tau and row["wpsnr"] >= 35 and (best is None or row["wpsnr"] > best["wpsnr"]):
                best = row
        out[r["img"]] = best
    return out


def summary(best, subset):
    b = [best[i] for i in subset]
    hit = [x["wpsnr"] for x in b if x is not None]
    s = f"rotte {len(hit):3d}/{len(b)} ({100 * len(hit) / len(b):4.1f}%)"
    if hit:
        s += f"  WPSNR attacchi riusciti: media {np.mean(hit):5.2f}  mediana {np.median(hit):5.2f}  max {np.max(hit):5.2f}"
    return s


def report(tag, det="cos", taus=(0.08, 0.10, 0.12)):
    d = load(tag)
    names = [r["img"] for r in d["res"]]
    A, B = split(names)
    print(f"== {tag}  cfg={d['cfg']}  detector={det}")
    q = [r["q_emb"] for r in d["res"]]
    print(f"   WPSNR embedding: min {min(q):.4f} max {max(q):.4f}")
    for tau in taus:
        best = best_attacks(d, det, tau)
        print(f" tau={tau:.2f} TUTTE {summary(best, names)}")
        print(f"          A    {summary(best, A)}")
        print(f"          B    {summary(best, B)}")
    best = best_attacks(d, det, 0.10)
    fam = collections.Counter(x["family"] for x in best.values() if x)
    print("  famiglia del miglior attacco (tau 0.10):", dict(fam.most_common()))
    return d


if __name__ == "__main__":
    tag = sys.argv[1]
    det = sys.argv[2] if len(sys.argv) > 2 else "cos"
    taus = tuple(float(x) for x in sys.argv[3:]) or (0.08, 0.10, 0.12)
    report(tag, det, taus)
