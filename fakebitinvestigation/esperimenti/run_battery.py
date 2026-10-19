"""
Fa girare la batteria v2 su un insieme di immagini e salva, per ogni attacco,
WPSNR(watermarked, attaccata) e la similarita' di ciascun detector.

Uso (da fakebitinvestigation/):
  python esperimenti/run_battery.py --tag base --target 62.05 [--rmin 32 --rmax 128] [--procs 12]
Output: esperimenti/risultati/<tag>.pkl  = lista di dict
  {img, family, attack, wpsnr, sim: {detector: valore}}
"""
import argparse, glob, os, pickle, sys, time
from multiprocessing import Pool

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import core                      # noqa: E402
from progress import pmap                    # noqa: E402
import detectors                 # noqa: E402
from battery import battery, battery_v3      # noqa: E402

ROOT = os.path.dirname(HERE)


def all_images():
    return sorted(glob.glob(os.path.join(ROOT, "sample-images", "*.bmp"))) + \
        [os.path.join(ROOT, "gara2022", n) for n in ("buildings.bmp", "rollercoaster.bmp", "tree.bmp")]


def run_one(args):
    path, cfg = args
    o = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    w = 2.0 * np.load(os.path.join(ROOT, "fakebitinvestigation.npy")).astype(float) - 1
    if cfg.get("transform", "dct") == "dwt":
        import core_dwt
        wm, a = core_dwt.embed(o, w, cfg["target"])
    else:
        wm, a = core.embed(o, w, cfg["target"], cfg["rmin"], cfg["rmax"], gamma=cfg.get("gamma", 0.0))
    dets = detectors.build(o, wm, cfg)
    name = os.path.splitext(os.path.basename(path))[0]
    rows = []
    for fam, an, att in (battery_v3(wm) if cfg.get('v3') else battery(wm)):
        q = core.wpsnr_fast(wm, att)
        rows.append(dict(img=name, family=fam, attack=an, wpsnr=q,
                         sim={k: d(att) for k, d in dets.items()}))
    return dict(img=name, amp=a, q_emb=core.wpsnr_fast(o, wm), rows=rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--target", type=float, default=62.05)
    ap.add_argument("--rmin", type=float, default=32.0)
    ap.add_argument("--rmax", type=float, default=128.0)
    ap.add_argument("--procs", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dets", default="cos")
    ap.add_argument("--gamma", type=float, default=0.0)
    ap.add_argument("--demod", type=float, default=0.0)
    ap.add_argument("--transform", default="dct", choices=["dct", "dwt"])
    ap.add_argument("--subset", default="all", choices=["all", "A", "B"])
    ap.add_argument("--v3", action="store_true", help="batteria v3 = v2 + attacchi anti-predizione")
    ap.add_argument("--folder", default="", help="cartella di BMP alternativa (es. esperimenti/set_c)")
    a = ap.parse_args()
    cfg = dict(target=a.target, rmin=a.rmin, rmax=a.rmax, gamma=a.gamma, transform=a.transform, demod=a.demod, dets=a.dets.split(","), v3=a.v3)
    imgs = sorted(glob.glob(os.path.join(a.folder, "*.bmp"))) if a.folder else all_images()
    if a.subset != "all":
        from analyze import split
        A, B = split([os.path.splitext(os.path.basename(i))[0] for i in imgs])
        keep = A if a.subset == "A" else B
        imgs = [i for i in imgs if os.path.splitext(os.path.basename(i))[0] in keep]
    if a.limit:
        imgs = imgs[:a.limit]
    t = time.time()
    with Pool(a.procs) as p:
        res = pmap(p, run_one, [(i, cfg) for i in imgs], a.tag)
    out = os.path.join(HERE, "risultati", a.tag + ".pkl")
    with open(out, "wb") as f:
        pickle.dump(dict(cfg=cfg, res=res), f)
    print(f"{len(imgs)} immagini, {len(res[0]['rows'])} attacchi ciascuna, {time.time() - t:.0f}s -> {out}")


if __name__ == "__main__":
    main()
