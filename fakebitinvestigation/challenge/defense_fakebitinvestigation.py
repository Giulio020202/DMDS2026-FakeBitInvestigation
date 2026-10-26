"""
Capture the Mark 2026 - HELPER for the DEFENSE PHASE of the challenge day (group fakebitinvestigation). NOT submitted.

It contains no embedding or attack logic: it uses embedding(), detection() and attack_image() of the
submitted files (../submission/) and changes at most the parameter TARGET_WPSNR (allowed fine-tuning).

It is used to choose the WPSNR target of each of the 3 images (e.g. 58.05 or 62.05 dB) and to produce
the files to upload. It works only on OUR images with OUR detection.

Mode 1 - TRIAL (default):
  for every image and every target:
    - embedding at that target;
    - checks: watermarked -> 1, original -> 0, detection time;
    - the same attack we use in the challenge (attack_image() of attacks_fakebitinvestigation.py) against our detection:
      it is a strong opponent that searches the lightest attack that breaks us.
  Then it tries all the combinations of targets and estimates the defense points with the table of the rules:
    - quality on the MEAN of the WPSNRs of the 3 images (answer of the TAs);
    - robustness on the mean WPSNR of the successful attacks; 6 + 2 of bonus if none is broken.
  It prints the best combination and the TARGET_BY_IMAGE line to copy into the embedding.
  WARNING: it is ONE opponent only; in the challenge 9 groups attack us, so the bonus is less likely
  than it looks here.

Mode 2 - FINAL (--final):
  after writing TARGET_BY_IMAGE in embedding_fakebitinvestigation.py, it redoes the embedding with the code
  as it is (no parameter passed from here), checks WPSNR and detection and saves in <out>/upload/
  the images to upload with the name required by the rules (fakebitinvestigation_<imageName>.bmp)
  and the encrypted detection detection_fakebitinvestigation.pyc, compiled from the current
  ../submission/detection_fakebitinvestigation.py with py_compile (the method shown in lab 1).

Usage (from submission/, like all the challenge commands):
  python ../challenge/defense_fakebitinvestigation.py --images DIR_WITH_THE_3_IMAGES --out DIR [--targets 58.05 62.05]
                                                      [--budget SECONDS] [--procs N]
  python ../challenge/defense_fakebitinvestigation.py --images DIR_WITH_THE_3_IMAGES --out DIR --final
"""
import argparse
import itertools
import os
import py_compile
import sys
import time
from multiprocessing import Pool

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SUBMISSION = os.path.join(HERE, "..", "submission")   # the code is the submitted one: a single copy, no duplicates
sys.path.insert(0, SUBMISSION)
import embedding_fakebitinvestigation as emb                             # noqa: E402
from detection_fakebitinvestigation import TAU, detection, wpsnr        # noqa: E402
from attacks_fakebitinvestigation import attack_image                   # noqa: E402

GROUP = "fakebitinvestigation"
DETECTION = os.path.join(SUBMISSION, "detection_fakebitinvestigation.py")


def quality_points(w):
    for lo, pts in ((66, 6), (62, 5), (58, 4), (54, 3), (50, 2), (35, 1)):
        if w >= lo:
            return pts
    return 0


def robustness_points(w):
    for hi, pts in ((38, 6), (41, 5), (44, 4), (47, 3), (50, 2), (53, 1)):
        if w < hi:
            return pts
    return 0


def _attack(job):
    """One image against our detection: all the work is done by attack_image() of the attack file."""
    fname, det_path, orig_path, wm_path, out_dir, budget, use_greedy, timeout = job
    return attack_image(wm_path, orig_path, det_path, out_dir, budget, use_greedy, timeout)


def checks(orig_path, wm_path):
    """watermarked -> 1, original -> 0, detection time."""
    t = time.time()
    on_wm, _ = detection(orig_path, wm_path, wm_path)
    dt = time.time() - t
    on_orig, _ = detection(orig_path, wm_path, orig_path)
    ok = on_wm == 1 and on_orig == 0 and dt < 5
    return ok, f"watermarked->{on_wm}, original->{on_orig}, detection {dt:.2f} s"


def embed_to(orig_path, wm_file, target, out_path):
    emb.TARGET_BY_IMAGE = {}
    emb.TARGET_WPSNR = target
    wm = emb.embedding(orig_path, wm_file)
    cv2.imwrite(out_path, wm)
    return wpsnr(cv2.imread(orig_path, cv2.IMREAD_GRAYSCALE), wm)


def main():
    ap = argparse.ArgumentParser(description="Defense phase: choice of the targets and files to upload")
    ap.add_argument("--images", required=True, help="folder with the 3 downloaded original images")
    ap.add_argument("--watermark", default=os.path.join(SUBMISSION, "fakebitinvestigation.npy"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--targets", type=float, nargs="+", default=[58.05, 62.05])
    ap.add_argument("--budget", type=float, default=900, help="attack seconds per image and target (default 15 min)")
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--final", action="store_true", help="produces the files to upload with the current code")
    a = ap.parse_args()
    files = sorted(f for f in os.listdir(a.images) if f.lower().endswith(".bmp"))
    if not files:
        sys.exit(f"no .bmp image in {a.images}")
    os.makedirs(a.out, exist_ok=True)

    # ------------------------------------------------------------ FINAL mode
    if a.final:
        up = os.path.join(a.out, "upload")
        os.makedirs(up, exist_ok=True)
        print(f"TARGET_WPSNR = {emb.TARGET_WPSNR}, TARGET_BY_IMAGE = {emb.TARGET_BY_IMAGE}, TAU = {TAU}")
        ws, all_ok = [], True
        for f in files:
            name = os.path.splitext(f)[0]
            orig = os.path.join(a.images, f)
            out = os.path.join(up, f"{GROUP}_{name}.bmp")
            wm = emb.embedding(orig, a.watermark)            # code as it is, no parameter from here
            cv2.imwrite(out, wm)
            w = wpsnr(cv2.imread(orig, cv2.IMREAD_GRAYSCALE), wm)
            ok, msg = checks(orig, out)
            all_ok &= ok
            ws.append(w)
            print(f"  {name:20s} WPSNR {w:.3f} dB  {msg}  {'OK' if ok else '!! CHECK !!'}")
        print(f"mean WPSNR {np.mean(ws):.3f} dB -> {quality_points(np.mean(ws))} quality points")
        # encrypted detection to upload (lab 1 method), always compiled from the current code
        pyc = os.path.join(up, "detection_fakebitinvestigation.pyc")
        py_compile.compile(DETECTION, cfile=pyc, dfile="detection_fakebitinvestigation.py", doraise=True)
        print(f"encrypted detection: {pyc} (Python {sys.version_info.major}.{sys.version_info.minor})")
        print(f"\nfiles to upload in {up}" if all_ok else "\n!! some check did NOT pass: do not upload !!")
        return

    # ------------------------------------------------------------ TRIAL mode
    print(f"=== embedding at {a.targets} dB and checks (tau = {TAU}) ===")
    jobs, wps = [], {}
    for t in a.targets:
        d_img = os.path.join(a.out, f"target_{t:.2f}", "img")
        os.makedirs(d_img, exist_ok=True)
        for f in files:
            name = os.path.splitext(f)[0]
            orig = os.path.join(a.images, f)
            # name "<group>_<image>" as the attack script wants it; here the "group" is the target
            wm_path = os.path.join(d_img, f"t{t:.2f}_{name}.bmp")
            wps[(name, t)] = embed_to(orig, a.watermark, t, wm_path)
            ok, msg = checks(orig, wm_path)
            print(f"  {name:20s} target {t:.2f}: WPSNR {wps[(name, t)]:.3f} dB, {msg}  {'' if ok else '!! CHECK !!'}")
            d_out = os.path.join(a.out, f"target_{t:.2f}", "attacks")
            os.makedirs(d_out, exist_ok=True)
            jobs.append((os.path.basename(wm_path), DETECTION, orig, wm_path, d_out, a.budget, True, 10.0))

    print(f"\n=== attack against us: {len(jobs)} trials, at most {a.budget / 60:.0f} min each ===", flush=True)
    res = {}
    with Pool(max(1, min(a.procs, len(jobs)))) as p:
        for r in p.imap_unordered(_attack, jobs):
            name, t = r["image"], float(r["group"][1:])
            res[(name, t)] = r["wpsnr"]                       # None = not broken
            outcome = f"BROKEN at {r['wpsnr']:.2f} dB with {r['desc']}" if r["wpsnr"] else "not broken"
            if r["error"]:
                outcome = f"ERROR: {r['error']}"
            print(f"  {name:20s} target {t:.2f}: {outcome}  ({r['calls']} calls, {r['time']:.0f} s)", flush=True)

    names = [os.path.splitext(f)[0] for f in files]
    print("\n=== combinations of targets (defense points against this opponent) ===")
    rows = []
    for combo in itertools.product(a.targets, repeat=len(names)):
        q = quality_points(np.mean([wps[(n, t)] for n, t in zip(names, combo)]))
        att = [res[(n, t)] for n, t in zip(names, combo) if res.get((n, t))]
        r = 6 + 2 if not att else robustness_points(np.mean(att))
        rows.append((q + r, q, r, len(att), combo))
    rows.sort(key=lambda x: (-x[0], -x[1]))
    print(f"  {'total':>6s} {'quality':>7s} {'robust.':>7s} {'broken':>6s}  target per image")
    for tot, q, r, n_br, combo in rows[:8]:
        print(f"  {tot:6d} {q:7d} {r:7d} {n_br:6d}  " + ", ".join(f"{n}={t:.2f}" for n, t in zip(names, combo)))
    mapping = {n: t for n, t in zip(names, rows[0][4])}
    print("\nProposal (to be evaluated: only one simulated opponent, in the challenge there are 9). Line for the embedding:")
    print(f"  TARGET_BY_IMAGE = {mapping}")
    print("The detection must NOT be touched (tau does not depend on the target). Then:\n"
          f"  python ../challenge/defense_fakebitinvestigation.py --images {a.images} --out {a.out} --final")


if __name__ == "__main__":
    main()
