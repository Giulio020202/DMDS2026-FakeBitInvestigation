"""
Capture the Mark 2026 - HELPER for the attack phase of the challenge (group fakebitinvestigation). NOT submitted.

It contains no attack: all the logic (allowed attacks, zones, families, search of the lightest attack
with the other group's detection as oracle, block greedy) is in ../submission/attacks_fakebitinvestigation.py,
function attack_image(). This file only does the "loop" over all the downloaded images:
  - matches every image <group>_<imageName>.bmp with detection_<group>.py (or the encrypted
    detection_<group>.pyc) and with the original <imageName>.bmp;
  - splits the total time among the images and attacks them in parallel (several processes);
  - writes the log for the TAs (Image | Group | WPSNR | Attack(s) with parameters) as it goes, with --resume;
  - --check: checks in a minute that every detection loads and says "present" on the watermarked image.

Expected folders:
  --images      downloaded watermarked images:   <group>_<imageName>.bmp
  --originals   original images:                 <imageName>.bmp (they go ONLY to their detection, as input1)
  --detections  detections of the other groups:  detection_<group>.py or detection_<group>.pyc
  --out         where to save:                   fakebitinvestigation_<group>_<imageName>.bmp, attack_log.csv
                                                 (to be sent), all_successful.csv (alternatives)

Usage (from submission/, like all the challenge commands):
  python ../challenge/search_attacks_fakebitinvestigation.py --images DIR --originals DIR --detections DIR --out DIR
         [--only group] [--procs N] [--total-min MINUTES] [--budget SECONDS] [--timeout SECONDS]
         [--no-greedy] [--resume] [--check]
"""
import os

# one thread per process (the processes already work in parallel) and no matplotlib window:
# it must be done BEFORE importing numpy/cv2
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse                                                          # noqa: E402
import csv                                                               # noqa: E402
import sys                                                               # noqa: E402
import tempfile                                                          # noqa: E402
import time                                                              # noqa: E402
from multiprocessing import Pool                                         # noqa: E402

import cv2                                                               # noqa: E402

cv2.setNumThreads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "submission"))              # the attack code is the submitted one
from attacks_fakebitinvestigation import OUR_GROUP, Oracle, attack_image  # noqa: E402


def attack_one(job):
    """One image: all the work is done by attack_image() of the attack file."""
    fname, det_path, orig_path, wm_path, out_dir, budget, use_greedy, timeout = job
    return attack_image(wm_path, orig_path, det_path, out_dir, budget, use_greedy, timeout)


LOG_HEADER = ["Image", "Group", "WPSNR", "Attack(s) with parameters"]


def find_detection(folder, group):
    """detection_<group>.py, or the encrypted detection_<group>.pyc if there is no .py."""
    for ext in (".py", ".pyc"):
        p = os.path.join(folder, f"detection_{group}{ext}")
        if os.path.exists(p):
            return p
    return os.path.join(folder, f"detection_{group}.py")


def check_one(job):
    """Quick check of a detection: does it load? does it say 'present' on the watermarked image? how long does it take?"""
    fname, det_path, orig_path, wm_path, timeout = job
    try:
        with tempfile.TemporaryDirectory() as tmp:
            o = Oracle(det_path, orig_path, wm_path, tmp, timeout)
            t = time.time()
            broken = o.broken(cv2.imread(wm_path, cv2.IMREAD_GRAYSCALE))
            dt = time.time() - t
            if o.errors:
                return fname, f"ERROR: {o.last_error}"
            if broken:
                return fname, f"WARNING: it says 'absent' on the watermarked image ({dt:.2f} s)"
            return fname, f"ok, {dt:.2f} s per call"
    except Exception as e:                             # noqa: BLE001
        return fname, f"ERROR: {type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser(description="Challenge attacks of group fakebitinvestigation")
    ap.add_argument("--images", required=True)
    ap.add_argument("--originals", required=True)
    ap.add_argument("--detections", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default=None, help="attacks only this group")
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--total-min", type=float, default=100, help="total minutes available (default 100)")
    ap.add_argument("--budget", type=float, default=None, help="seconds per image (default: from --total-min)")
    ap.add_argument("--timeout", type=float, default=10, help="maximum seconds for one call to the detection")
    ap.add_argument("--no-greedy", action="store_true",
                    help="skips the zones chosen with the other group's detection (allowed by the TAs)")
    ap.add_argument("--resume", action="store_true", help="skips the images already in the log")
    ap.add_argument("--check", action="store_true",
                    help="check only: every detection loads, says 'present' on the watermarked image, time")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    log_path = os.path.join(a.out, "attack_log.csv")
    all_path = os.path.join(a.out, "all_successful.csv")

    done = set()
    if a.check:
        pass                                           # the check does not touch the logs
    elif a.resume and os.path.exists(log_path):
        with open(log_path, newline="") as f:
            done = {(r["Group"], r["Image"]) for r in csv.DictReader(f)}
    else:
        for path, header in ((log_path, LOG_HEADER), (all_path, ["Image", "Group", "WPSNR", "Attack"])):
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(header)

    jobs = []
    for fname in sorted(os.listdir(a.images)):
        if not fname.endswith(".bmp") or "_" not in fname:
            continue
        group, image = fname[:-4].split("_", 1)
        if group == OUR_GROUP or (a.only and group != a.only) or (group, image) in done:
            continue
        det_path = find_detection(a.detections, group)
        orig_path = os.path.join(a.originals, f"{image}.bmp")
        if not os.path.exists(det_path) or not os.path.exists(orig_path):
            print(f"[skip] {fname}: missing {det_path if not os.path.exists(det_path) else orig_path}")
            continue
        jobs.append([fname, det_path, orig_path, os.path.join(a.images, fname), a.out, 0, not a.no_greedy, a.timeout])
    if not jobs:
        sys.exit("no image to attack")

    if a.check:
        print(f"check of {len(jobs)} images ({len({j[0].split('_', 1)[0] for j in jobs})} groups):")
        with Pool(max(1, min(a.procs, len(jobs)))) as p:
            for fname, msg in p.imap(check_one, [(j[0], j[1], j[2], j[3], a.timeout) for j in jobs]):
                print(f"  {fname:40s} {msg}", flush=True)
        return

    procs = max(1, min(a.procs, len(jobs)))
    rounds = -(-len(jobs) // procs)                    # rounds of images in parallel (rounded up)
    budget = a.budget if a.budget else 0.95 * a.total_min * 60 / rounds
    for j in jobs:
        j[5] = budget
    print(f"{len(jobs)} images, {procs} processes ({rounds} rounds), budget {budget / 60:.1f} min per image, "
          f"expected maximum time ~{rounds * budget / 60:.0f} min", flush=True)

    t0, broken, failed = time.time(), 0, []
    with Pool(procs) as p:
        for k, r in enumerate(p.imap_unordered(attack_one, [tuple(j) for j in jobs]), 1):
            tag = f"[{k}/{len(jobs)} {(time.time() - t0) / 60:.0f} min]"
            info = f"({r['calls']} calls, {r['errors']} errors, {r['time']:.0f} s)"
            if r["error"]:
                failed.append(f"{r['group']}_{r['image']}: {r['error']}")
                print(f"{tag} [ERROR] {r['group']}_{r['image']}: {r['error']} {info}", flush=True)
            elif r["wpsnr"] is None:
                print(f"{tag} [not broken] {r['group']}_{r['image']} {info}", flush=True)
            else:
                broken += 1
                print(f"{tag} [BROKEN] {r['group']}_{r['image']}: {r['wpsnr']:.2f} dB with {r['desc']} {info}",
                      flush=True)
                with open(log_path, "a", newline="") as f:
                    csv.writer(f).writerow([r["image"], r["group"], f"{r['wpsnr']:.2f}", r["desc"]])
            with open(all_path, "a", newline="") as f:
                csv.writer(f).writerows([[r["image"], r["group"], w, d] for w, d in r["found"]])

    print(f"\nbroken {broken}/{len(jobs)} in {(time.time() - t0) / 60:.1f} min; log: {log_path}")
    if failed:
        print("images with errors (to be looked at by hand):\n  " + "\n  ".join(failed))


if __name__ == "__main__":
    main()
