"""
Capture the Mark 2026 - DEMO of group fakebitinvestigation (version 2.0 hybrid DCT + wavelet)

Shows how to use every submitted file:
  1. embedding_fakebitinvestigation.py   embedding(original, watermark) -> watermarked
  2. detection_fakebitinvestigation.py   detection(original, watermarked, attacked) -> (present?, WPSNR)
  3. attacks_fakebitinvestigation.py     attacks(watermarked, name, parameters) -> attacked
  4. attacks_fakebitinvestigation.py     automatic search of the lightest attack, attack_image() (with --attacks)
  5. roc_fakebitinvestigation.py         ROC and threshold tau (with --roc, quick version)

Usage (from submission/):
  python demo_fakebitinvestigation.py [--image FILE.bmp] [--watermark FILE.npy] [--out DIR] [--attacks] [--roc]
Default: demo_images/0000.bmp, fakebitinvestigation.npy (both in this folder), folder ./demo_output.
With --roc the quick ROC uses the 10 images of demo_images/: the demo works with the submitted folder alone.
"""
import argparse
import os
import shutil
import subprocess
import sys
import time

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from embedding_fakebitinvestigation import embedding                    # noqa: E402
from detection_fakebitinvestigation import TAU, detection, wpsnr        # noqa: E402
from attacks_fakebitinvestigation import MASKS, attacks, describe       # noqa: E402

GROUP = "fakebitinvestigation"

# (chain, parameters, mask): from the lightest to the heaviest
DEMO_ATTACKS = [
    ("jpeg", [50], None),
    ("jpeg", [10], None),
    ("blur", [0.8], None),
    ("awgn", [8], None),
    ("sharp", [1, 1], None),
    ("median", [5], None),
    ("resize", [0.5], None),
    (["blur", "jpeg"], [[0.6], [15]], None),
    ("jpeg", [5], "smooth50"),
    ("awgn", [40], "smooth70"),
    ("jpeg", [2], None),            # destructive attack: WPSNR < 35, it does not count as successful
]


# Runs the whole demo: embedding of one image, detection on the watermarked image and on the original,
# the attacks of DEMO_ATTACKS with their detection outcome, and optionally (--attacks, --roc) the
# automatic attack search against our detection and the quick ROC. Everything is printed on screen
# and the images are saved in the --out folder.
#
# The options are read from the command line (see the usage above), so the function has no parameters.
# @return  None
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=os.path.join(HERE, "demo_images", "0000.bmp"))
    ap.add_argument("--watermark", default=os.path.join(HERE, "fakebitinvestigation.npy"))
    ap.add_argument("--out", default=os.path.join(HERE, "demo_output"))
    ap.add_argument("--attacks", action="store_true", help="also tries the attack script against our detection")
    ap.add_argument("--roc", action="store_true", help="also runs the quick ROC (--quick)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    name = os.path.splitext(os.path.basename(a.image))[0]

    # ---------------------------------------------------------------- 1. embedding
    print("=== 1. EMBEDDING ===")
    t = time.time()
    wm = embedding(a.image, a.watermark)
    original = cv2.imread(a.image, cv2.IMREAD_GRAYSCALE)
    p_wm = os.path.join(a.out, f"{GROUP}_{name}.bmp")
    cv2.imwrite(p_wm, wm)
    print(f"embedding({os.path.basename(a.image)}, {os.path.basename(a.watermark)}) in {time.time() - t:.2f} s")
    print(f"WPSNR(original, watermarked) = {wpsnr(original, wm):.2f} dB   -> saved in {p_wm}")

    # ---------------------------------------------------------------- 2. detection
    print(f"\n=== 2. DETECTION (tau = {TAU}) ===")
    print(f"{'image given as attacked':66s} {'output1':>7s} {'output2 (WPSNR)':>16s} {'time':>7s}  outcome")

    # Calls our detection with the given image as the "attacked" one and prints one row of the table
    # (output1, output2, time and outcome).
    #
    # @param label        text shown in the first column
    # @param path         path of the image given to the detection as input3
    # @param is_original  True if path is the original image (then the expected output1 is 0)
    # @return             None
    def run(label, path, is_original=False):
        t = time.time()
        o1, o2 = detection(a.image, p_wm, path)
        dt = time.time() - t
        if is_original:
            outcome = "correct: on the original the watermark is absent" if o1 == 0 else "ERROR: false positive"
        elif o1 == 0 and o2 >= 35:
            outcome = "SUCCESSFUL ATTACK"
        elif o1 == 0:
            outcome = "absent, but WPSNR < 35: attack not valid"
        else:
            outcome = "watermark present"
        print(f"{label:66s} {o1:7d} {o2:16.2f} {dt:6.2f}s  {outcome}")

    run("watermarked (no attack)", p_wm)
    run("original (must give 0)", a.image, is_original=True)

    # ---------------------------------------------------------------- 3. attacks
    print("\n=== 3. ATTACKS (attacks_fakebitinvestigation.py) + detection ===")
    for k, (names, params, mname) in enumerate(DEMO_ATTACKS):
        mask = MASKS[mname](wm) if mname else None
        att = attacks(p_wm, names, params, mask=mask)
        p_att = os.path.join(a.out, f"{GROUP}_demo_{name}_{k:02d}.bmp")
        cv2.imwrite(p_att, att)
        run(describe(names, params, mname), p_att)

    # ---------------------------------------------------------------- 4. attacks (optional)
    if a.attacks:
        print("\n=== 4. AUTOMATIC ATTACK SEARCH against our detection (as if we were another group 'demo') ===")
        g = os.path.join(a.out, "attacks")
        os.makedirs(g, exist_ok=True)
        p_demo = os.path.join(g, f"demo_{name}.bmp")              # name <group>_<imageName>.bmp, as in the challenge
        shutil.copy(p_wm, p_demo)
        sys.stdout.flush()
        subprocess.run([sys.executable, os.path.join(HERE, "attacks_fakebitinvestigation.py"), "search",
                        "--image", p_demo, "--original", a.image,
                        "--detection", os.path.join(HERE, "detection_fakebitinvestigation.py"),
                        "--out", os.path.join(g, "out"), "--budget", "300"], check=True)

    # ---------------------------------------------------------------- 5. ROC (optional)
    if a.roc:
        print("\n=== 5. Quick ROC (the real tau comes from the full run, see README) ===")
        sys.stdout.flush()
        subprocess.run([sys.executable, os.path.join(HERE, "roc_fakebitinvestigation.py"), "--quick",
                        "--watermark", a.watermark, "--images", os.path.join(HERE, "demo_images"),
                        "--out", os.path.join(a.out, "roc_quick.png")], check=True)

    print(f"\nAll the demo files are in {a.out}")


if __name__ == "__main__":
    main()
