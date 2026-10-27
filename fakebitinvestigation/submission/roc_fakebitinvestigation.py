"""
Capture the Mark 2026 - computation of the threshold tau (ROC) of group fakebitinvestigation
(version 2.0, HYBRID DCT + wavelet: W has 2048 values, 1024 DCT + 1024 wavelet).

Follows the procedure of p.4 of the rules (Lab4, ROC-curves, example 2), with the extra checks of Lab 4
(H0-A, H0-B and separate test set):
  1. embedding of our watermark W_original on a set of images;
  2. in a loop, random attacks (only the allowed ones: awgn, blur, sharpening, jpeg,
     resize, median, also combined or localized) on the watermarked images;
  3. extraction with the same function as the detection;
  4. sim(W_extracted from the watermarked image, W extracted from the attacked one) -> score with label 1 (H1)
     (it is exactly the similarity computed by the detection);
  5. scores with label 0, with the SAME similarity function as the detection (maximum over the variants):
     H0-R: for every attack, N_RANDOM random watermarks W_random (in +-1 form, the same mapping as
           the embedding, put in both parts with the amplitude of each part taken from W_extracted)
           compared with the W extracted from the attacked image (it is the procedure of the rules);
     H0-A: the SAME attack applied to the non-watermarked original (Lab 4);
     H0-B: the SAME attack applied to the same image with the watermark of "another group"
           (same scheme, random watermark) (Lab 4);
  6. the images are split into CALIBRATION (70%) and TEST (30%, never used to choose tau);
     on the calibration: ROC curve (sklearn) and tau = the lowest threshold with FPR <= FPR_TARGET
     on EACH of the three kinds of H0 (inside the allowed interval [0, 0.1]);
     on the test: tau frozen, FPR (per kind) and TPR are measured.

This code is NOT part of the detection (p.4: the threshold is computed only once, outside).

Usage (from any folder):
    python roc_fakebitinvestigation.py [--images DIR] [--watermark FILE.npy] [--attacks N] [--quick] [--out FILE.png]
  --images     folder with the BMP 512x512 images (default: ../sample-images, the 101 images of the full run;
               if it is not there, demo_images/ of this folder, 10 images)
  --watermark  .npy watermark (default: fakebitinvestigation.npy in this folder)
  --attacks    random attacks per image (default 40)
  --quick      quick test: only the first 10 images and 5 attacks each; the tau that comes out is NOT
               the real one (too few samples), it only shows that everything runs
  --out        where to save the curve (default: roc_fakebitinvestigation.png next to this file)
Output: tau printed on screen + .png curve.
Extra dependencies with respect to embedding/detection: scikit-learn, matplotlib, pillow, scikit-image.
"""
import argparse
import glob
import io
import os
import sys
import tempfile
import time
from multiprocessing import Pool

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, median_filter, uniform_filter
from scipy.signal import medfilt
from skimage.transform import rescale
from sklearn.metrics import roc_curve, auc

import embedding_fakebitinvestigation as emb
import detection_fakebitinvestigation as det

HERE = os.path.dirname(os.path.abspath(__file__))
WATERMARK = os.path.join(HERE, "fakebitinvestigation.npy")         # our watermark, in this folder
IMAGES = os.path.join(HERE, "..", "sample-images")                 # the 101 images of the full run
if not os.path.isdir(IMAGES):
    IMAGES = os.path.join(HERE, "demo_images")                     # only the submitted folder: the 10 of the demo
FPR_TARGET = 0.001      # FPR chosen for tau (the rules allow [0, 0.1])
N_RANDOM = 10           # random watermarks for every attack (samples with label 0)
SEED = 2026
TEST_EVERY = 10          # images with index % 10 >= 7 -> TEST set (30%), the others -> CALIBRATION


# ----------------------------------------------------------------- attacks (lab)
# Rounds and clips an image to valid gray levels, like a real BMP.
#
# @param x  image as a float array
# @return   uint8 image (values in 0..255)
def _u8(x):
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


# Additive white gaussian noise.
#
# @param img   input image (uint8)
# @param std   standard deviation of the noise, in gray levels
# @param seed  seed of the random generator (same seed -> same noise)
# @return      noisy image (uint8)
def awgn(img, std, seed):
    return _u8(img.astype(float) + np.random.default_rng(seed).normal(0.0, std, img.shape))


# Gaussian blur.
#
# @param img    input image (uint8)
# @param sigma  standard deviation of the gaussian filter, in pixels
# @return       blurred image (uint8)
def blur(img, sigma):
    return _u8(gaussian_filter(img.astype(float), sigma))


# Sharpening by unsharp masking: img + alpha * (img - blur(img, sigma)).
#
# @param img    input image (uint8)
# @param sigma  standard deviation of the blur used for the mask
# @param alpha  strength of the sharpening
# @return       sharpened image (uint8)
def sharpening(img, sigma, alpha):
    f = img.astype(float)
    return _u8(f + alpha * (f - gaussian_filter(f, sigma)))


# Median filter (scipy medfilt, zero-padded borders, as in the labs).
#
# @param img  input image (uint8)
# @param k    size of the (odd) square window
# @return     filtered image (uint8)
def median(img, k):
    return _u8(medfilt(img.astype(float), k))


# Lab resize: rescale down by 'scale' and up again with skimage, then crop to the original size
# (if the result is smaller, the last row/column is repeated).
#
# @param img    input image (uint8)
# @param scale  scale factor of the first rescale (0 < scale < 1)
# @return       resized image with the original size (uint8)
def resizing(img, scale):
    x, y = img.shape
    a = rescale(img.astype(float), scale)
    a = rescale(a, 1 / scale)[:x, :y]
    if a.shape != (x, y):
        a = np.pad(a, ((0, x - a.shape[0]), (0, y - a.shape[1])), mode='edge')
    return _u8(a)


# JPEG compression and decompression in memory (Pillow), no file is written.
#
# @param img  input image (uint8)
# @param qf   JPEG quality factor (1..100)
# @return     decompressed image (uint8)
def jpeg_compression(img, qf):
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, "JPEG", quality=int(qf))
    buf.seek(0)
    return np.asarray(Image.open(buf), dtype=np.uint8)


# Mask of the smoothest zones of an image (lowest local standard deviation on 16x16 windows).
#
# @param img  image the mask is computed from
# @param q    fraction of the image to select (0..1)
# @return     boolean mask, same shape as img
def _smooth_mask(img, q):
    f = img.astype(float)
    m = uniform_filter(f, 16)
    sd = np.sqrt(np.maximum(uniform_filter(f * f, 16) - m * m, 0))
    return sd <= np.quantile(sd, q)


# Picks a random attack among the allowed ones, with random parameters. It returns a function, so
# that exactly the same attack can be applied to several images (watermarked, original,
# watermarked by another group).
#
# @param rng  numpy random generator (np.random.default_rng) used to draw the attack and its parameters
# @return     (function, name): function image -> attacked image (uint8), and a short name of the attack
def random_attack(rng):
    k = rng.integers(10)
    if k == 0:
        q = int(rng.integers(3, 95)); return (lambda x: jpeg_compression(x, q)), "jpeg"
    if k == 1:
        s, sd = rng.uniform(1, 30), int(rng.integers(1 << 30)); return (lambda x: awgn(x, s, sd)), "awgn"
    if k == 2:
        s = rng.uniform(0.3, 3); return (lambda x: blur(x, s)), "blur"
    if k == 3:
        s, a = float(rng.choice([0.5, 1, 2, 4, 10])), rng.uniform(0.2, 3); return (lambda x: sharpening(x, s, a)), "sharp"
    if k == 4:
        m = int(rng.choice([3, 5, 7, 9])); return (lambda x: median(x, m)), "median"
    if k == 5:
        s = round(rng.uniform(0.2, 0.95), 2); return (lambda x: resizing(x, s)), "resize"
    if k == 6:
        s, sd, b = rng.uniform(5, 30), int(rng.integers(1 << 30)), rng.uniform(0.5, 2)
        return (lambda x: blur(awgn(x, s, sd), b)), "awgn+blur"
    if k == 7:
        b, q = rng.uniform(0.3, 1.5), int(rng.integers(3, 50)); return (lambda x: jpeg_compression(blur(x, b), q)), "blur+jpeg"
    if k == 8:   # localized on the smooth zones (mask computed from the attacked image itself)
        qm, q = rng.uniform(0.3, 0.7), int(rng.integers(2, 20))

        # JPEG applied only on the smooth zones of the image it receives.
        #
        # @param x  image to attack (uint8)
        # @return   attacked image (uint8): JPEG inside the mask, x unchanged outside
        def f(x):
            m = _smooth_mask(x, qm)
            out = x.copy()
            out[m] = jpeg_compression(x, q)[m]
            return out
        return f, "local jpeg"
    m = int(rng.choice([3, 5, 7])); return (lambda x: median_filter(x, size=m)), "median_nd"


# ----------------------------------------------------------------- one image
# Computes all the ROC samples of one image: embedding of our watermark and of a random one
# ("another group"), then n_att random attacks; for each attack one H1 score, N_RANDOM H0-R scores,
# one H0-A score and one H0-B score. It runs in a separate process of the Pool.
#
# @param args  tuple (path, n_att, seed, watermark): path of the image, number of attacks,
#              seed of the random generator, path of our .npy watermark
# @return      (scores, kinds, wps): lists with the similarity of each sample, its kind
#              ('H1', 'H0-R', 'H0-A', 'H0-B') and the WPSNR of the attacked image
def _one_image(args):
    path, n_att, seed, watermark = args
    rng = np.random.default_rng(seed)
    original = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    watermarked = emb.embedding(path, watermark)
    # "another group": same image, same scheme, random watermark (temporary file for embedding())
    with tempfile.TemporaryDirectory() as tmp:
        other_wm = os.path.join(tmp, "other.npy")
        np.save(other_wm, rng.integers(0, 2, emb.N_BITS))
        other = emb.embedding(path, other_wm)
    ctx = det._context(original)
    W_ext = det._extract(ctx, watermarked)
    half = W_ext.size // 2
    # amplitude (rms) of each part: the random watermark is put in the same form as W_ext
    amp = np.array([np.sqrt(np.mean(W_ext[:half] ** 2)), np.sqrt(np.mean(W_ext[half:] ** 2))])
    scores, kinds, wps = [], [], []       # kinds: 'H1', 'H0-R', 'H0-A', 'H0-B'
    for _ in range(n_att):
        f, _ = random_attack(rng)
        attacked = f(watermarked)
        vecs = det._attacked_vectors(ctx, original, attacked)
        q = emb.wpsnr(watermarked, attacked) if not np.array_equal(watermarked, attacked) else 99.0
        scores.append(det._score(W_ext, vecs)); kinds.append('H1'); wps.append(q)
        for _ in range(N_RANDOM):
            r = rng.choice([-1.0, 1.0], half)
            W_random = np.concatenate([amp[0] * r, amp[1] * r])
            scores.append(det._score(W_random, vecs)); kinds.append('H0-R'); wps.append(q)
        for kind, img in (('H0-A', original), ('H0-B', other)):
            a = f(img)
            scores.append(det._score(W_ext, det._attacked_vectors(ctx, original, a))); kinds.append(kind)
            wps.append(emb.wpsnr(watermarked, a))
    return scores, kinds, wps


# The lowest threshold with FPR <= fpr_target: at most floor(fpr_target * n) H0 samples are at or
# above the threshold.
#
# @param h0          scores of the H0 samples (array)
# @param fpr_target  highest false positive rate allowed (e.g. 0.001)
# @return            threshold tau (float)
def _tau(h0, fpr_target):
    h0s = np.sort(h0)
    m = int(np.floor(fpr_target * len(h0s)))
    return float(np.nextafter(h0s[len(h0s) - m - 1], np.inf))


# Runs the whole ROC procedure: samples of all the images in parallel, split into calibration and
# test, tau on the calibration, FPR and TPR on the test, results printed and ROC curve saved as .png.
#
# The options are read from the command line (see the usage above), so the function has no parameters.
# @return  None
def main():
    ap = argparse.ArgumentParser(description="ROC and threshold tau of group fakebitinvestigation (v2.0)")
    ap.add_argument("--images", default=IMAGES)
    ap.add_argument("--watermark", default=WATERMARK)
    ap.add_argument("--attacks", type=int, default=40)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "roc_fakebitinvestigation.png"))
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.images, "*.bmp")))
    n_att = a.attacks
    if a.quick:
        files, n_att = files[:10], 5
    if not files:
        sys.exit(f"no .bmp image in {a.images}")
    t0 = time.time()
    with Pool(max(1, (os.cpu_count() or 2) - 1)) as p:
        res, res_idx = [], []
        jobs = [(f, n_att, SEED + i, a.watermark) for i, f in enumerate(files)]
        for i, r in enumerate(p.imap(_one_image, jobs, chunksize=1)):
            res.append(r); res_idx.append(i)
            k = len(res)                     # progress bar (on stderr)
            el = time.time() - t0
            sys.stderr.write(f"\r[{'#' * (30 * k // len(files)):30s}] {k}/{len(files)} images, "
                             f"{el / 60:.1f} min, ~{el / k * (len(files) - k) / 60:.1f} min left ")
            sys.stderr.flush()
        sys.stderr.write("\n")
    # res_idx[j] = index of the image of result j: decides whether it is calibration or test
    test_set = {i for i in range(len(files)) if i % TEST_EVERY >= 7}
    print(f"{len(files)} images ({len(files) - len(test_set)} calibration, {len(test_set)} test), {n_att} attacks "
          f"each, {N_RANDOM} random watermarks per attack, {time.time() - t0:.0f}s")

    # Puts together the samples of a subset of the images.
    #
    # @param sel  indices (in res) of the results to merge
    # @return     (sc, kd, wp): arrays with the scores, the kinds and the WPSNR of all the samples
    def gather(sel):
        sc = np.concatenate([np.asarray(res[j][0]) for j in sel])
        kd = np.concatenate([np.asarray(res[j][1]) for j in sel])
        wp = np.concatenate([np.asarray(res[j][2]) for j in sel])
        return sc, kd, wp

    cal = [j for j in range(len(res)) if res_idx[j] not in test_set]
    tst = [j for j in range(len(res)) if res_idx[j] in test_set]
    sc, kd, wp = gather(cal)
    h1 = sc[kd == 'H1']
    taus = {k: _tau(sc[kd == k], FPR_TARGET) for k in ('H0-R', 'H0-A', 'H0-B')}
    tau = max(taus.values())
    labels = (kd == 'H1').astype(int)
    fpr, tpr, thr = roc_curve(labels, sc, drop_intermediate=False)
    print(f"CALIBRATION: {len(h1)} H1 samples, H0-R {np.sum(kd == 'H0-R')}, H0-A {np.sum(kd == 'H0-A')}, "
          f"H0-B {np.sum(kd == 'H0-B')}; AUC (all H0 together) = {auc(fpr, tpr):.5f}")
    for k, t in taus.items():
        x = sc[kd == k]
        print(f"  {k}: mean {x.mean():+.4f}, std {x.std():.4f}, 99.9% quantile {np.quantile(x, 0.999):.4f}, "
              f"max {x.max():.4f} -> threshold for FPR <= {FPR_TARGET}: {t:.4f}")
    valid = (kd == 'H1') & (wp >= 35)
    print(f"tau = maximum of the three = {tau:.4f}  ->  FPR H0-R {np.mean(sc[kd == 'H0-R'] >= tau):.5f}, "
          f"H0-A {np.mean(sc[kd == 'H0-A'] >= tau):.5f}, H0-B {np.mean(sc[kd == 'H0-B'] >= tau):.5f}; "
          f"TPR {np.mean(h1 >= tau):.4f}, TPR on the attacks with WPSNR >= 35 only: {np.mean(sc[valid] >= tau):.4f}")
    sc2, kd2, wp2 = gather(tst)
    valid2 = (kd2 == 'H1') & (wp2 >= 35)
    print(f"TEST (tau frozen {tau:.4f}): FPR H0-R {np.mean(sc2[kd2 == 'H0-R'] >= tau):.5f}, "
          f"H0-A {np.mean(sc2[kd2 == 'H0-A'] >= tau):.5f}, H0-B {np.mean(sc2[kd2 == 'H0-B'] >= tau):.5f}; "
          f"TPR {np.mean(sc2[kd2 == 'H1'] >= tau):.4f}, TPR on the attacks with WPSNR >= 35 only: {np.mean(sc2[valid2] >= tau):.4f}")
    for t in (0.08, 0.086, 0.09, 0.10):
        print(f"  tau={t:.3f} (calibration): FPR H0-R {np.mean(sc[kd == 'H0-R'] >= t):.5f}, "
              f"H0-A {np.mean(sc[kd == 'H0-A'] >= t):.5f}, H0-B {np.mean(sc[kd == 'H0-B'] >= t):.5f}, TPR {np.mean(h1 >= t):.4f}")

    import matplotlib
    matplotlib.use("Agg")                        # no window: we only save the file
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, lw=2, label=f"ROC (AUC = {auc(fpr, tpr):.4f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.axvline(FPR_TARGET, color="r", lw=1, label=f"FPR = {FPR_TARGET} -> tau = {tau:.3f}")
    plt.xscale("symlog", linthresh=1e-3)
    plt.xlim(0, 1); plt.ylim(0, 1.01)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC - fakebitinvestigation"); plt.legend(loc="lower right")
    plt.savefig(a.out, dpi=120, bbox_inches="tight")
    print(f"curve saved in {a.out}")


if __name__ == "__main__":
    main()
