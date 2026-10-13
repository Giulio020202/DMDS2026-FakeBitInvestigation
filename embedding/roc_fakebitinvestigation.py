"""
Capture the Mark 2026 - calcolo della soglia tau (ROC) del gruppo fakebitinvestigation.

Segue la procedura di p.4 del regolamento (Lab4, ROC-curves, esempio 2):
  1. embedding del nostro watermark W_original su un insieme di immagini;
  2. in un ciclo, attacchi casuali (solo quelli permessi: awgn, blur, sharpening, jpeg,
     resize, median, anche combinati o localizzati) sulle immagini watermarkate;
  3. estrazione con la stessa funzione della detection;
  4. sim(W_extracted dalla watermarkata, W estratto dall'attaccata) -> score con label 1
     (e' esattamente la similarita' calcolata dalla detection);
  5. per ogni attacco, N_RANDOM watermark casuali W_random (in forma +-1, la stessa mappatura
     dell'embedding) -> sim(W_random, W estratto dall'attaccata) con la STESSA funzione di
     similarita' della detection (massimo sulle varianti di estrazione) -> label 0;
  6. curva ROC (sklearn) e scelta di tau: la soglia piu' bassa con FPR <= FPR_TARGET
     (dentro l'intervallo ammesso [0, 0.1]).

Questo codice NON fa parte della detection (p.4: la soglia si calcola una volta sola, fuori).

Uso (dalla cartella del progetto):
    python roc_fakebitinvestigation.py [cartella_immagini] [n_attacchi_per_immagine]
Output: stampa tau e salva la curva in roc_fakebitinvestigation.png
"""
import glob
import io
import os
import sys
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

WATERMARK = "fakebitinvestigation.npy"
FPR_TARGET = 0.001      # FPR scelta per tau (il regolamento ammette [0, 0.1])
N_RANDOM = 10           # watermark casuali per ogni attacco (campioni con label 0)
SEED = 2026


# ----------------------------------------------------------------- attacchi (lab)
def _u8(x):
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def awgn(img, std, seed):
    return _u8(img.astype(float) + np.random.default_rng(seed).normal(0.0, std, img.shape))


def blur(img, sigma):
    return _u8(gaussian_filter(img.astype(float), sigma))


def sharpening(img, sigma, alpha):
    f = img.astype(float)
    return _u8(f + alpha * (f - gaussian_filter(f, sigma)))


def median(img, k):
    return _u8(medfilt(img.astype(float), k))


def resizing(img, scale):
    x, y = img.shape
    a = rescale(img.astype(float), scale)
    a = rescale(a, 1 / scale)[:x, :y]
    if a.shape != (x, y):
        a = np.pad(a, ((0, x - a.shape[0]), (0, y - a.shape[1])), mode='edge')
    return _u8(a)


def jpeg_compression(img, qf):
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, "JPEG", quality=int(qf))
    buf.seek(0)
    return np.asarray(Image.open(buf), dtype=np.uint8)


def _smooth_mask(img, q):
    f = img.astype(float)
    m = uniform_filter(f, 16)
    sd = np.sqrt(np.maximum(uniform_filter(f * f, 16) - m * m, 0))
    return sd <= np.quantile(sd, q)


def random_attack(img, rng):
    """Un attacco casuale tra quelli permessi (con parametri casuali)."""
    k = rng.integers(10)
    if k == 0:
        return jpeg_compression(img, int(rng.integers(3, 95))), "jpeg"
    if k == 1:
        return awgn(img, rng.uniform(1, 30), int(rng.integers(1 << 30))), "awgn"
    if k == 2:
        return blur(img, rng.uniform(0.3, 3)), "blur"
    if k == 3:
        return sharpening(img, float(rng.choice([0.5, 1, 2, 4, 10])), rng.uniform(0.2, 3)), "sharp"
    if k == 4:
        return median(img, int(rng.choice([3, 5, 7, 9]))), "median"
    if k == 5:
        return resizing(img, round(rng.uniform(0.2, 0.95), 2)), "resize"
    if k == 6:
        return blur(awgn(img, rng.uniform(5, 30), int(rng.integers(1 << 30))), rng.uniform(0.5, 2)), "awgn+blur"
    if k == 7:
        return jpeg_compression(blur(img, rng.uniform(0.3, 1.5)), int(rng.integers(3, 50))), "blur+jpeg"
    if k == 8:   # localizzato sulle zone lisce (maschera calcolata dalla watermarkata)
        m = _smooth_mask(img, rng.uniform(0.3, 0.7))
        out = img.copy()
        out[m] = jpeg_compression(img, int(rng.integers(2, 20)))[m]
        return out, "jpeg locale"
    return median_filter(img, size=int(rng.choice([3, 5, 7]))), "median_nd"


# ----------------------------------------------------------------- un'immagine
def _one_image(args):
    path, n_att, seed = args
    rng = np.random.default_rng(seed)
    original = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    watermarked = emb.embedding(path, WATERMARK)
    c0 = det._dct2(original.astype(np.float64))
    uu, vv = det._select_pairs(c0)
    W_ext = det._extract(c0, uu, vv, watermarked.astype(np.float64))
    scores, labels, wps = [], [], []
    for _ in range(n_att):
        attacked, _ = random_attack(watermarked, rng)
        vecs = det._attacked_vectors(c0, uu, vv, original, attacked)
        q = emb.wpsnr(watermarked, attacked) if not np.array_equal(watermarked, attacked) else 99.0
        scores.append(det._score(W_ext, vecs)); labels.append(1); wps.append(q)
        for _ in range(N_RANDOM):
            W_random = rng.choice([-1.0, 1.0], W_ext.size)
            scores.append(det._score(W_random, vecs)); labels.append(0); wps.append(q)
    return scores, labels, wps


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else "sample-images"
    n_att = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    files = sorted(glob.glob(os.path.join(folder, "*.bmp")))
    t0 = time.time()
    with Pool(max(1, (os.cpu_count() or 2) - 1)) as p:
        res = p.map(_one_image, [(f, n_att, SEED + i) for i, f in enumerate(files)], chunksize=1)
    scores = np.concatenate([r[0] for r in res])
    labels = np.concatenate([r[1] for r in res])
    wps = np.concatenate([r[2] for r in res])

    fpr, tpr, thr = roc_curve(labels, scores, drop_intermediate=False)
    h0 = scores[labels == 0]
    h1 = scores[labels == 1]
    # tau = la soglia piu' bassa con FPR <= FPR_TARGET: lasciamo sopra soglia al massimo
    # floor(FPR_TARGET * n) campioni H0, mettendo tau appena sopra il successivo.
    h0s = np.sort(h0)
    m = int(np.floor(FPR_TARGET * len(h0s)))
    tau = float(np.nextafter(h0s[len(h0s) - m - 1], np.inf))
    valid = (labels == 1) & (wps >= 35)
    print(f"{len(files)} immagini, {n_att} attacchi ciascuna, {N_RANDOM} watermark casuali per attacco "
          f"({len(h1)} campioni H1, {len(h0)} H0), {time.time() - t0:.0f}s")
    print(f"AUC = {auc(fpr, tpr):.5f}")
    print(f"H0: media {h0.mean():+.4f}, dev.std {h0.std():.4f}, quantile 99.9% {np.quantile(h0, 0.999):.4f}, max {h0.max():.4f}")
    print(f"tau (FPR <= {FPR_TARGET}) = {tau:.4f}  ->  FPR {np.mean(h0 >= tau):.5f}, TPR {np.mean(h1 >= tau):.4f}; "
          f"TPR sui soli attacchi con WPSNR >= 35: {np.mean(scores[valid] >= tau):.4f}")
    for t in (0.08, 0.10, 0.12):
        print(f"  tau={t:.2f}: FPR {np.mean(h0 >= t):.5f}, TPR {np.mean(h1 >= t):.4f}")

    import matplotlib
    matplotlib.use("Agg")                        # nessuna finestra: salviamo solo il file
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, lw=2, label=f"ROC (AUC = {auc(fpr, tpr):.4f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.axvline(FPR_TARGET, color="r", lw=1, label=f"FPR = {FPR_TARGET} -> tau = {tau:.3f}")
    plt.xscale("symlog", linthresh=1e-3)
    plt.xlim(0, 1); plt.ylim(0, 1.01)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC - fakebitinvestigation"); plt.legend(loc="lower right")
    plt.savefig("roc_fakebitinvestigation.png", dpi=120, bbox_inches="tight")


if __name__ == "__main__":
    main()
