"""
Script di prova per l'embedding. Non va consegnato come embedding/detection:
stampa a video, ed e' giusto cosi' perche' serve solo a noi.

Uso:
    python test_embedding.py <cartella_immagini_bmp> <watermark.npy> [tau]

Per ogni immagine:
  1. chiama embedding() del nostro file e salva  fakebitinvestigation_<nome>.bmp
  2. calcola il WPSNR con il codice dei lab e i punti "embedding quality"
  3. estrae il watermark con il prototipo di estrazione (sara' il cuore della detection)
  4. controlli di sanita': originale -> watermark assente, 1000 watermark casuali -> sim ~ 0
     (deviazione standard attesa 1/32)
  5. qualche attacco permesso (awgn, blur, sharp, jpeg, resize, median):
     sim(W_estratto, W_attaccato), WPSNR(watermarked, attaccata), esito.
"""
import io
import os
import sys
import glob
import time

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, median_filter

import embedding_fakebitinvestigation as emb

GROUP = "fakebitinvestigation"


# --------------------------------------------------------------- estrazione
def extract(original, image):
    """Prototipo di estrazione non-blind (andra' copiato nella detection).
    W_i = [(x1/c1 - 1) - (x2/c2 - 1)] / (1/|c1| + 1/|c2|)
    - immagine watermarkata  -> W ~ a * w
    - stesso guadagno g su entrambi i coefficienti -> W ~ g * a * w
    - immagine originale     -> W = 0 esatto
    """
    c0 = emb._dct2(original.astype(np.float64))
    uu, vv = emb._select_pairs(c0)
    c1 = emb._dct2(image.astype(np.float64))
    a1, a2 = c0[uu, vv], c0[uu, vv + 1]
    r1 = c1[uu, vv] / a1 - 1.0
    r2 = c1[uu, vv + 1] / a2 - 1.0
    return (r1 - r2) / (1.0 / np.abs(a1) + 1.0 / np.abs(a2))


def similarity(x, y):
    nx, ny = np.linalg.norm(x), np.linalg.norm(y)
    if nx == 0 or ny == 0:
        return 0.0
    return float(np.dot(x, y) / (nx * ny))


# ------------------------------------------------------------------ attacchi
def u8(x):
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def jpeg(img, qf):
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, "JPEG", quality=int(qf))
    buf.seek(0)
    return np.asarray(Image.open(buf), dtype=np.uint8)


def resize(img, scale):
    h, w = img.shape
    small = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


ATTACKS = [
    ("awgn sigma=10", lambda x: u8(x + np.random.default_rng(1).normal(0, 10, x.shape))),
    ("awgn sigma=20", lambda x: u8(x + np.random.default_rng(1).normal(0, 20, x.shape))),
    ("blur sigma=1", lambda x: u8(gaussian_filter(x.astype(float), 1.0))),
    ("blur sigma=2", lambda x: u8(gaussian_filter(x.astype(float), 2.0))),
    ("sharp s=1 a=1", lambda x: u8(x + 1.0 * (x - gaussian_filter(x.astype(float), 1.0)))),
    ("sharp s=10 a=0.3", lambda x: u8(x + 0.3 * (x - gaussian_filter(x.astype(float), 10.0)))),
    ("jpeg QF=50", lambda x: jpeg(x, 50)),
    ("jpeg QF=20", lambda x: jpeg(x, 20)),
    ("jpeg QF=10", lambda x: jpeg(x, 10)),
    ("resize 0.5", lambda x: resize(x, 0.5)),
    ("resize 0.3", lambda x: resize(x, 0.3)),
    ("median 3x3", lambda x: median_filter(x, 3)),
    ("median 5x5", lambda x: median_filter(x, 5)),
    ("median 7x7", lambda x: median_filter(x, 7)),
    ("jpeg50 + blur1", lambda x: u8(gaussian_filter(jpeg(x, 50).astype(float), 1.0))),
]


def quality_points(q):
    for thr, p in [(66, 6), (62, 5), (58, 4), (54, 3), (50, 2), (35, 1)]:
        if q >= thr:
            return p
    return 0


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else "."
    wm_path = sys.argv[2] if len(sys.argv) > 2 else GROUP + ".npy"
    tau = float(sys.argv[3]) if len(sys.argv) > 3 else 0.10
    out_dir = "watermarked"
    os.makedirs(out_dir, exist_ok=True)
    bits = np.load(wm_path).astype(np.float64)
    w_true = 2 * bits - 1

    files = sorted(glob.glob(os.path.join(folder, "*.bmp")))
    print(f"{len(files)} immagini, tau provvisoria = {tau}")
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        t0 = time.time()
        wm = emb.embedding(f, wm_path)
        dt = time.time() - t0
        cv2.imwrite(os.path.join(out_dir, f"{GROUP}_{name}.bmp"), wm)
        orig = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        q = emb.wpsnr(orig, wm)
        W_ext = extract(orig, wm)
        print(f"\n=== {name}: WPSNR = {q:.3f} dB -> {quality_points(q)} punti qualita'  ({dt:.1f}s)")
        print(f"  sim(W_ext, w_assegnato)    = {similarity(W_ext, w_true):.4f}   (deve essere ~1)")
        print(f"  sim(W_ext, W da originale) = {similarity(W_ext, extract(orig, orig)):.4f}   (deve essere 0)")
        rng = np.random.default_rng(0)
        h0 = np.array([similarity(W_ext, rng.choice([-1.0, 1.0], 1024)) for _ in range(1000)])
        print(f"  sim(W_ext, 1000 watermark casuali +-1): dev.std = {h0.std():.4f} (teoria 1/32 = 0.031), "
              f"max = {h0.max():.4f}")
        for an, f_att in ATTACKS:
            att = f_att(wm)
            wq = emb.wpsnr(wm, att)
            s = similarity(W_ext, extract(orig, att))
            status = "DISTRUTTO (attacco riuscito)" if (s < tau and wq >= 35) else (
                "rimosso ma WPSNR<35 (attacco fallito)" if s < tau else "presente")
            print(f"  {an:18s} WPSNR_att={wq:6.2f}  sim={s:+.3f}  -> {status}")


if __name__ == "__main__":
    main()
