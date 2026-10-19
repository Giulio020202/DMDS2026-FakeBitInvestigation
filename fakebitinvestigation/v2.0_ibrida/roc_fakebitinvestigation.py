"""
Capture the Mark 2026 - calcolo della soglia tau (ROC) del gruppo fakebitinvestigation
(versione 2.0, IBRIDA DCT + wavelet: W ha 2048 valori, 1024 DCT + 1024 wavelet).

Segue la procedura di p.4 del regolamento (Lab4, ROC-curves, esempio 2), con i controlli in piu' del Lab 4
(H0-A, H0-B e set di test separato):
  1. embedding del nostro watermark W_original su un insieme di immagini;
  2. in un ciclo, attacchi casuali (solo quelli permessi: awgn, blur, sharpening, jpeg,
     resize, median, anche combinati o localizzati) sulle immagini watermarkate;
  3. estrazione con la stessa funzione della detection;
  4. sim(W_extracted dalla watermarkata, W estratto dall'attaccata) -> score con label 1 (H1)
     (e' esattamente la similarita' calcolata dalla detection);
  5. score con label 0, con la STESSA funzione di similarita' della detection (massimo sulle varianti):
     H0-R: per ogni attacco, N_RANDOM watermark casuali W_random (in forma +-1, la stessa mappatura
           dell'embedding, messo in entrambe le parti con l'ampiezza di ciascuna parte presa da W_extracted)
           confrontati con il W estratto dall'attaccata (e' la procedura del regolamento);
     H0-A: lo STESSO attacco applicato all'originale non watermarkata (Lab 4);
     H0-B: lo STESSO attacco applicato alla stessa immagine con il watermark di un "altro gruppo"
           (stesso schema, watermark casuale) (Lab 4);
  6. le immagini sono divise in CALIBRAZIONE (70%) e TEST (30%, mai usate per scegliere tau);
     sulla calibrazione: curva ROC (sklearn) e tau = la soglia piu' bassa con FPR <= FPR_TARGET
     su CIASCUNO dei tre tipi di H0 (dentro l'intervallo ammesso [0, 0.1]);
     sul test: tau congelata, si misurano FPR (per tipo) e TPR.

Questo codice NON fa parte della detection (p.4: la soglia si calcola una volta sola, fuori).

Uso (da qualunque cartella):
    python roc_fakebitinvestigation.py [--images DIR] [--watermark FILE.npy] [--attacks N] [--quick] [--out FILE.png]
  --images     cartella con le immagini BMP 512x512 (default: ../sample-images, 101 immagini)
  --watermark  watermark .npy (default: ../fakebitinvestigation.npy)
  --attacks    attacchi casuali per immagine (default 40)
  --quick      prova veloce: solo le prime 10 immagini e 5 attacchi ciascuna; la tau che esce NON e'
               quella vera (troppi pochi campioni), serve solo a vedere che tutto gira
  --out        dove salvare la curva (default: roc_fakebitinvestigation.png accanto a questo file)
Output: tau stampata a video + curva .png.
Dipendenze in piu' rispetto a embedding/detection: scikit-learn, matplotlib, pillow, scikit-image.
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
WATERMARK = os.path.join(HERE, "..", "fakebitinvestigation.npy")   # watermark e immagini sono nella cartella superiore
FPR_TARGET = 0.001      # FPR scelta per tau (il regolamento ammette [0, 0.1])
N_RANDOM = 10           # watermark casuali per ogni attacco (campioni con label 0)
SEED = 2026
TEST_EVERY = 10          # immagini con indice % 10 >= 7 -> set di TEST (30%), le altre -> CALIBRAZIONE


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


def random_attack(rng):
    """Sceglie un attacco casuale tra quelli permessi e restituisce (funzione, nome): la stessa
    funzione si applica a piu' immagini (watermarkata, originale, watermarkata da un altro gruppo)."""
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
    if k == 8:   # localizzato sulle zone lisce (maschera calcolata dall'immagine attaccata stessa)
        qm, q = rng.uniform(0.3, 0.7), int(rng.integers(2, 20))

        def f(x):
            m = _smooth_mask(x, qm)
            out = x.copy()
            out[m] = jpeg_compression(x, q)[m]
            return out
        return f, "jpeg locale"
    m = int(rng.choice([3, 5, 7])); return (lambda x: median_filter(x, size=m)), "median_nd"


# ----------------------------------------------------------------- un'immagine
def _one_image(args):
    path, n_att, seed, watermark = args
    rng = np.random.default_rng(seed)
    original = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    watermarked = emb.embedding(path, watermark)
    # "altro gruppo": stessa immagine, stesso schema, watermark casuale (file temporaneo per embedding())
    with tempfile.TemporaryDirectory() as tmp:
        other_wm = os.path.join(tmp, "altro.npy")
        np.save(other_wm, rng.integers(0, 2, emb.N_BITS))
        other = emb.embedding(path, other_wm)
    ctx = det._context(original)
    W_ext = det._extract(ctx, watermarked)
    half = W_ext.size // 2
    # ampiezza (rms) di ciascuna parte: il watermark casuale viene messo nella stessa forma di W_ext
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


def _tau(h0, fpr_target):
    """La soglia piu' bassa con FPR <= fpr_target: sopra soglia al massimo floor(fpr_target * n) campioni H0."""
    h0s = np.sort(h0)
    m = int(np.floor(fpr_target * len(h0s)))
    return float(np.nextafter(h0s[len(h0s) - m - 1], np.inf))


def main():
    ap = argparse.ArgumentParser(description="ROC e soglia tau del gruppo fakebitinvestigation (v2.0)")
    ap.add_argument("--images", default=os.path.join(HERE, "..", "sample-images"))
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
        sys.exit(f"nessuna immagine .bmp in {a.images}")
    t0 = time.time()
    with Pool(max(1, (os.cpu_count() or 2) - 1)) as p:
        res, res_idx = [], []
        jobs = [(f, n_att, SEED + i, a.watermark) for i, f in enumerate(files)]
        for i, r in enumerate(p.imap(_one_image, jobs, chunksize=1)):
            res.append(r); res_idx.append(i)
            k = len(res)                     # barra di avanzamento (su stderr)
            el = time.time() - t0
            sys.stderr.write(f"\r[{'#' * (30 * k // len(files)):30s}] {k}/{len(files)} immagini, "
                             f"{el / 60:.1f} min, mancano ~{el / k * (len(files) - k) / 60:.1f} min ")
            sys.stderr.flush()
        sys.stderr.write("\n")
    # res_idx[j] = indice dell'immagine del risultato j: decide se e' di calibrazione o di test
    test_set = {i for i in range(len(files)) if i % TEST_EVERY >= 7}
    print(f"{len(files)} immagini ({len(files) - len(test_set)} calibrazione, {len(test_set)} test), {n_att} attacchi "
          f"ciascuna, {N_RANDOM} watermark casuali per attacco, {time.time() - t0:.0f}s")

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
    print(f"CALIBRAZIONE: {len(h1)} campioni H1, H0-R {np.sum(kd == 'H0-R')}, H0-A {np.sum(kd == 'H0-A')}, "
          f"H0-B {np.sum(kd == 'H0-B')}; AUC (H0 insieme) = {auc(fpr, tpr):.5f}")
    for k, t in taus.items():
        x = sc[kd == k]
        print(f"  {k}: media {x.mean():+.4f}, dev.std {x.std():.4f}, quantile 99.9% {np.quantile(x, 0.999):.4f}, "
              f"max {x.max():.4f} -> soglia per FPR <= {FPR_TARGET}: {t:.4f}")
    valid = (kd == 'H1') & (wp >= 35)
    print(f"tau = massimo delle tre = {tau:.4f}  ->  FPR H0-R {np.mean(sc[kd == 'H0-R'] >= tau):.5f}, "
          f"H0-A {np.mean(sc[kd == 'H0-A'] >= tau):.5f}, H0-B {np.mean(sc[kd == 'H0-B'] >= tau):.5f}; "
          f"TPR {np.mean(h1 >= tau):.4f}, TPR sui soli attacchi con WPSNR >= 35: {np.mean(sc[valid] >= tau):.4f}")
    sc2, kd2, wp2 = gather(tst)
    valid2 = (kd2 == 'H1') & (wp2 >= 35)
    print(f"TEST (tau congelata {tau:.4f}): FPR H0-R {np.mean(sc2[kd2 == 'H0-R'] >= tau):.5f}, "
          f"H0-A {np.mean(sc2[kd2 == 'H0-A'] >= tau):.5f}, H0-B {np.mean(sc2[kd2 == 'H0-B'] >= tau):.5f}; "
          f"TPR {np.mean(sc2[kd2 == 'H1'] >= tau):.4f}, TPR sui soli attacchi con WPSNR >= 35: {np.mean(sc2[valid2] >= tau):.4f}")
    for t in (0.08, 0.086, 0.09, 0.10):
        print(f"  tau={t:.3f} (calibrazione): FPR H0-R {np.mean(sc[kd == 'H0-R'] >= t):.5f}, "
              f"H0-A {np.mean(sc[kd == 'H0-A'] >= t):.5f}, H0-B {np.mean(sc[kd == 'H0-B'] >= t):.5f}, TPR {np.mean(h1 >= t):.4f}")

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
    plt.savefig(a.out, dpi=120, bbox_inches="tight")
    print(f"curva salvata in {a.out}")


if __name__ == "__main__":
    main()
