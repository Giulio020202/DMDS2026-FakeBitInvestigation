"""
Funzioni comuni per gli esperimenti (NON vanno consegnate).

- wpsnr_fast: stesso numero del WPSNR dei lab (convoluzione circolare 'wrap'
  fatta via FFT invece che con convolve2d). Verificata in check_core.py.
- embed: stessa procedura di embedding_fakebitinvestigation.embedding, ma con
  parametri passabili (target, banda) e WPSNR veloce, per poter provare varianti.
- Extractor / cosine: estrazione non-blind e similarita'.
"""
import os
import sys

import numpy as np
from scipy.fft import dctn, idctn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "v2.0_ibrida"))   # versione consegnata: costanti e WPSNR dei lab
import embedding_fakebitinvestigation as emb  # noqa: E402

N = 512


# ----------------------------------------------------------------- WPSNR veloce
def _csf_transfer():
    k = emb._get_csf_filter()                 # 41x41, centro in (20,20)
    K = np.zeros((N, N))
    K[:41, :41] = k
    K = np.roll(K, (-20, -20), axis=(0, 1))   # centro in (0,0) -> conv. circolare 'same'
    return np.fft.fft2(K)


_H = _csf_transfer()
_H2 = np.abs(_H) ** 2


def wpsnr_fast(a, b):
    A = a.astype(np.float64) / 255.0
    B = b.astype(np.float64) / 255.0
    if np.array_equal(A, B):
        return 9999999.0
    E = np.fft.fft2(A - B)
    wmse = np.sum(np.abs(E) ** 2 * _H2) / (N * N) ** 2
    return 20 * np.log10(1.0 / np.sqrt(wmse))


# ------------------------------------------------------------------ embedding
def dct2(x):
    return dctn(x, norm='ortho')


def idct2(c):
    return idctn(c, norm='ortho')


def select_pairs(coeffs, r_min=emb.R_MIN, r_max=emb.R_MAX, n_bits=emb.N_BITS):
    H, W = coeffs.shape
    u = np.arange(H)[:, None]
    v = np.arange(0, W - 1, 2)[None, :]
    uu = np.broadcast_to(u, (H, v.shape[1])).ravel()
    vv = np.broadcast_to(v, (H, v.shape[1])).ravel()
    r = np.sqrt(uu**2 + (vv + 0.5)**2)
    band = (r >= r_min) & (r <= r_max)
    uu, vv = uu[band], vv[band]
    strength = np.minimum(np.abs(coeffs[uu, vv]), np.abs(coeffs[uu, vv + 1]))
    best = np.argsort(-strength, kind='stable')[:n_bits]
    best = np.sort(best)
    return uu[best], vv[best]


def to_image(c):
    return np.clip(np.rint(idct2(c)), 0, 255).astype(np.uint8)


def texture_mask(original, gamma, win=16, smooth=8.0):
    """Maschera spaziale >0 con media 1: piu' alta dove l'originale e' testurizzato.
    gamma = 0 -> maschera costante (schema consegnato)."""
    from scipy.ndimage import uniform_filter, gaussian_filter
    f = original.astype(np.float64)
    m = uniform_filter(f, win)
    sd = np.sqrt(np.maximum(uniform_filter(f * f, win) - m * m, 0)) + 2.0
    k = gaussian_filter(sd ** gamma, smooth)
    return k / k.mean()


def embed(original, w, target=emb.TARGET_WPSNR, r_min=emb.R_MIN, r_max=emb.R_MAX,
          amp_weights=None, gamma=0.0):
    """Embedding parametrico. amp_weights (opzionale, len 1024) moltiplica l'ampiezza
    per coppia; gamma > 0 modula il watermark nello spazio con texture_mask.
    Con i valori di default e' identico allo schema consegnato."""
    c = dct2(original.astype(np.float64))
    uu, vv = select_pairs(c, r_min, r_max)
    s1 = np.sign(c[uu, vv]); s1[s1 == 0] = 1
    s2 = np.sign(c[uu, vv + 1]); s2[s2 == 0] = 1
    ww = w if amp_weights is None else w * amp_weights

    if gamma:
        P = np.zeros_like(c)
        P[uu, vv] = s1 * ww
        P[uu, vv + 1] = -s2 * ww
        pattern = idct2(P) * texture_mask(original, gamma)
        base = original.astype(np.float64)

    def mark(a):
        if gamma:
            return np.clip(np.rint(base + a * pattern), 0, 255).astype(np.uint8)
        o = c.copy()
        o[uu, vv] += s1 * a * ww
        o[uu, vv + 1] -= s2 * a * ww
        return to_image(o)

    lo, hi = emb.AMP_MIN, emb.AMP_MAX
    best, best_a = mark(lo), lo
    for _ in range(emb.N_ITER):
        mid = np.sqrt(lo * hi)
        cand = mark(mid)
        if wpsnr_fast(original, cand) >= target:
            lo, best, best_a = mid, cand, mid
        else:
            hi = mid
    return best, best_a


# ------------------------------------------------------------------ estrazione
class Extractor:
    """Pre-calcola tutto quello che dipende solo dall'originale (come farebbe la detection)."""

    def __init__(self, original, r_min=emb.R_MIN, r_max=emb.R_MAX):
        self.c0 = dct2(original.astype(np.float64))
        self.uu, self.vv = select_pairs(self.c0, r_min, r_max)
        self.a1 = self.c0[self.uu, self.vv]
        self.a2 = self.c0[self.uu, self.vv + 1]
        # stessa forma "sicura" della detection: s*(X - C)/max(|C|, 1)
        self.s1 = np.where(self.a1 < 0, -1.0, 1.0)
        self.s2 = np.where(self.a2 < 0, -1.0, 1.0)
        self.m1 = np.maximum(np.abs(self.a1), 1.0)
        self.m2 = np.maximum(np.abs(self.a2), 1.0)
        self.den = 1.0 / self.m1 + 1.0 / self.m2

    def from_dct(self, c1):
        r1 = self.s1 * (c1[self.uu, self.vv] - self.a1) / self.m1
        r2 = self.s2 * (c1[self.uu, self.vv + 1] - self.a2) / self.m2
        return (r1 - r2) / self.den

    def __call__(self, img):
        return self.from_dct(dct2(img.astype(np.float64)))


def cosine(x, y):
    nx, ny = np.linalg.norm(x), np.linalg.norm(y)
    if nx == 0 or ny == 0:
        return 0.0
    return float(np.dot(x, y) / (nx * ny))
