"""
Variante WAVELET dello schema (solo per il confronto, non consegnata).
Stessa idea della versione DCT: coppie differenziali di coefficienti adiacenti,
ampiezza uniforme calibrata al WPSNR target, estrazione a rapporto (0 esatto sull'originale).
Cambia solo la trasformata: DWT ortogonale db4 (mode='periodization', vale Parseval),
coefficienti presi dalle sottobande di dettaglio dei livelli LEVELS (livello 3 ~ 0.06-0.125
cicli/pixel, livello 4 ~ 0.03-0.06: circa la stessa banda della versione DCT r 32-128).
"""
import numpy as np
import pywt

import core

WAVELET = "db4"
NLEV = 4


def fwd(img):
    coeffs = pywt.wavedec2(np.asarray(img, dtype=np.float64), WAVELET, mode="periodization", level=NLEV)
    arr, slices = pywt.coeffs_to_array(coeffs)
    return arr, slices


def inv(arr, slices):
    return pywt.waverec2(pywt.array_to_coeffs(arr, slices, output_format="wavedec2"), WAVELET, mode="periodization")


def candidate_mask(slices, levels):
    """Maschera (sull'array dei coefficienti) delle sottobande di dettaglio dei livelli scelti.
    In wavedec2 l'indice 1 e' il livello piu' grossolano (NLEV), l'ultimo il livello 1."""
    shape = (512, 512)
    m = np.zeros(shape, bool)
    for lev in levels:
        k = NLEV - lev + 1
        for key in ("ad", "da", "dd"):
            m[slices[k][key]] = True
    return m


def select_pairs(arr, slices, levels, n_bits=1024):
    m = candidate_mask(slices, levels)
    # coppie orizzontali (i,j),(i,j+1) con j pari, entrambe dentro la stessa sottobanda
    ii, jj = np.nonzero(m[:, :-1] & m[:, 1:])
    keep = jj % 2 == 0
    ii, jj = ii[keep], jj[keep]
    strength = np.minimum(np.abs(arr[ii, jj]), np.abs(arr[ii, jj + 1]))
    best = np.sort(np.argsort(-strength, kind="stable")[:n_bits])
    return ii[best], jj[best]


def embed(original, w, target=62.05, levels=(3, 4)):
    arr, sl = fwd(original)
    ii, jj = select_pairs(arr, sl, levels)
    s1 = np.sign(arr[ii, jj]); s1[s1 == 0] = 1
    s2 = np.sign(arr[ii, jj + 1]); s2[s2 == 0] = 1

    def mark(a):
        o = arr.copy()
        o[ii, jj] += s1 * a * w
        o[ii, jj + 1] -= s2 * a * w
        return np.clip(np.rint(inv(o, sl)), 0, 255).astype(np.uint8)

    lo, hi = 0.2, 40.0
    best, best_a = mark(lo), lo
    for _ in range(14):
        mid = np.sqrt(lo * hi)
        cand = mark(mid)
        if core.wpsnr_fast(original, cand) >= target:
            lo, best, best_a = mid, cand, mid
        else:
            hi = mid
    return best, best_a


class Extractor:
    def __init__(self, original, levels=(3, 4)):
        self.c0, self.sl = fwd(original)
        self.ii, self.jj = select_pairs(self.c0, self.sl, levels)
        self.a1 = self.c0[self.ii, self.jj]
        self.a2 = self.c0[self.ii, self.jj + 1]
        self.den = 1.0 / np.abs(self.a1) + 1.0 / np.abs(self.a2)

    def __call__(self, img):
        c1, _ = fwd(img)
        r1 = c1[self.ii, self.jj] / self.a1 - 1.0
        r2 = c1[self.ii, self.jj + 1] / self.a2 - 1.0
        return (r1 - r2) / self.den
