"""
Batteria d'attacco v2 (solo attacchi permessi a p.6: AWGN, blur, sharpening, JPEG,
resize, median; singoli, combinati, globali o localizzati).

Le localizzazioni usano SOLO l'immagine watermarkata (vietato usare l'originale, p.3/p.6).
Le funzioni base sono quelle dei lab (LAB-2 01-Image-processing / LAB3-ROC), con due
differenze necessarie per lavorare su uint8 come farebbe un vero BMP:
  - gli ingressi vengono convertiti in float prima di blur/sharpening (con uint8 la
    versione del lab andrebbe in overflow nello sharpening);
  - l'uscita di ogni attacco viene arrotondata e saturata a uint8.
"""
import io

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, median_filter, uniform_filter
from scipy.signal import medfilt
from skimage.transform import rescale


def u8(x):
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


# ------------------------------------------------------------ attacchi base (lab)
def awgn(img, std, seed=123):
    rng = np.random.default_rng(seed)
    return u8(img.astype(float) + rng.normal(0.0, std, img.shape))


def blur(img, sigma):
    return u8(gaussian_filter(img.astype(float), sigma))


def sharpening(img, sigma, alpha):
    f = img.astype(float)
    return u8(f + alpha * (f - gaussian_filter(f, sigma)))


def median(img, k):
    """medfilt come nei lab (bordi a zero)."""
    return u8(medfilt(img.astype(float), k))


def median_nd(img, k):
    """median con bordi riflessi (variante che un attaccante attento userebbe)."""
    return median_filter(img, size=k)


def resizing(img, scale):
    """Resize del lab: rescale giu' e su con skimage, poi taglio [:x,:y].
    Se esce piu' piccolo di 512 (scale tipo 0.7) replichiamo l'ultima riga/colonna,
    altrimenti il WPSNR non si potrebbe calcolare (e' quello che farebbe un attaccante)."""
    x, y = img.shape
    a = rescale(img.astype(float), scale)
    a = rescale(a, 1 / scale)
    a = a[:x, :y]
    if a.shape != (x, y):
        a = np.pad(a, ((0, x - a.shape[0]), (0, y - a.shape[1])), mode='edge')
    return u8(a)


def resize_cv(img, scale, up=cv2.INTER_CUBIC):
    h, w = img.shape
    s = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    return u8(cv2.resize(s, (w, h), interpolation=up))


def jpeg(img, qf):
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, "JPEG", quality=int(qf))
    buf.seek(0)
    return np.asarray(Image.open(buf), dtype=np.uint8)


# ------------------------------------------------------------------ maschere
def block_std(img, b=16):
    f = img.astype(float)
    m = uniform_filter(f, b)
    v = uniform_filter(f * f, b) - m * m
    return np.sqrt(np.maximum(v, 0))


def masks(wm):
    """Maschere calcolate solo dalla watermarkata. Restituisce dict nome -> bool 512x512."""
    out = {}
    sd = block_std(wm, 16)
    for q in (0.3, 0.5, 0.7):
        out[f"lisce{int(q * 100)}"] = sd <= np.quantile(sd, q)      # il q% piu' liscio
        out[f"text{int(q * 100)}"] = sd >= np.quantile(sd, 1 - q)   # il q% piu' testurizzato
    f = wm.astype(float)
    g = gaussian_filter(np.hypot(cv2.Sobel(f, cv2.CV_64F, 1, 0), cv2.Sobel(f, cv2.CV_64F, 0, 1)), 2)
    out["bordi30"] = g >= np.quantile(g, 0.7)
    # zone lisce a blocchi 8x8 interi, allineati alla griglia del JPEG
    sd8 = block_std(wm, 8)[4::8, 4::8]
    out["lisceblk50"] = np.kron(sd8 <= np.quantile(sd8, 0.5), np.ones((8, 8), bool))
    return out


def localize(wm, att, mask):
    o = wm.copy()
    o[mask] = att[mask]
    return o


# ------------------------------------------------------------------ batteria
def battery(wm):
    """Genera (famiglia, nome, immagine attaccata) per tutti gli attacchi della batteria."""
    # --- globali singoli
    for q in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90]:
        yield "jpeg", f"jpeg QF={q}", jpeg(wm, q)
    for s in [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 25, 28, 32]:
        yield "awgn", f"awgn s={s}", awgn(wm, s)
    for s in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0]:
        yield "blur", f"blur s={s}", blur(wm, s)
        yield "blur", f"blur aniso s=({s},{s / 3:.2f})", blur(wm, (s, s / 3))
        yield "blur", f"blur aniso s=({s / 3:.2f},{s})", blur(wm, (s / 3, s))
    for s in [0.5, 1, 2, 4, 10, 20]:
        for a in [0.2, 0.5, 1, 2, 3]:
            yield "sharp", f"sharp s={s} a={a}", sharpening(wm, s, a)
    for k in [3, 5, 7, 9, 11, 13, (1, 3), (3, 1), (1, 5), (5, 1), (1, 7), (7, 1), (3, 5), (5, 3),
              (1, 9), (9, 1), (3, 7), (7, 3)]:
        yield "median", f"medfilt k={k}", median(wm, k)
        yield "median", f"median_nd k={k}", median_nd(wm, k)
    for s in np.round(np.arange(0.15, 0.96, 0.05), 2):
        yield "resize", f"resize lab s={s}", resizing(wm, s)
        yield "resize", f"resize cv2 s={s}", resize_cv(wm, s)
        yield "resize", f"resize cv2lin s={s}", resize_cv(wm, s, cv2.INTER_LINEAR)

    # --- combinazioni
    for s in [4, 8, 12, 16, 20, 25, 30, 40]:
        for b in [0.5, 0.8, 1.0, 1.5, 2.0, 3.0]:
            yield "awgn+blur", f"awgn s={s} + blur {b}", blur(awgn(wm, s), b)
    for q in [3, 5, 7, 10, 15, 20, 30, 50]:
        for b in [0.5, 0.8, 1.2]:
            yield "jpeg+blur", f"jpeg {q} + blur {b}", blur(jpeg(wm, q), b)
            yield "blur+jpeg", f"blur {b} + jpeg {q}", jpeg(blur(wm, b), q)
        for k in [3, 5]:
            yield "median+jpeg", f"median {k} + jpeg {q}", jpeg(median_nd(wm, k), q)
            yield "jpeg+median", f"jpeg {q} + median {k}", median_nd(jpeg(wm, q), k)
    for s in [0.3, 0.4, 0.5, 0.55, 0.6, 0.7]:
        for q in [10, 20, 40, 70]:
            yield "resize+jpeg", f"resize lab {s} + jpeg {q}", jpeg(resizing(wm, s), q)
    for b in [1, 2, 3, 4]:
        for s in [2, 4, 10]:
            for a in [0.5, 1, 2, 3]:
                # blur + sharpening a sigma piu' larga: riporta su le basse frequenze
                # (WPSNR alto) e lascia attenuate le medie, dove stiamo noi
                yield "blur+sharp", f"blur {b} + sharp s={s} a={a}", sharpening(blur(wm, b), s, a)
    for q in [5, 10, 20, 40]:
        for s, a in [(1, 1), (2, 1), (4, 1), (2, 2)]:
            yield "jpeg+sharp", f"jpeg {q} + sharp s={s} a={a}", sharpening(jpeg(wm, q), s, a)
    for k in [5, 7, 9]:
        for s, a in [(2, 1), (4, 1), (4, 2)]:
            yield "median+sharp", f"median {k} + sharp s={s} a={a}", sharpening(median_nd(wm, k), s, a)
    for s in [5, 10, 15, 20]:
        for k in [3, 5]:
            yield "awgn+median", f"awgn {s} + median {k}", median_nd(awgn(wm, s), k)
    for s in [0.4, 0.5, 0.55, 0.7]:
        for b in [0.5, 1.0]:
            for q in [10, 30]:
                yield "resize+blur+jpeg", f"resize lab {s} + blur {b} + jpeg {q}", jpeg(blur(resizing(wm, s), b), q)

    # --- localizzati (maschere dalla sola watermarkata)
    for mname, m in masks(wm).items():
        for q in [2, 3, 4, 5, 6, 8, 10, 15, 25]:
            yield "loc-jpeg", f"jpeg {q} su {mname}", localize(wm, jpeg(wm, q), m)
        for s in [10, 20, 30, 40, 60]:
            yield "loc-awgn", f"awgn {s} su {mname}", localize(wm, awgn(wm, s), m)
        for s in [1, 2, 3, 5]:
            yield "loc-blur", f"blur {s} su {mname}", localize(wm, blur(wm, s), m)
        for k in [5, 7, 9, 13]:
            yield "loc-median", f"median {k} su {mname}", localize(wm, median_nd(wm, k), m)
        for s in [0.2, 0.3, 0.55]:
            yield "loc-resize", f"resize lab {s} su {mname}", localize(wm, resizing(wm, s), m)
        for s, b in [(20, 1), (30, 1.5), (40, 2)]:
            yield "loc-awgn+blur", f"awgn {s}+blur {b} su {mname}", localize(wm, blur(awgn(wm, s), b), m)
        for b, s, a in [(2, 4, 2), (3, 10, 2)]:
            yield "loc-blur+sharp", f"blur {b}+sharp {s},{a} su {mname}", localize(wm, sharpening(blur(wm, b), s, a), m)


def battery_v3_extra(wm):
    """(14/10) Attacchi pensati CONTRO la detection con predizione dell'attacco: rumore prima
    del JPEG (rende il JPEG non piu' prevedibile), doppio JPEG, parametri fuori dalla griglia
    del predittore, catene di tre attacchi."""
    for s in [1, 2, 3, 5, 8]:
        for q in [5, 8, 12, 20]:
            yield "anti-awgn+jpeg", f"awgn {s} + jpeg {q}", jpeg(awgn(wm, s), q)
    for q1 in [10, 20, 30, 50]:
        for q2 in [5, 7, 10, 15]:
            yield "anti-2jpeg", f"jpeg {q1} + jpeg {q2}", jpeg(jpeg(wm, q1), q2)
    for b in [0.65, 0.95, 1.15, 1.4]:
        for q in [5, 8, 12, 20]:
            yield "anti-blur+jpeg", f"blur {b} + jpeg {q}", jpeg(blur(wm, b), q)
    for q in [5, 10, 20]:
        for b in [0.65, 1.15]:
            yield "anti-jpeg+blur", f"jpeg {q} + blur {b}", blur(jpeg(wm, q), b)
        yield "anti-sharp+jpeg", f"sharp 1,0.5 + jpeg {q}", jpeg(sharpening(wm, 1, 0.5), q)
        yield "anti-3", f"median 3 + awgn 2 + jpeg {q}", jpeg(awgn(median_nd(wm, 3), 2), q)
        yield "anti-3", f"blur 0.8 + jpeg {q} + blur 0.6", blur(jpeg(blur(wm, 0.8), q), 0.6)
        yield "anti-3", f"resize lab 0.5 + jpeg {q} + median 3", median_nd(jpeg(resizing(wm, 0.5), q), 3)
    m = masks(wm)["lisce50"]
    for s in [2, 4]:
        for q in [3, 5, 8]:
            yield "anti-loc", f"awgn {s} + jpeg {q} su lisce50", localize(wm, jpeg(awgn(wm, s), q), m)


def battery_v3(wm):
    yield from battery(wm)
    yield from battery_v3_extra(wm)
