"""
Capture the Mark 2026 - attacchi del gruppo fakebitinvestigation (BOZZA, 19/10)

Solo gli attacchi ammessi dal regolamento (p.6): AWGN, blur, sharpening, JPEG, resize, median,
anche combinati (catene) o localizzati (su una zona dell'immagine).

Interfaccia (struttura d'esempio del regolamento, p.6):
    output1 = attacks(input1, attack_name, param_array)
  input1      = percorso dell'immagine watermarkata (oppure gia' un array uint8 512x512)
  attack_name = nome di un attacco ('awgn', 'blur', 'sharp', 'jpeg', 'resize', 'median')
                oppure una LISTA di nomi = catena, applicati nell'ordine dato
  param_array = parametri dell'attacco (lista); per una catena, una lista di liste
  mask        = (facoltativo) maschera booleana 512x512: l'attacco viene applicato solo li',
                fuori resta la watermarkata. Le maschere si calcolano SOLO dalla watermarkata
                (regolamento aggiornato: l'originale non si usa per progettare, scegliere,
                regolare, localizzare o eseguire un attacco).
  output1     = immagine attaccata, array uint8 512x512

Parametri di ciascun attacco:
  'awgn'   [std, seed]        rumore gaussiano, deviazione standard in livelli di grigio (seed facoltativo)
  'blur'   [sigma]            filtro gaussiano; sigma puo' essere una coppia (sy, sx) = blur anisotropo
  'sharp'  [sigma, alpha]     unsharp masking: img + alpha * (img - blur(img, sigma))
  'jpeg'   [QF]               compressione JPEG con fattore di qualita' 1..100
  'resize' [scale, metodo]    rimpicciolisce di 'scale' e ritorna a 512x512; metodo = 'lab' (default,
                              skimage come nei lab), 'cubic', 'linear' o 'area' (OpenCV)
  'median' [k]                filtro mediano k x k (oppure coppia (ky, kx)), bordi riflessi

Esempi:
    attacks("gruppoX_lena.bmp", "jpeg", [20])
    attacks("gruppoX_lena.bmp", ["blur", "jpeg"], [[0.6], [12]])
    attacks(wm, "awgn", [30], mask=smooth_mask(wm, 0.5))     # rumore forte solo sul 50% piu' liscio

Le funzioni base sono quelle dei lab (LAB-2 / LAB3-ROC) con due differenze, necessarie per lavorare
su uint8 come un vero BMP: l'ingresso diventa float prima di blur e sharpening (con uint8 lo
sharpening del lab va in overflow) e l'uscita di ogni attacco e' arrotondata e saturata a uint8.
"""
import io

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, median_filter, uniform_filter
from skimage.transform import rescale


def _u8(x):
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------
# ATTACCHI BASE
# ----------------------------------------------------------------------------
def awgn(img, std, seed=123):
    rng = np.random.default_rng(seed)
    return _u8(img.astype(float) + rng.normal(0.0, std, img.shape))


def blur(img, sigma):
    return _u8(gaussian_filter(img.astype(float), sigma))


def sharpening(img, sigma, alpha):
    f = img.astype(float)
    return _u8(f + alpha * (f - gaussian_filter(f, sigma)))


def jpeg_compression(img, qf):
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, "JPEG", quality=int(qf))
    buf.seek(0)
    return np.asarray(Image.open(buf), dtype=np.uint8)


def resizing(img, scale):
    """Resize dei lab: rescale giu' e su con skimage, poi taglio a 512x512.
    Se esce piu' piccola (scale tipo 0.7) replichiamo l'ultima riga/colonna, cosi' il WPSNR
    si puo' sempre calcolare."""
    x, y = img.shape
    a = rescale(img.astype(float), scale)
    a = rescale(a, 1 / scale)[:x, :y]
    if a.shape != (x, y):
        a = np.pad(a, ((0, x - a.shape[0]), (0, y - a.shape[1])), mode="edge")
    return _u8(a)


_CV_INTERP = {"cubic": cv2.INTER_CUBIC, "linear": cv2.INTER_LINEAR, "area": cv2.INTER_AREA}


def resizing_cv(img, scale, method="cubic"):
    """Resize con OpenCV: giu' con INTER_AREA, su con il metodo scelto."""
    h, w = img.shape
    s = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    return _u8(cv2.resize(s, (w, h), interpolation=_CV_INTERP[method]))


def median(img, k):
    return median_filter(img, size=k)


ATTACKS = {
    "awgn": lambda img, p: awgn(img, *p),
    "blur": lambda img, p: blur(img, p[0]),
    "sharp": lambda img, p: sharpening(img, p[0], p[1]),
    "jpeg": lambda img, p: jpeg_compression(img, p[0]),
    "resize": lambda img, p: resizing(img, p[0]) if len(p) < 2 or p[1] == "lab" else resizing_cv(img, p[0], p[1]),
    "median": lambda img, p: median(img, p[0]),
}


# ----------------------------------------------------------------------------
# MASCHERE (calcolate SOLO dalla watermarkata)
# ----------------------------------------------------------------------------
def _block_std(img, b):
    f = img.astype(float)
    m = uniform_filter(f, b)
    return np.sqrt(np.maximum(uniform_filter(f * f, b) - m * m, 0))


def smooth_mask(wm, q=0.5, b=16):
    """La frazione q dell'immagine piu' liscia (deviazione standard locale piu' bassa).
    Nei nostri test (batteria v2) i migliori attacchi erano localizzati su queste zone."""
    sd = _block_std(wm, b)
    return sd <= np.quantile(sd, q)


def textured_mask(wm, q=0.5, b=16):
    """La frazione q dell'immagine piu' testurizzata."""
    sd = _block_std(wm, b)
    return sd >= np.quantile(sd, 1 - q)


def edge_mask(wm, q=0.3):
    """La frazione q dell'immagine con i bordi piu' forti (gradiente di Sobel)."""
    f = wm.astype(float)
    g = gaussian_filter(np.hypot(cv2.Sobel(f, cv2.CV_64F, 1, 0), cv2.Sobel(f, cv2.CV_64F, 0, 1)), 2)
    return g >= np.quantile(g, 1 - q)


def smooth_blocks8_mask(wm, q=0.5):
    """Come smooth_mask ma a blocchi 8x8 interi, allineati alla griglia del JPEG
    (un JPEG localizzato cosi' non lascia blocchi "a meta'")."""
    sd8 = _block_std(wm, 8)[4::8, 4::8]
    return np.kron(sd8 <= np.quantile(sd8, q), np.ones((8, 8), bool))


MASKS = {
    "lisce30": lambda wm: smooth_mask(wm, 0.3),
    "lisce50": lambda wm: smooth_mask(wm, 0.5),
    "lisce70": lambda wm: smooth_mask(wm, 0.7),
    "lisceblk50": lambda wm: smooth_blocks8_mask(wm, 0.5),
    "text30": lambda wm: textured_mask(wm, 0.3),
    "text50": lambda wm: textured_mask(wm, 0.5),
    "bordi30": lambda wm: edge_mask(wm, 0.3),
}


# ----------------------------------------------------------------------------
# FUNZIONE PRINCIPALE
# ----------------------------------------------------------------------------
def attacks(input1, attack_name, param_array, mask=None):
    wm = cv2.imread(input1, 0) if isinstance(input1, str) else np.asarray(input1, dtype=np.uint8)
    if isinstance(attack_name, str):
        names, params = [attack_name], [param_array]
    else:
        names, params = list(attack_name), list(param_array)
    if len(names) != len(params):
        raise ValueError("serve una lista di parametri per ogni attacco della catena")

    out = wm
    for name, p in zip(names, params):
        if name not in ATTACKS:
            raise ValueError(f"attacco non ammesso: {name!r} (ammessi: {', '.join(ATTACKS)})")
        out = ATTACKS[name](out, list(p))

    if mask is not None:
        loc = wm.copy()
        loc[mask] = out[mask]
        out = loc
    return out


def describe(attack_name, param_array, mask_name=None):
    """Testo per il log di gara, es. 'blur [0.6] + jpeg [12] su lisce50'."""
    if isinstance(attack_name, str):
        attack_name, param_array = [attack_name], [param_array]
    s = " + ".join(f"{n} {list(p)}" for n, p in zip(attack_name, param_array))
    return f"{s} su {mask_name}" if mask_name else s
