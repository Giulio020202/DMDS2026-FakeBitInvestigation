"""
Capture the Mark 2026 - attacchi del gruppo fakebitinvestigation (FILE UNICO DEGLI ATTACCHI, da consegnare)

Contiene TUTTO il nostro codice di attacco, senza dipendere da altri file nostri:
  1. gli attacchi ammessi dal regolamento (p.6): AWGN, blur, sharpening, JPEG, resize, median, anche combinati
     (catene) o localizzati (su una zona), con la funzione attacks() nella struttura d'esempio del regolamento;
  2. le zone per gli attacchi localizzati, calcolate SOLO dalla watermarkata;
  3. la ricerca automatica dell'attacco piu' leggero che rompe un'immagine (attack_image): famiglie di
     attacchi dal piu' leggero al piu' forte, la detection del gruppo attaccato come "oracolo" (ammesso dai
     TA), poi il greedy a blocchi che restituisce alla watermarkata le zone che non servono (ammesso dai TA).
L'originale non viene mai letto ne' usato per costruire l'attacco: lo riceve solo la detection del gruppo
attaccato, come suo input1 (regolamento p.6 e risposta dei TA del 20/10).
Unica funzione "esterna": il WPSNR dei lab, copiato qui sotto.

Interfaccia (struttura d'esempio del regolamento, p.6):
    output1 = attacks(input1, attack_name, param_array)
  input1      = percorso dell'immagine watermarkata (oppure gia' un array uint8 512x512)
  attack_name = nome di un attacco ('awgn', 'blur', 'sharp', 'jpeg', 'resize', 'median')
                oppure una LISTA di nomi = catena, applicati nell'ordine dato
  param_array = parametri dell'attacco (lista); per una catena, una lista di liste
  mask        = (facoltativo) maschera booleana 512x512: l'attacco viene applicato solo li',
                fuori resta la watermarkata. Le maschere si calcolano SOLO dalla watermarkata.
  output1     = immagine attaccata, array uint8 512x512

Parametri di ciascun attacco:
  'awgn'   [std, seed]        rumore gaussiano, deviazione standard in livelli di grigio (seed facoltativo)
  'blur'   [sigma]            filtro gaussiano; sigma puo' essere una coppia (sy, sx) = blur anisotropo
  'sharp'  [sigma, alpha]     unsharp masking: img + alpha * (img - blur(img, sigma))
  'jpeg'   [QF]               compressione JPEG con fattore di qualita' 1..100
  'resize' [scale, metodo]    rimpicciolisce di 'scale' e ritorna a 512x512; metodo = 'lab' (default,
                              skimage come nei lab), 'cubic', 'linear' o 'area' (OpenCV)
  'median' [k]                filtro mediano k x k (oppure coppia (ky, kx)), bordi riflessi

Esempi da Python:
    attacks("gruppoX_lena.bmp", "jpeg", [20])
    attacks("gruppoX_lena.bmp", ["blur", "jpeg"], [[0.6], [12]])
    attacks(wm, "awgn", [30], mask=smooth_mask(wm, 0.5))     # rumore forte solo sul 50% piu' liscio
    attack_image("gruppoX_lena.bmp", "lena.bmp", "detection_gruppoX.py", "out/", budget=600)

Da riga di comando:
    # un attacco (o una catena) su un'immagine, salvato in OUT.bmp; stampa il WPSNR
    python attacks_fakebitinvestigation.py applica WATERMARKATA.bmp OUT.bmp jpeg 20
    python attacks_fakebitinvestigation.py applica WATERMARKATA.bmp OUT.bmp blur,jpeg 0.6 12
    # ricerca automatica dell'attacco piu' leggero che rompe UNA immagine, con la detection del gruppo attaccato
    python attacks_fakebitinvestigation.py cerca --image gruppoX_lena.bmp --original lena.bmp \
           --detection detection_gruppoX.py --out OUTDIR [--budget SECONDI] [--no-greedy]

Le funzioni base sono quelle dei lab (LAB-2 / LAB3-ROC) con due differenze, necessarie per lavorare
su uint8 come un vero BMP: l'ingresso diventa float prima di blur e sharpening (con uint8 lo
sharpening del lab va in overflow) e l'uscita di ogni attacco e' arrotondata e saturata a uint8.
"""
import argparse
import contextlib
import importlib.util
import io
import os
import signal
import sys
import tempfile
import time
import traceback

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, median_filter, uniform_filter
from scipy.signal import convolve2d
from skimage.transform import rescale


# ----------------------------------------------------------------------------
# WPSNR: copia identica del codice fornito ai laboratori
# ----------------------------------------------------------------------------
def _csffun(u, v):
    f = np.sqrt(u**2 + v**2)
    w = 2 * np.pi * f / 60
    sigma = 2
    Sw = 1.5 * np.exp(-sigma**2 * w**2 / 2) - np.exp(-2 * sigma**2 * w**2 / 2)
    sita = np.arctan2(v, u)
    bita = 8
    f0 = 11.13
    w0 = 2 * np.pi * f0 / 60
    exp_term = np.exp(bita * (w - w0))
    Ow = (1 + exp_term * (np.cos(2 * sita))**4) / (1 + exp_term)
    return Sw * Ow


def _csfmat():
    min_f, max_f, step_f = -20, 20, 1
    freq_range = np.arange(min_f, max_f + step_f, step_f)
    u, v = np.meshgrid(freq_range, freq_range, indexing='xy')
    return _csffun(u, v)


def _get_csf_filter():
    Fmat = _csfmat()
    filter_coeffs = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(Fmat)))
    return np.real(filter_coeffs)


def wpsnr(image_a, image_b):
    if image_a.shape != image_b.shape:
        raise ValueError("Input images must have the same dimensions.")
    if image_a.dtype != np.float64 and image_a.dtype != np.float32:
        A = image_a.astype(np.float64) / 255.0
    else:
        A = image_a.copy()
    if image_b.dtype != np.float64 and image_b.dtype != np.float32:
        B = image_b.astype(np.float64) / 255.0
    else:
        B = image_b.copy()
    if A.max() > 1.0 or A.min() < 0.0 or B.max() > 1.0 or B.min() < 0.0:
        raise ValueError("Input image values must be in the interval [0, 1] for floats or [0, 255] for integers.")
    if np.array_equal(A, B):
        return 9999999.0
    error_image = A - B
    csf_filter = _get_csf_filter()
    weighted_error = convolve2d(error_image, csf_filter, mode='same', boundary='wrap')
    wmse = np.mean(weighted_error**2)
    if wmse == 0:
        return 9999999.0
    return 20 * np.log10(1.0 / np.sqrt(wmse))


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


def _fmt(v):
    """Numero o coppia in forma leggibile: 0.8 -> '0.8', (1, 7) -> '1x7'."""
    if isinstance(v, (tuple, list)):
        return "x".join(_fmt(x) for x in v)
    return f"{v:g}" if isinstance(v, float) else str(v)


def _describe_one(name, p):
    if name == "jpeg":
        return f"JPEG QF={p[0]}"
    if name == "awgn":
        return f"AWGN std={_fmt(p[0])}" + (f" (seed {p[1]})" if len(p) > 1 else "")
    if name == "blur":
        return f"Blur sigma={_fmt(p[0])}"
    if name == "sharp":
        return f"Sharpening sigma={_fmt(p[0])} alpha={_fmt(p[1])}"
    if name == "resize":
        return f"Resize scale={_fmt(p[0])} ({p[1] if len(p) > 1 else 'lab'})"
    if name == "median":
        k = p[0]
        return f"Median {_fmt(k)}" if isinstance(k, (tuple, list)) else f"Median {k}x{k}"
    return f"{name} {list(p)}"


def describe(attack_name, param_array, mask_name=None):
    """Testo per il log di gara, nello stile dell'esempio del regolamento ('JPEG QF=90, Median 5x5'),
    con gli attacchi nell'ordine in cui sono applicati, es. 'Blur sigma=0.6, JPEG QF=12 (solo zona lisce50)'."""
    if isinstance(attack_name, str):
        attack_name, param_array = [attack_name], [param_array]
    s = ", ".join(_describe_one(n, list(p)) for n, p in zip(attack_name, param_array))
    return f"{s} (solo zona {mask_name}, calcolata dalla watermarkata)" if mask_name else s


# ----------------------------------------------------------------------------
# RICERCA AUTOMATICA DELL'ATTACCO PIU' LEGGERO (fase di attacco)
# ----------------------------------------------------------------------------
OUR_GROUP = "fakebitinvestigation"
MIN_WPSNR = 35.0
MAX_ERRORS = 3                 # errori di fila della detection altrui prima di saltare l'immagine
SEARCH_SHARE = 0.75            # quota del budget per la ricerca; il resto al greedy

QF = [95, 90, 80, 70, 60, 50, 40, 30, 25, 20, 16, 13, 10, 8, 6, 5, 4, 3, 2, 1]
# famiglia -> (catena di nomi, funzione forza -> param_array, griglia dal piu' LEGGERO al piu' FORTE, tipo)
# tipo: 'int' (QF) / 'float' / 'float-down' (resize: forza crescente = scala decrescente) si raffinano;
# 'none' = solo i valori della griglia.
# ORDINE = efficacia: prima le catene con JPEG (contro di noi erano gli attacchi vincenti nel 93% dei casi).
FAMILIES = {
    "jpeg+blur0.8": (["jpeg", "blur"], lambda p: [[p], [0.8]], QF[:-1], "int"),
    "blur0.6+jpeg": (["blur", "jpeg"], lambda p: [[0.6], [p]], QF[:-1], "int"),
    "median3+jpeg": (["median", "jpeg"], lambda p: [[3], [p]], QF[:-1], "int"),
    "awgn2+jpeg": (["awgn", "jpeg"], lambda p: [[2, 7], [p]], QF[:-1], "int"),
    "blur1.0+jpeg": (["blur", "jpeg"], lambda p: [[1.0], [p]], QF[:-1], "int"),
    "resize0.5+jpeg": (["resize", "jpeg"], lambda p: [[0.5], [p]], [90, 70, 50, 40, 30, 20, 13, 8, 5, 3], "int"),
    "jpeg": (["jpeg"], lambda p: [[p]], QF, "int"),
    "awgn4+jpeg": (["awgn", "jpeg"], lambda p: [[4, 7], [p]], QF[:-1], "int"),
    "blur0.8+jpeg+blur0.6": (["blur", "jpeg", "blur"], lambda p: [[0.8], [p], [0.6]], QF[:-1], "int"),
    "jpeg50+jpeg": (["jpeg", "jpeg"], lambda p: [[50], [p]], QF[:-1], "int"),
    "blur": (["blur"], lambda p: [[p]], [0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.6, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0], "float"),
    "median": (["median"], lambda p: [[p]], [3, 5, 7, 9, 11, 13, 15], "none"),
    "resize-lab": (["resize"], lambda p: [[p, "lab"]], [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5,
                                                      0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15], "float-down"),
    "awgn": (["awgn"], lambda p: [[p, 7]], [2, 4, 6, 8, 10, 13, 16, 20, 25, 30, 40, 55, 70], "float"),
    "awgn+blur1": (["awgn", "blur"], lambda p: [[p, 7], [1.0]], [4, 8, 12, 16, 20, 25, 30, 40, 55], "float"),
    "blur+sharp4": (["blur", "sharp"], lambda p: [[p], [4, 1.5]], [0.5, 0.8, 1.0, 1.3, 1.6, 2.0, 2.5, 3.0, 4.0], "float"),
    "blur+sharp10": (["blur", "sharp"], lambda p: [[p], [10, 2]], [0.5, 0.8, 1.0, 1.3, 1.6, 2.0, 2.5, 3.0, 4.0], "float"),
    "awgn+median3": (["awgn", "median"], lambda p: [[p, 7], [3]], [4, 8, 12, 16, 20, 25, 30], "float"),
    "median5+sharp": (["median", "sharp"], lambda p: [[5], [4, p]], [0.3, 0.6, 1.0, 1.5, 2.0], "float"),
    "resize-cubic": (["resize"], lambda p: [[p, "cubic"]], [0.9, 0.8, 0.7, 0.6, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2],
                     "float-down"),
    "resize-area": (["resize"], lambda p: [[p, "area"]], [0.9, 0.8, 0.7, 0.6, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2],
                    "float-down"),
    "blur-oriz": (["blur"], lambda p: [[(p / 3, p)]], [0.5, 0.8, 1.2, 1.6, 2.0, 3.0, 4.0, 6.0], "float"),
    "blur-vert": (["blur"], lambda p: [[(p, p / 3)]], [0.5, 0.8, 1.2, 1.6, 2.0, 3.0, 4.0, 6.0], "float"),
    "median-oriz": (["median"], lambda p: [[(1, p)]], [3, 5, 7, 9, 11, 15], "none"),
    "median-vert": (["median"], lambda p: [[(p, 1)]], [3, 5, 7, 9, 11, 15], "none"),
}
# zone in ordine di efficacia (contro di noi i migliori attacchi localizzati erano sulle zone lisce)
MASK_ORDER = ["lisce50", "lisce70", "lisceblk50", "lisce30", "text50", "text30", "bordi30"]
GREEDY_BLOCKS = (64, 32)


# ----------------------------------------------------------------------------
# WPSNR veloce: stessa formula dei lab (convolve2d 'same' con bordo 'wrap' = convoluzione
# circolare), calcolata con la FFT. Usato solo per scartare i candidati: il valore riportato nel
# log e' sempre ricalcolato con wpsnr() dei lab.
# ----------------------------------------------------------------------------
_H = None


def wpsnr_fast(a, b):
    global _H
    if _H is None:
        k = _get_csf_filter()
        pad = np.zeros((512, 512))
        pad[:k.shape[0], :k.shape[1]] = k
        _H = np.fft.rfft2(np.roll(pad, (-(k.shape[0] // 2), -(k.shape[1] // 2)), axis=(0, 1)))
    e = (a.astype(np.float64) - b.astype(np.float64)) / 255.0
    if not e.any():
        return 9999999.0
    we = np.fft.irfft2(np.fft.rfft2(e) * _H, s=e.shape)
    return 20 * np.log10(1.0 / np.sqrt(np.mean(we ** 2)))


# ----------------------------------------------------------------------------
# ORACOLO: la detection del gruppo attaccato, con timeout e gestione degli errori
# ----------------------------------------------------------------------------
class OracleError(Exception):
    pass


class _Timeout(Exception):
    pass


def _on_alarm(signum, frame):
    raise _Timeout()


class Oracle:
    def __init__(self, det_path, orig_path, wm_path, tmpdir, timeout):
        spec = importlib.util.spec_from_file_location(f"det_{os.getpid()}_{time.time_ns()}", det_path)
        mod = importlib.util.module_from_spec(spec)
        with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn), contextlib.redirect_stderr(dn):
            spec.loader.exec_module(mod)
        if not hasattr(mod, "detection"):
            raise OracleError(f"{os.path.basename(det_path)} non definisce detection()")
        self.det, self.orig, self.wm = mod.detection, orig_path, wm_path
        self.tmp = os.path.join(tmpdir, "att.bmp")
        self.timeout = timeout
        self.calls = self.errors = self.consecutive = 0
        self.last_error = ""
        self.cache = {}                                # stessa immagine -> stesso esito, niente chiamata
        signal.signal(signal.SIGALRM, _on_alarm)

    def broken(self, att):
        """True se la detection dice 'assente'. L'immagine passa per il BMP come quella caricata.
        Un errore o un timeout della detection conta come 'non rotta' (prudenza)."""
        key = att.tobytes()
        if key in self.cache:
            return self.cache[key]
        cv2.imwrite(self.tmp, att)
        self.calls += 1
        try:
            signal.setitimer(signal.ITIMER_REAL, self.timeout)
            with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn), contextlib.redirect_stderr(dn):
                out = self.det(self.orig, self.wm, self.tmp)
            signal.setitimer(signal.ITIMER_REAL, 0)
            res = int(out[0]) == 0
        except _Timeout:
            return self._fail(f"timeout (> {self.timeout:.0f} s)")
        except Exception as e:                       # noqa: BLE001  (codice altrui: qualunque errore)
            signal.setitimer(signal.ITIMER_REAL, 0)
            return self._fail(f"{type(e).__name__}: {e}")
        self.consecutive = 0
        self.cache[key] = res
        return res

    def _fail(self, msg):
        self.errors += 1
        self.consecutive += 1
        self.last_error = msg
        if self.consecutive >= MAX_ERRORS:
            raise OracleError(f"detection in errore {MAX_ERRORS} volte di fila, ultimo: {msg}")
        return False


def search(oracle, wm, fam, mask=None, deadline=None):
    """Attacco piu' leggero della famiglia che rompe. Restituisce (wpsnr, names, params) o None."""
    names, to_params, grid, kind = FAMILIES[fam]
    make = lambda p: attacks(wm, names, to_params(p), mask=mask)      # noqa: E731

    # 1. solo i valori sopra 35 dB (la griglia va dal leggero al forte: ci si ferma al primo sotto)
    valid = []
    for p in grid:
        if wpsnr_fast(wm, make(p)) < MIN_WPSNR:
            break
        valid.append(p)
    if not valid or not oracle.broken(make(valid[-1])):
        return None
    # 2. ricerca binaria del primo valore che rompe (assumendo: piu' forte -> piu' facile rompere)
    lo_i, hi_i = -1, len(valid) - 1                # valid[hi_i] rompe; valid[lo_i] no (-1 = nessuno)
    while hi_i - lo_i > 1:
        if deadline and time.time() > deadline:
            break
        mid = (lo_i + hi_i) // 2
        if oracle.broken(make(valid[mid])):
            hi_i = mid
        else:
            lo_i = mid
    best = valid[hi_i]
    # 3. raffinamento tra l'ultimo che non rompe e il primo che rompe
    if lo_i >= 0 and kind != "none" and not (deadline and time.time() > deadline):
        lo = valid[lo_i]
        if kind == "int":                          # JPEG: i QF interi tra i due, dal piu' leggero (QF alto)
            for p in range(lo - 1, best, -1):
                if oracle.broken(make(p)):
                    best = p
                    break
        else:                                      # 'float' o 'float-down': bisezione
            a, b = lo, best
            for _ in range(3):
                m = round((a + b) / 2, 3)
                if oracle.broken(make(m)):
                    b = best = m
                else:
                    a = m
    params = to_params(best)
    return wpsnr(wm, attacks(wm, names, params, mask=mask)), names, params


def greedy_blocks(oracle, wm, att, deadline):
    """Restituisce alla watermarkata i blocchi che non servono (prima 64x64, poi 32x32).
    Restituisce (immagine, keep): keep = maschera dei pixel in cui l'attacco e' rimasto, cosi'
    l'attacco e' riproducibile (attacco di base, poi watermarkata fuori da keep)."""
    cur = att.copy()
    keep = np.ones(wm.shape, bool)
    for b in GREEDY_BLOCKS:
        diff = np.abs(cur.astype(int) - wm.astype(int))
        blocks = [(i, j) for i in range(0, 512, b) for j in range(0, 512, b) if diff[i:i + b, j:j + b].any()]
        # prima i blocchi che "costano" di piu' in WPSNR (i piu' modificati)
        blocks.sort(key=lambda ij: -diff[ij[0]:ij[0] + b, ij[1]:ij[1] + b].sum())
        for i, j in blocks:
            if time.time() > deadline:
                return cur, keep
            trial = cur.copy()
            trial[i:i + b, j:j + b] = wm[i:i + b, j:j + b]
            if oracle.broken(trial):
                cur = trial
                keep[i:i + b, j:j + b] = False
    return cur, keep


def attack_image(wm_path, orig_path, det_path, out_dir, budget=600, use_greedy=True, timeout=10):
    """Attacca UNA immagine watermarkata di un altro gruppo (<gruppo>_<imageName>.bmp) e salva in out_dir
    il riuscito con il WPSNR piu' alto come fakebitinvestigation_<gruppo>_<imageName>.bmp (+ PNG delle zone
    se e' stato usato il greedy). L'originale (orig_path) NON viene mai letto qui: lo riceve solo la
    detection del gruppo attaccato, come input1. budget = secondi a disposizione.
    Restituisce un dizionario con l'esito; non solleva mai eccezioni (gli errori finiscono in res['error'])."""
    fname = os.path.basename(wm_path)
    group, image = fname[:-4].split("_", 1)
    t0 = time.time()
    res = {"image": image, "group": group, "wpsnr": None, "desc": None, "found": [], "calls": 0,
           "errors": 0, "error": None, "time": 0.0}
    oracle = None
    try:
        wm = cv2.imread(wm_path, cv2.IMREAD_GRAYSCALE)
        if wm is None or wm.shape != (512, 512):
            raise OracleError(f"immagine non leggibile o non 512x512: {fname}")
        with tempfile.TemporaryDirectory() as tmp:
            oracle = Oracle(det_path, orig_path, wm_path, tmp, timeout)
            # controllo: sulla watermarkata non attaccata la loro detection deve dire "presente"
            if oracle.broken(wm) and oracle.consecutive == 0:
                raise OracleError("la loro detection dice 'assente' gia' sulla watermarkata: non attacchiamo")
            found = []                                 # (wpsnr, descrizione, immagine)
            search_deadline = t0 + SEARCH_SHARE * budget if use_greedy else t0 + budget
            masks = {m: MASKS[m](wm) for m in MASK_ORDER}
            tries = [(fam, None) for fam in FAMILIES] + [(fam, m) for m in MASK_ORDER for fam in FAMILIES]
            for fam, mname in tries:
                if time.time() > search_deadline:
                    break
                mask = masks[mname] if mname else None
                r = search(oracle, wm, fam, mask, search_deadline)
                if r is not None:
                    w, names, params = r
                    found.append((w, describe(names, params, mname), attacks(wm, names, params, mask=mask)))
            keep_masks = {}
            if use_greedy and found:
                w, desc, att = max(found, key=lambda x: x[0])
                g, keep = greedy_blocks(oracle, wm, att, t0 + budget)
                if oracle.broken(g):
                    zfile = f"zone_{OUR_GROUP}_{group}_{image}.png"
                    gdesc = f"{desc}; poi solo dentro i blocchi bianchi di {zfile} (fuori: watermarkata invariata)"
                    found.append((wpsnr(wm, g), gdesc, g))
                    keep_masks[gdesc] = (zfile, keep)

            # tra i riusciti, il WPSNR piu' alto, ricontrollato con la detection
            found.sort(key=lambda x: -x[0])
            res["found"] = [(round(w, 2), d) for w, d, _ in found]
            for w, desc, att in found:
                if w >= MIN_WPSNR and oracle.broken(att):
                    cv2.imwrite(os.path.join(out_dir, f"{OUR_GROUP}_{group}_{image}.bmp"), att)
                    if desc in keep_masks:                  # maschera delle zone, per la verifica dei TA
                        zfile, keep = keep_masks[desc]
                        cv2.imwrite(os.path.join(out_dir, zfile), keep.astype(np.uint8) * 255)
                    res["wpsnr"], res["desc"] = w, desc
                    break
    except OracleError as e:
        res["error"] = str(e)
    except Exception:                                  # noqa: BLE001  (non deve mai fermare la gara)
        res["error"] = traceback.format_exc(limit=2).strip().splitlines()[-1]
    if oracle is not None:
        res["calls"], res["errors"] = oracle.calls, oracle.errors
    res["time"] = time.time() - t0
    return res


# ----------------------------------------------------------------------------
# RIGA DI COMANDO
# ----------------------------------------------------------------------------
def _num(x):
    return int(x) if x.lstrip("-").isdigit() else float(x)


def main():
    ap = argparse.ArgumentParser(description="Attacchi del gruppo fakebitinvestigation")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("applica", help="applica un attacco (o una catena) e salva l'immagine")
    a1.add_argument("image")
    a1.add_argument("out")
    a1.add_argument("attack", help="nome o catena separata da virgole, es. blur,jpeg")
    a1.add_argument("params", nargs="+", help="parametri in ordine; per 'resize' il metodo e' facoltativo")
    a2 = sub.add_parser("cerca", help="cerca l'attacco piu' leggero che rompe UNA immagine")
    a2.add_argument("--image", required=True, help="watermarkata dell'altro gruppo: <gruppo>_<imageName>.bmp")
    a2.add_argument("--original", required=True, help="originale: va SOLO alla loro detection, come input1")
    a2.add_argument("--detection", required=True, help="detection_<gruppo>.py del gruppo attaccato")
    a2.add_argument("--out", required=True)
    a2.add_argument("--budget", type=float, default=600, help="secondi a disposizione (default 600)")
    a2.add_argument("--timeout", type=float, default=10, help="secondi massimi per una chiamata alla detection")
    a2.add_argument("--no-greedy", action="store_true")
    a = ap.parse_args()

    if a.cmd == "applica":
        names = a.attack.split(",")
        n_par = {"awgn": 1, "blur": 1, "sharp": 2, "jpeg": 1, "resize": 1, "median": 1}
        vals, params = list(a.params), []
        for n in names:
            if n not in n_par:
                sys.exit(f"attacco non ammesso: {n!r} (ammessi: {', '.join(n_par)})")
            p = [_num(vals.pop(0)) for _ in range(n_par[n])]
            if n == "resize" and vals and not vals[0].lstrip("-").replace(".", "", 1).isdigit():
                p.append(vals.pop(0))
            params.append(p)
        if vals:
            sys.exit(f"parametri in piu': {vals}")
        wm = cv2.imread(a.image, cv2.IMREAD_GRAYSCALE)
        out = attacks(wm, names, params)
        cv2.imwrite(a.out, out)
        print(f"{describe(names, params)} -> {a.out}, WPSNR {wpsnr(wm, out):.2f} dB")
        return

    os.makedirs(a.out, exist_ok=True)
    r = attack_image(a.image, a.original, a.detection, a.out, a.budget, not a.no_greedy, a.timeout)
    info = f"({r['calls']} chiamate alla detection, {r['errors']} errori, {r['time']:.0f} s)"
    if r["error"]:
        print(f"ERRORE: {r['error']} {info}")
    elif r["wpsnr"] is None:
        print(f"non rotta {info}")
    else:
        print(f"ROTTA a {r['wpsnr']:.2f} dB con {r['desc']} {info}")
        saved = os.path.join(a.out, f"{OUR_GROUP}_{r['group']}_{r['image']}.bmp")
        print(f"salvata in {saved}")


if __name__ == "__main__":
    main()
