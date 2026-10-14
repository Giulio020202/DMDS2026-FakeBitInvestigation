"""
Capture the Mark 2026 - detection del gruppo fakebitinvestigation (bozza v0.1)

Detection NON-blind (p.5 del regolamento):
  input1 = percorso dell'immagine originale
  input2 = percorso dell'immagine watermarkata
  input3 = percorso dell'immagine attaccata
  output1 = 1 se il watermark e' presente nell'attaccata, 0 altrimenti
  output2 = WPSNR(watermarkata, attaccata), con il codice dei lab

Come funziona:
  1. dall'originale ricalcoliamo le 1024 coppie di coefficienti DCT usate dall'embedding
     (stessa funzione, stesse costanti: devono restare IDENTICHE a quelle dell'embedding);
  2. W_ext = watermark estratto dalla watermarkata, W_att = estratto dall'attaccata;
     il watermark NON viene letto da file ne' scritto nel codice: viene estratto (p.5);
  3. similarita' = massimo tra due misure (vedi spiegazione_embedding.md):
     - coseno classico, provato anche dopo aver "de-stirato" l'attaccata (il resize
       dei lab puo' restituire un'immagine allungata di 1-2 pixel);
     - coseno "pesato nello spazio": prima dell'estrazione ogni pixel di
       (attaccata - originale) viene pesato con 1 / (energia locale di (attaccata - originale) + EPS),
       cosi' le zone devastate da un attacco localizzato contano meno di quelle intatte.
       Il peso dipende solo da attaccata e originale, quindi su un'immagine che non contiene
       il nostro watermark la similarita' resta centrata in 0 (misurato: nessun bias);
  4. presente se similarita' >= TAU. La decisione dipende SOLO dalla similarita': il regolamento
     aggiornato vieta di saltare la detection o di forzare output1 in base al WPSNR (anche < 25 dB).

Regole rispettate: file unico, nessun percorso relativo, nessuno script esterno, unica
funzione "esterna" e' il WPSNR dei lab (copiato qui sotto), nessuna print, nessuna finestra,
tempo misurato ben sotto i 5 s.
"""
import cv2
import numpy as np
from scipy.fft import dctn
from scipy.ndimage import uniform_filter
from scipy.signal import convolve2d

# ----------------------------------------------------------------------------
# PARAMETRI (devono coincidere con quelli di embedding_fakebitinvestigation.py)
# ----------------------------------------------------------------------------
R_MIN = 32.0
R_MAX = 128.0
N_BITS = 1024
TAU = 0.11                      # da roc_fakebitinvestigation.py: 0.108 per FPR <= 0.1%, arrotondata per prudenza
SQ_WIN = 16                     # finestra (pixel) della pesatura spaziale
SQ_EPS = 0.5                    # regolarizzazione della pesatura spaziale
STRETCH = (510, 511, 513, 514)  # dimensioni intermedie del resize dei lab da compensare
C_FLOOR = 1.0                   # minimo |coefficiente| al denominatore dell'estrazione


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


# ----------------------------------------------------------------------------
# FUNZIONI INTERNE (identiche all'embedding)
# ----------------------------------------------------------------------------
def _dct2(x):
    return dctn(x, norm='ortho')


def _select_pairs(coeffs):
    H, W = coeffs.shape
    u = np.arange(H)[:, None]
    v = np.arange(0, W - 1, 2)[None, :]
    uu = np.broadcast_to(u, (H, v.shape[1])).ravel()
    vv = np.broadcast_to(v, (H, v.shape[1])).ravel()
    r = np.sqrt(uu**2 + (vv + 0.5)**2)
    band = (r >= R_MIN) & (r <= R_MAX)
    uu, vv = uu[band], vv[band]
    strength = np.minimum(np.abs(coeffs[uu, vv]), np.abs(coeffs[uu, vv + 1]))
    best = np.argsort(-strength, kind='stable')[:N_BITS]
    best = np.sort(best)
    return uu[best], vv[best]


def _extract(c0, uu, vv, img):
    """W_i = [(X1/C1 - 1) - (X2/C2 - 1)] / (1/|C1| + 1/|C2|)  (0 esatto sull'originale).
    Scritta come s*(X - C)/max(|C|, C_FLOOR): identica quando |C| >= C_FLOOR (sempre vero sulle
    foto naturali: il coefficiente scelto piu' piccolo e' circa 13), ma senza divisioni per zero
    su immagini senza contenuto nella banda (es. costanti o gradienti)."""
    c1 = _dct2(img)
    a1, a2 = c0[uu, vv], c0[uu, vv + 1]
    s1 = np.where(a1 < 0, -1.0, 1.0)
    s2 = np.where(a2 < 0, -1.0, 1.0)
    m1 = np.maximum(np.abs(a1), C_FLOOR)
    m2 = np.maximum(np.abs(a2), C_FLOOR)
    r1 = s1 * (c1[uu, vv] - a1) / m1
    r2 = s2 * (c1[uu, vv + 1] - a2) / m2
    return (r1 - r2) / (1.0 / m1 + 1.0 / m2)


def _similarity(x, y):
    nx, ny = np.linalg.norm(x), np.linalg.norm(y)
    if nx == 0 or ny == 0:
        return 0.0
    return float(np.dot(x, y) / (nx * ny))


def _destretch(img, m):
    """Inverte lo stiramento del resize dei lab quando l'immagine intermedia ha m pixel."""
    x = (np.arange(img.shape[0]) + 0.5) * m / img.shape[0] - 0.5
    mx, my = np.meshgrid(x, x)
    return cv2.remap(img.astype(np.float32), mx.astype(np.float32), my.astype(np.float32),
                     cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE).astype(np.float64)


def _attacked_vectors(c0, uu, vv, original, attacked):
    """Vettori estratti dall'attaccata, uno per ciascuna variante di estrazione:
    (1) estrazione normale e sull'attaccata "de-stirata" (resize dei lab);
    (2) estrazione con pesatura spaziale (peso calcolato solo da attaccata e originale)."""
    o = original.astype(np.float64)
    a = attacked.astype(np.float64)
    vecs = [_extract(c0, uu, vv, a)]
    for m in STRETCH:
        vecs.append(_extract(c0, uu, vv, _destretch(attacked, m)))
    d = a - o
    rho = 1.0 / (uniform_filter(d * d, SQ_WIN) + SQ_EPS)
    vecs.append(_extract(c0, uu, vv, o + rho * d))
    return vecs


def _score(reference, vecs):
    """Similarita' usata per la decisione: massimo dei coseni con le varianti."""
    return max(_similarity(reference, y) for y in vecs)


# ----------------------------------------------------------------------------
# DETECTION (firma imposta dal regolamento)
# ----------------------------------------------------------------------------
def detection(input1, input2, input3):
    original = cv2.imread(input1, cv2.IMREAD_GRAYSCALE)
    watermarked = cv2.imread(input2, cv2.IMREAD_GRAYSCALE)
    attacked = cv2.imread(input3, cv2.IMREAD_GRAYSCALE)

    c0 = _dct2(original.astype(np.float64))
    uu, vv = _select_pairs(c0)
    W_ext = _extract(c0, uu, vv, watermarked.astype(np.float64))
    sim = _score(W_ext, _attacked_vectors(c0, uu, vv, original, attacked))

    output2 = wpsnr(watermarked, attacked)
    output1 = 1 if sim >= TAU else 0
    return output1, output2
