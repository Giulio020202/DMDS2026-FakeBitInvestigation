"""
Capture the Mark 2026 - detection del gruppo fakebitinvestigation (versione 2.0, IBRIDA DCT + wavelet)

Detection NON-blind (p.5 del regolamento):
  input1 = percorso dell'immagine originale
  input2 = percorso dell'immagine watermarkata
  input3 = percorso dell'immagine attaccata
  output1 = 1 se il watermark e' presente nell'attaccata, 0 altrimenti
  output2 = WPSNR(watermarkata, attaccata), con il codice dei lab

Come funziona (stessa logica della v1.0, con due parti):
  1. dall'originale ricalcoliamo le 1024 coppie DCT e le 1024 coppie wavelet usate dall'embedding
     (stesse funzioni, stesse costanti: devono restare IDENTICHE a quelle dell'embedding);
  2. da un'immagine si estrae un vettore di 2048 valori: i 1024 della parte DCT seguiti dai 1024
     della parte wavelet. W_ext = estratto dalla watermarkata, W_att = estratto dall'attaccata;
     il watermark NON viene letto da file ne' scritto nel codice: viene estratto (p.5);
  3. similarita' = massimo dei coseni tra W_ext e il vettore estratto da:
     - l'attaccata;
     - l'attaccata "de-stirata" (il resize dei lab puo' restituire un'immagine allungata di 1-2 pixel);
     - l'attaccata con la pesatura spaziale 1 / (energia locale di (attaccata - originale) + EPS),
       che dipende solo da attaccata e originale (nessun bias se il nostro watermark non c'e');
  4. presente se similarita' >= TAU. La decisione dipende SOLO dalla similarita': il regolamento
     aggiornato vieta di saltare la detection o di forzare output1 in base al WPSNR (anche < 25 dB).

Regole rispettate: file unico, nessun percorso relativo, nessuno script esterno, unica
funzione "esterna" e' il WPSNR dei lab (copiato qui sotto), nessuna print, nessuna finestra.
"""
import cv2
import numpy as np
import pywt
from scipy.fft import dctn
from scipy.ndimage import uniform_filter
from scipy.signal import convolve2d

# ----------------------------------------------------------------------------
# PARAMETRI (devono coincidere con quelli di embedding_fakebitinvestigation.py)
# ----------------------------------------------------------------------------
R_MIN = 32.0
R_MAX = 128.0
N_BITS = 1024
WAVELET = "db4"
DWT_LEVELS = 4
DWT_USE = (3, 4)
TAU = 0.087                     # ROC (roc_fakebitinvestigation.py, run completa del 19/10): 0.0867 = la soglia piu' alta tra
                                # H0-R 0.0835, H0-A 0.0841, H0-B 0.0867 (FPR <= 0.1% su ciascuno), arrotondata per eccesso
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
# FUNZIONI INTERNE (selezione identica all'embedding)
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


def _dwt2(x):
    arr, slices = pywt.coeffs_to_array(pywt.wavedec2(x, WAVELET, mode='periodization', level=DWT_LEVELS))
    return arr, slices


def _select_pairs_dwt(arr, slices):
    m = np.zeros(arr.shape, bool)
    for lev in DWT_USE:
        k = DWT_LEVELS - lev + 1
        for key in ('ad', 'da', 'dd'):
            m[slices[k][key]] = True
    ii, jj = np.nonzero(m[:, :-1] & m[:, 1:])
    keep = jj % 2 == 0
    ii, jj = ii[keep], jj[keep]
    strength = np.minimum(np.abs(arr[ii, jj]), np.abs(arr[ii, jj + 1]))
    best = np.sort(np.argsort(-strength, kind='stable')[:N_BITS])
    return ii[best], jj[best]


def _pair_extract(c0, ii, jj, c1):
    """W_i = [(X1/C1 - 1) - (X2/C2 - 1)] / (1/|C1| + 1/|C2|)  (0 esatto sull'originale),
    in forma sicura s*(X - C)/max(|C|, C_FLOOR) come nella v1.0."""
    a1, a2 = c0[ii, jj], c0[ii, jj + 1]
    s1 = np.where(a1 < 0, -1.0, 1.0)
    s2 = np.where(a2 < 0, -1.0, 1.0)
    m1 = np.maximum(np.abs(a1), C_FLOOR)
    m2 = np.maximum(np.abs(a2), C_FLOOR)
    r1 = s1 * (c1[ii, jj] - a1) / m1
    r2 = s2 * (c1[ii, jj + 1] - a2) / m2
    return (r1 - r2) / (1.0 / m1 + 1.0 / m2)


def _context(original):
    """Tutto quello che dipende solo dall'originale: coefficienti e posizioni delle due parti."""
    o = original.astype(np.float64)
    c0 = _dct2(o)
    uu, vv = _select_pairs(c0)
    a0, slices = _dwt2(o)
    ii, jj = _select_pairs_dwt(a0, slices)
    return c0, uu, vv, a0, ii, jj


def _extract(ctx, img):
    """Vettore di 2048 valori: parte DCT seguita dalla parte wavelet."""
    c0, uu, vv, a0, ii, jj = ctx
    x = np.asarray(img, dtype=np.float64)
    return np.concatenate([_pair_extract(c0, uu, vv, _dct2(x)), _pair_extract(a0, ii, jj, _dwt2(x)[0])])


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


def _attacked_vectors(ctx, original, attacked):
    """Vettori estratti dall'attaccata: normale, de-stirata (resize dei lab), pesata nello spazio."""
    o = original.astype(np.float64)
    a = attacked.astype(np.float64)
    vecs = [_extract(ctx, a)]
    for m in STRETCH:
        vecs.append(_extract(ctx, _destretch(attacked, m)))
    d = a - o
    rho = 1.0 / (uniform_filter(d * d, SQ_WIN) + SQ_EPS)
    vecs.append(_extract(ctx, o + rho * d))
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

    ctx = _context(original)
    W_ext = _extract(ctx, watermarked)
    sim = _score(W_ext, _attacked_vectors(ctx, original, attacked))

    output2 = wpsnr(watermarked, attacked)
    output1 = 1 if sim >= TAU else 0
    return output1, output2
