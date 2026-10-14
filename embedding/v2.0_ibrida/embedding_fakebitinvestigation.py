"""
Capture the Mark 2026 - embedding del gruppo fakebitinvestigation (versione 2.0, IBRIDA DCT + wavelet)

Stesso schema della versione 1.0 (spread spectrum a coppie differenziali, watermark bipolare,
ampiezza additiva uniforme calibrata al WPSNR target), ma lo STESSO watermark viene inserito
in due trasformate diverse:
  1) DCT globale, banda radiale R_MIN..R_MAX (identica alla v1.0);
  2) wavelet db4 a DWT_LEVELS livelli, sottobande di dettaglio dei livelli DWT_USE
     (livello 3 ~ 0.06-0.125 cicli/pixel, livello 4 ~ 0.03-0.06: circa la stessa banda della DCT,
     ma con funzioni base localizzate nello spazio).
Le due parti hanno ampiezze relative K_DCT e K_DWT; un'unica scala viene calibrata con la
bisezione in modo da ottenere esattamente TARGET_WPSNR (codice WPSNR dei lab).

Regole rispettate (document.pdf, regolamento aggiornato):
  - firma embedding(input1, input2) -> output1
      input1 = stringa col nome/percorso dell'immagine originale (BMP 512x512 grigi)
      input2 = stringa col nome/percorso del watermark (.npy, 1024 bit 0/1)
      output1 = immagine watermarkata (numpy uint8 512x512)
  - immagini lette con OpenCV;
  - il codice non stampa nulla, non apre finestre, non scrive file;
  - il "fine-tuning" consentito dopo il 27/10 riguarda solo parametri e
    posizioni di embedding: sono tutti nelle costanti qui sotto.
"""
import cv2
import numpy as np
import pywt
from scipy.fft import dctn, idctn
from scipy.signal import convolve2d

# ----------------------------------------------------------------------------
# PARAMETRI (gli unici da toccare nel fine-tuning; vanno copiati identici nella detection)
# ----------------------------------------------------------------------------
TARGET_WPSNR = 62.05   # dB: appena sopra la soglia 62 della tabella punti (5 punti)
R_MIN = 32.0           # banda radiale dei coefficienti DCT (come la v1.0)
R_MAX = 128.0
N_BITS = 1024          # lunghezza del watermark assegnato
WAVELET = "db4"        # wavelet della seconda parte
DWT_LEVELS = 4         # livelli della decomposizione
DWT_USE = (3, 4)       # livelli le cui sottobande di dettaglio ospitano il watermark
K_DCT = 1.0            # ampiezza relativa della parte DCT
K_DWT = 1.0            # ampiezza relativa della parte wavelet
AMP_MIN, AMP_MAX = 0.2, 40.0   # intervallo di ricerca della scala
N_ITER = 14                    # passi di bisezione


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
# PARTE 1: DCT globale (identica alla v1.0)
# ----------------------------------------------------------------------------
def _dct2(x):
    return dctn(x, norm='ortho')


def _idct2(c):
    return idctn(c, norm='ortho')


def _select_pairs(coeffs):
    """N_BITS coppie (u,v),(u,v+1) [v pari] nella banda R_MIN<=r<=R_MAX con il min |coeff.| piu' grande."""
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


# ----------------------------------------------------------------------------
# PARTE 2: wavelet db4 (stesse coppie differenziali, dentro una sottobanda)
# ----------------------------------------------------------------------------
def _dwt2(x):
    """Coefficienti DWT (ortogonale, 'periodization': vale Parseval) messi in un unico array 512x512."""
    arr, slices = pywt.coeffs_to_array(pywt.wavedec2(x, WAVELET, mode='periodization', level=DWT_LEVELS))
    return arr, slices


def _idwt2(arr, slices):
    return pywt.waverec2(pywt.array_to_coeffs(arr, slices, output_format='wavedec2'), WAVELET, mode='periodization')


def _select_pairs_dwt(arr, slices):
    """N_BITS coppie orizzontali (i,j),(i,j+1) [j pari] dentro le sottobande di dettaglio dei
    livelli DWT_USE, con il min |coeff.| piu' grande, ordinate per posizione."""
    m = np.zeros(arr.shape, bool)
    for lev in DWT_USE:
        k = DWT_LEVELS - lev + 1           # in wavedec2 l'indice 1 e' il livello piu' grossolano
        for key in ('ad', 'da', 'dd'):
            m[slices[k][key]] = True
    ii, jj = np.nonzero(m[:, :-1] & m[:, 1:])
    keep = jj % 2 == 0
    ii, jj = ii[keep], jj[keep]
    strength = np.minimum(np.abs(arr[ii, jj]), np.abs(arr[ii, jj + 1]))
    best = np.sort(np.argsort(-strength, kind='stable')[:N_BITS])
    return ii[best], jj[best]


def _pair_pattern(c, ii, jj, w):
    """Spinte ad ampiezza 1: +sgn(C1)*w sul primo coefficiente, -sgn(C2)*w sul secondo."""
    s1 = np.where(c[ii, jj] < 0, -1.0, 1.0)
    s2 = np.where(c[ii, jj + 1] < 0, -1.0, 1.0)
    P = np.zeros_like(c)
    P[ii, jj] = s1 * w
    P[ii, jj + 1] = -s2 * w
    return P


# ----------------------------------------------------------------------------
# EMBEDDING (firma imposta dal regolamento)
# ----------------------------------------------------------------------------
def embedding(input1, input2):
    original = cv2.imread(input1, cv2.IMREAD_GRAYSCALE)
    bits = np.load(input2).astype(np.float64).ravel()[:N_BITS]
    w = 2.0 * bits - 1.0                          # 0/1 -> -1/+1
    o = original.astype(np.float64)

    c = _dct2(o)
    uu, vv = _select_pairs(c)
    arr, slices = _dwt2(o)
    ii, jj = _select_pairs_dwt(arr, slices)
    # pattern del watermark nei pixel (somma delle due parti, entrambe lineari)
    pattern = K_DCT * _idct2(_pair_pattern(c, uu, vv, w)) + K_DWT * _idwt2(_pair_pattern(arr, ii, jj, w), slices)

    def marked(s):
        return np.clip(np.rint(o + s * pattern), 0, 255).astype(np.uint8)

    # Calibrazione della scala: la piu' grande che mantiene WPSNR >= TARGET_WPSNR (bisezione logaritmica).
    lo, hi = AMP_MIN, AMP_MAX
    best = marked(lo)
    for _ in range(N_ITER):
        mid = np.sqrt(lo * hi)
        candidate = marked(mid)
        if wpsnr(original, candidate) >= TARGET_WPSNR:
            lo, best = mid, candidate
        else:
            hi = mid

    output1 = best
    return output1
