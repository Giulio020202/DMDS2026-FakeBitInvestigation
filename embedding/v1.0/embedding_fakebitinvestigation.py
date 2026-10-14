"""
Capture the Mark 2026 - embedding del gruppo fakebitinvestigation (bozza v0.2)

Schema: spread spectrum nella DCT globale dell'immagine (Cox et al., lab 3)
con quattro modifiche pensate per la challenge:
  1) watermark bipolare (bit 0/1 -> -1/+1);
  2) ogni bit va su una coppia di coefficienti DCT adiacenti (u,v),(u,v+1):
     il primo viene "spinto" di +a*w, il secondo di -a*w (in modulo).
     In detection guardiamo la differenza delle variazioni relative, quindi
     un attacco che moltiplica entrambi i coefficienti per lo stesso guadagno
     (blur, sharpening, resize, contrasto...) non cancella il watermark;
  3) i coefficienti stanno nella banda radiale R_MIN..R_MAX (frequenze medio-basse)
     e l'ampiezza 'a' e' uguale per tutte le coppie (additiva), calibrata
     in automatico per ottenere esattamente il WPSNR che vogliamo;
  4) (v0.2) il pattern del watermark nei pixel viene modulato da una maschera
     spaziale che lo rinforza nelle zone testurizzate e lo attenua in quelle lisce
     (GAMMA). Il WPSNR dei lab non cambia dove mettiamo l'energia, mentre per
     l'attaccante toglierla dalle texture costa molto di piu' (vedi sez. 11 del .md).

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
from scipy.fft import dctn, idctn
from scipy.ndimage import gaussian_filter, uniform_filter
from scipy.signal import convolve2d

# ----------------------------------------------------------------------------
# PARAMETRI (gli unici da toccare nel fine-tuning)
# ----------------------------------------------------------------------------
TARGET_WPSNR = 62.05   # dB: appena sopra la soglia 62 della tabella punti (5 punti).
                       # 58.05 se i TA confermano che chi non viene mai rotto prende
                       # robustezza piena + bonus (vedi sez. 4.8 del .md)
R_MIN = 32.0           # banda radiale dei coefficienti DCT usati (frequenze
R_MAX = 128.0          # ~0.03-0.125 cicli/pixel): vedi spiegazione nel .md
N_BITS = 1024          # lunghezza del watermark assegnato
AMP_MIN, AMP_MAX = 0.2, 40.0   # intervallo di ricerca dell'ampiezza
N_ITER = 14                    # passi di bisezione (precisione ~0.03% sull'ampiezza)
GAMMA = 0.0            # forza della maschera spaziale (0 = nessuna maschera, schema v0.1).
                       # 0.5 era piu' robusta a parita' di tau, ma NON a parita' di falsi positivi (sez. 11 del .md)
MASK_WIN = 16          # finestra (pixel) per la deviazione standard locale
MASK_SMOOTH = 8.0      # sigma (pixel) dello smussamento della maschera
MASK_OFFSET = 2.0      # aggiunto alla dev. std locale: le zone piatte non vanno a zero


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
# FUNZIONI INTERNE (le stesse andranno copiate nel file di detection)
# ----------------------------------------------------------------------------
def _dct2(x):
    # DCT-II 2D ortonormale: identica a dct(dct(x, axis=0, norm='ortho'), axis=1, norm='ortho') dei lab
    return dctn(x, norm='ortho')


def _idct2(c):
    return idctn(c, norm='ortho')


def _select_pairs(coeffs):
    """
    Sceglie le N_BITS coppie (u,v),(u,v+1) [v pari] nella banda R_MIN<=r<=R_MAX
    con il minimo |coefficiente| piu' grande. Dipende solo dall'immagine originale,
    quindi la detection (non-blind) ritrova esattamente le stesse posizioni.
    Restituisce le coppie ordinate per posizione (u,v): ordine deterministico.
    """
    H, W = coeffs.shape
    u = np.arange(H)[:, None]
    v = np.arange(0, W - 1, 2)[None, :]
    uu = np.broadcast_to(u, (H, v.shape[1])).ravel()
    vv = np.broadcast_to(v, (H, v.shape[1])).ravel()
    r = np.sqrt(uu**2 + (vv + 0.5)**2)          # frequenza radiale (indice DCT) della coppia
    band = (r >= R_MIN) & (r <= R_MAX)
    uu, vv = uu[band], vv[band]
    strength = np.minimum(np.abs(coeffs[uu, vv]), np.abs(coeffs[uu, vv + 1]))
    best = np.argsort(-strength, kind='stable')[:N_BITS]
    best = np.sort(best)                          # ordine per posizione, non per ampiezza
    return uu[best], vv[best]


def _apply_mark(coeffs, uu, vv, w, amp):
    """Coefficienti DCT watermarkati: +amp*w sul primo modulo, -amp*w sul secondo."""
    s1 = np.sign(coeffs[uu, vv])
    s1[s1 == 0] = 1.0
    s2 = np.sign(coeffs[uu, vv + 1])
    s2[s2 == 0] = 1.0
    out = coeffs.copy()
    out[uu, vv] += s1 * amp * w
    out[uu, vv + 1] -= s2 * amp * w
    return out


def _to_image(coeffs):
    return np.clip(np.rint(_idct2(coeffs)), 0, 255).astype(np.uint8)


def _texture_mask(original):
    """Maschera spaziale >0 con media 1: piu' alta dove l'originale e' testurizzato.
    (dev. std locale + MASK_OFFSET)^GAMMA, smussata e normalizzata."""
    f = original.astype(np.float64)
    m = uniform_filter(f, MASK_WIN)
    sd = np.sqrt(np.maximum(uniform_filter(f * f, MASK_WIN) - m * m, 0)) + MASK_OFFSET
    k = gaussian_filter(sd ** GAMMA, MASK_SMOOTH)
    return k / k.mean()


def _masked_pattern(coeffs, uu, vv, w, mask):
    """Pattern del watermark nei pixel ad ampiezza 1 (stesse spinte di _apply_mark:
    +sgn(C1)*w sul primo coefficiente, -sgn(C2)*w sul secondo), moltiplicato per la maschera."""
    s1 = np.sign(coeffs[uu, vv])
    s1[s1 == 0] = 1.0
    s2 = np.sign(coeffs[uu, vv + 1])
    s2[s2 == 0] = 1.0
    P = np.zeros_like(coeffs)
    P[uu, vv] = s1 * w
    P[uu, vv + 1] = -s2 * w
    return _idct2(P) * mask


def _marked_image(original, coeffs, uu, vv, w, amp, pattern):
    """Immagine watermarkata con ampiezza amp. Senza maschera (pattern None) e' lo schema v0.1."""
    if pattern is None:
        return _to_image(_apply_mark(coeffs, uu, vv, w, amp))
    return np.clip(np.rint(original.astype(np.float64) + amp * pattern), 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------
# EMBEDDING (firma imposta dal regolamento)
# ----------------------------------------------------------------------------
def embedding(input1, input2):
    original = cv2.imread(input1, cv2.IMREAD_GRAYSCALE)
    bits = np.load(input2).astype(np.float64).ravel()[:N_BITS]
    w = 2.0 * bits - 1.0                          # 0/1 -> -1/+1

    coeffs = _dct2(original.astype(np.float64))
    uu, vv = _select_pairs(coeffs)
    pattern = _masked_pattern(coeffs, uu, vv, w, _texture_mask(original)) if GAMMA else None

    # Calibrazione dell'ampiezza: la piu' grande che mantiene WPSNR >= TARGET_WPSNR.
    # Il WPSNR decresce con l'ampiezza -> bisezione (in scala logaritmica).
    lo, hi = AMP_MIN, AMP_MAX
    best = _marked_image(original, coeffs, uu, vv, w, lo, pattern)
    for _ in range(N_ITER):
        mid = np.sqrt(lo * hi)
        candidate = _marked_image(original, coeffs, uu, vv, w, mid, pattern)
        if wpsnr(original, candidate) >= TARGET_WPSNR:
            lo, best = mid, candidate
        else:
            hi = mid

    output1 = best
    return output1
