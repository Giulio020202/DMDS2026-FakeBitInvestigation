"""
Predizione dell'attacco (idea della terza iterazione, 14/10).

La detection e' non-blind: ha l'originale. Gli attacchi permessi (tranne l'AWGN) sono
deterministici, quindi se l'attaccata e' T(watermarkata) la detection puo' calcolare T(originale)
e sottrarlo: attaccata - T(originale) = T(originale + P) - T(originale), cioe' resta solo la parte
del watermark sopravvissuta, senza il rumore che l'attacco ha messo sull'immagine stessa.

T viene stimato cercando, in una libreria di attacchi permessi (singoli e a coppie), quello che
minimizza ||attaccata - T(originale)||^2. La scelta usa SOLO originale e attaccata, mai la
watermarkata: se l'attaccata non contiene il nostro watermark, la scelta non dipende da w e la
similarita' resta centrata in 0 (stessa lezione della pesatura spaziale, sez. 11.4 del .md).
"""
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, median_filter

BLUR = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.7, 2.0, 2.5, 3.0)
MED = (3, 5, 7)
JNEAR = 6           # raggio della ricerca del QF attorno al migliore, nelle catene


def _u8(x):
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def jpeg(img, q):
    return cv2.imdecode(cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, int(q)])[1], 0)


def _op(img, kind, p):
    if kind == 'jpeg':
        return jpeg(img, p)
    if kind == 'blur':
        return _u8(gaussian_filter(img.astype(np.float64), p))
    return median_filter(img, size=p)


def _singles():
    return [('jpeg', q) for q in range(1, 101)] + [('blur', s) for s in BLUR] + [('med', k) for k in MED]


def predict(original, attacked):
    """Restituisce (T(originale) come float, descrizione di T, SSE)."""
    a = attacked.astype(np.float64)

    def sse(x):
        d = a - x
        return float(np.einsum('ij,ij->', d, d))

    best = [sse(original.astype(np.float64)), original, 'id']

    def consider(img, name):
        e = sse(img)
        if e < best[0]:
            best[0], best[1], best[2] = e, img, name

    # 1) attacco singolo
    cache = {}
    for k, p in _singles():
        cache[(k, p)] = _op(original, k, p)
        consider(cache[(k, p)], f'{k}{p}')
    if best[2] == 'id':
        return original.astype(np.float64), 'id', best[0]
    s_name = best[2]
    k1 = 'jpeg' if s_name.startswith('jpeg') else ('blur' if s_name.startswith('blur') else 'med')
    p1 = float(s_name[len(k1):]) if k1 == 'blur' else int(s_name[len(k1):])
    # parametri da riprovare per il primo attacco (se e' un JPEG, il QF puo' cambiare un po' in catena)
    p1s = list(range(max(1, p1 - JNEAR), min(100, p1 + JNEAR) + 1)) if k1 == 'jpeg' else [p1]
    # 2) catene di due: x poi T1, e T1 poi x
    for k, p in _singles():
        if k == 'jpeg' and k1 == 'jpeg' and p % 3:
            continue        # doppio JPEG: primo QF a passo 3, secondo vicino a quello trovato
        pre = cache[(k, p)]
        for q in p1s:
            consider(_op(pre, k1, q), f'{k}{p}+{k1}{q}')
        if k != 'jpeg':
            for q in (p1s[::2] if k1 == 'jpeg' else p1s):
                consider(_op(cache[(k1, q)], k, p), f'{k1}{q}+{k}{p}')
    return best[1].astype(np.float64), best[2], best[0]


# ---------------------------------------------------------------------------------------------
# VERSIONE VELOCE (14/10 sera). La versione sopra costava 20-33 s per detection quando l'attacco
# stimato conteneva un mediano (scipy.median_filter e' lentissimo) -> oltre i 5 s del regolamento.
# Stessa idea, ma:
#   - mediano con cv2.medianBlur e blur con cv2.GaussianBlur: risultato IDENTICO a scipy
#     (verificato: mediano uguale lontano dai bordi, blur 0 pixel diversi), 20-300 volte piu' veloci;
#   - tutte le candidate in uint8 e distanza con cv2.norm (0.13 ms invece di 6 ms);
#   - catene piu' mirate: QF vicini a quello trovato, griglie piu' rade per i filtri dopo/prima.
# ---------------------------------------------------------------------------------------------
BLUR_POST = (0.5, 0.65, 0.8, 1.0, 1.2, 1.5, 2.0)


def _blur_cv(img, s):
    r = int(4.0 * s + 0.5)
    return _u8(cv2.GaussianBlur(img.astype(np.float64), (2 * r + 1, 2 * r + 1), s, borderType=cv2.BORDER_REFLECT))


def _op_fast(img, kind, p):
    if kind == 'jpeg':
        return jpeg(img, p)
    if kind == 'blur':
        return _blur_cv(img, p)
    return cv2.medianBlur(img, p)


def predict_fast(original, attacked):
    """Come predict(): restituisce (T(originale) float, descrizione, SSE), ma in ~0.2-0.6 s."""
    att = np.ascontiguousarray(attacked, dtype=np.uint8)
    best = [cv2.norm(att, original, cv2.NORM_L2SQR), original, 'id']

    def consider(img, name):
        e = cv2.norm(att, img, cv2.NORM_L2SQR)
        if e < best[0]:
            best[0], best[1], best[2] = e, img, name

    cache = {}
    for k, p in _singles():
        cache[(k, p)] = _op_fast(original, k, p)
        consider(cache[(k, p)], f'{k}{p}')
    if best[2] == 'id':
        return original.astype(np.float64), 'id', best[0]
    name1 = best[2]
    k1 = 'jpeg' if name1.startswith('jpeg') else ('blur' if name1.startswith('blur') else 'med')
    p1 = float(name1[len(k1):]) if k1 == 'blur' else int(name1[len(k1):])
    filters = [('blur', s) for s in BLUR] + [('med', m) for m in MED]
    if k1 == 'jpeg':
        qs = list(range(max(1, p1 - JNEAR), min(100, p1 + JNEAR) + 1))
        for k, p in filters:                                   # filtro poi JPEG
            for q in qs:
                consider(jpeg(cache[(k, p)], q), f'{k}{p}+jpeg{q}')
        # JPEG poi filtro: griglia congiunta QF (vicini al trovato) x filtro (griglia rada),
        # poi il filtro sulla griglia completa al QF migliore
        post = [('blur', s) for s in BLUR_POST] + [('med', m) for m in MED]
        bq, be = p1, np.inf
        for q in qs:
            jq = cache[('jpeg', q)]
            for k, p in post:
                img = _op_fast(jq, k, p)
                e = cv2.norm(att, img, cv2.NORM_L2SQR)
                consider(img, f'jpeg{q}+{k}{p}')
                if e < be:
                    bq, be = q, e
        for k, p in filters:
            consider(_op_fast(cache[('jpeg', bq)], k, p), f'jpeg{bq}+{k}{p}')
        for q1 in range(5, 100, 5):                            # doppio JPEG
            for q in qs[JNEAR - 2:JNEAR + 3]:
                consider(jpeg(cache[('jpeg', q1)], q), f'jpeg{q1}+jpeg{q}')
    else:
        f1 = cache[(k1, p1)]
        for q in range(1, 101):                                # filtro poi JPEG
            consider(jpeg(f1, q), f'{k1}{p1}+jpeg{q}')
        for q in range(2, 101, 3):                             # JPEG poi filtro
            consider(_op_fast(cache[('jpeg', q)], k1, p1), f'jpeg{q}+{k1}{p1}')
        for k, p in filters:                                   # due filtri
            consider(_op_fast(cache[(k, p)], k1, p1), f'{k}{p}+{k1}{p1}')
    return best[1].astype(np.float64), best[2], best[0]
