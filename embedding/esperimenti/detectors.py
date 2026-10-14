"""
Detector da confrontare. Ognuno: f(attaccata uint8) -> similarita'.

Tutti calcolano cos(W_ext, y) con y che dipende SOLO da attaccata, watermarkata e originale
attraverso quantita' indipendenti dal watermark (pesi stimati sui coefficienti NON marcati).
Quindi se l'attaccata non contiene il nostro watermark, sim ha media 0 e dev. std 1/32
(stesso argomento del .md, par. 4.2) e la soglia tau resta valida.

  cos      : coseno semplice (detection attuale)
  wcos     : coseno con y_i = W_att_i * g_i / var_i  (filtro adattato: g_i guadagno e var_i
             rumore dell'attacco stimati nell'intorno DCT della coppia i, sui coefficienti non marcati)
  sp:<win>:<eps> : pesatura SPAZIALE. Prima dell'estrazione la differenza (attaccata - originale)
             viene moltiplicata pixel per pixel per rho = 1 / (media locale di (attaccata - watermarkata)^2 + eps)
             (media su finestre win x win). Dove l'attacco non ha toccato l'immagine la differenza
             e' esattamente il watermark e pesa molto; dove l'attacco ha distrutto, pesa poco.
             L'estrazione e' lineare in (immagine - originale), quindi e' un'estrazione "pesata".
             ATTENZIONE: misurato in h0.py -> falsi positivi enormi (il peso contiene il nostro
             watermark e lo "fa ricomparire"). NON usare. Tenuto solo per documentazione.
  sq:<win>:<eps> : come sp ma con rho = 1 / (media locale di (attaccata - ORIGINALE)^2 + eps).
             Se l'attaccata non contiene il nostro watermark, rho non dipende da w -> nessun bias.
  pr       : (14/10) PREDIZIONE dell'attacco: T stimato da originale e attaccata (predict.py),
             estrazione da (attaccata - T(originale) + originale). Scelta di T indipendente da w.
  prsq:<win>:<eps> : pr + pesatura spaziale calcolata sul residuo (attaccata - T(originale)).
  rs_<det> : come <det> ma provando anche a "de-stirare" l'attaccata (resize del lab che esce
             a 510..514 px e viene tagliato/esteso): max sui candidati
"""
import cv2
import numpy as np
from scipy.ndimage import uniform_filter

import core
import predict

WIN = 15                       # finestra (in coefficienti DCT) per stimare guadagno e rumore
STRETCH = (510, 511, 513, 514)  # dimensioni "intermedie" del resize del lab da compensare


def destretch(img, m):
    """Il resize del lab con uscita a m px e taglio/estensione a 512 equivale a
    att(j) = wm((j+0.5)*512/m - 0.5). Qui facciamo l'inverso: U(x) = att((x+0.5)*m/512 - 0.5)."""
    x = (np.arange(512) + 0.5) * m / 512 - 0.5
    mx, my = np.meshgrid(x, x)
    return cv2.remap(img.astype(np.float32), mx.astype(np.float32), my.astype(np.float32),
                     cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


class Ctx:
    def __init__(self, original, wm, cfg):
        if cfg.get("transform", "dct") == "dwt":
            import core_dwt
            self.ex = core_dwt.Extractor(original)
            self.W_ext = self.ex(wm.astype(np.float64))
            return
        self.ex = core.Extractor(original, cfg["rmin"], cfg["rmax"])
        self.Xw = core.dct2(wm.astype(np.float64))
        self.W_ext = self.ex.from_dct(self.Xw)
        M = np.ones((512, 512))
        M[self.ex.uu, self.ex.vv] = 0
        M[self.ex.uu, self.ex.vv + 1] = 0
        self.M = M
        self.cnt = uniform_filter(M, WIN)
        self.Sww = uniform_filter(self.Xw ** 2 * M, WIN)
        ex = self.ex
        self.kvar = (1 / ex.a1 ** 2 + 1 / ex.a2 ** 2) / ex.den ** 2   # rumore coeff. -> rumore su W_i

    def y_cos(self, Xa):
        return self.ex.from_dct(Xa)

    def y_wcos(self, Xa):
        ex, M = self.ex, self.M
        Saw = uniform_filter(Xa * self.Xw * M, WIN)
        Saa = uniform_filter(Xa ** 2 * M, WIN)
        g = Saw / np.maximum(self.Sww, 1e-9)
        s2 = np.maximum((Saa - 2 * g * Saw + g * g * self.Sww) / np.maximum(self.cnt, 1e-9), 1e-6)
        gi = 0.5 * (g[ex.uu, ex.vv] + g[ex.uu, ex.vv + 1])
        vi = 0.5 * (s2[ex.uu, ex.vv] + s2[ex.uu, ex.vv + 1]) * self.kvar
        wgt = np.maximum(gi, 0) / vi
        return ex.from_dct(Xa) * wgt


def build(original, wm, cfg):
    c = Ctx(original, wm, cfg)
    o = original.astype(np.float64)
    w = wm.astype(np.float64)
    if cfg.get("demod"):
        # "demodulazione": la maschera spaziale dell'embedding dipende solo dall'originale,
        # quindi la detection la ricalcola e divide (immagine - originale) per la maschera
        # prima di estrarre. Cosi' W_ext torna a*w (niente mescolamento tra coefficienti).
        M = core.texture_mask(original, cfg["demod"])
        ex0 = c.ex
        c.ex = lambda img: ex0(o + (np.asarray(img, dtype=np.float64) - o) / M)
        c.W_ext = c.ex(w)

    memo = {}

    def pred(a):
        # la stessa attaccata viene passata a tutti i detector: predizione calcolata una volta sola
        key = (id(a), a.shape, float(a[::37, ::37].sum()))
        if memo.get('k') != key:
            memo['k'], memo['v'] = key, predict.predict_fast(original, np.clip(np.rint(a), 0, 255).astype(np.uint8))[0]   # versione veloce (14/10)
        return memo['v']

    def make(kind):
        """kind -> f(immagine attaccata float) -> similarita'"""
        if kind == "cos":
            return lambda a: core.cosine(c.W_ext, c.ex(a))
        if kind == "wcos":
            return lambda a: core.cosine(c.W_ext, c.y_wcos(core.dct2(a)))
        if kind.startswith("sp:"):
            _, win, eps = kind.split(":")
            win, eps = int(win), float(eps)

            def f(a):
                rho = 1.0 / (uniform_filter((a - w) ** 2, win) + eps)
                return core.cosine(c.W_ext, c.ex.from_dct(core.dct2(rho * (a - o) + o)))
            return f
        if kind == "pr":
            return lambda a: core.cosine(c.W_ext, c.ex(a - pred(a) + o))
        if kind.startswith("prsq:"):
            _, win, eps = kind.split(":")
            win, eps = int(win), float(eps)

            def f(a):
                r = a - pred(a)
                rho = 1.0 / (uniform_filter(r ** 2, win) + eps)
                return core.cosine(c.W_ext, c.ex(rho * r + o))
            return f
        if kind.startswith("sq:"):
            _, win, eps = kind.split(":")
            win, eps = int(win), float(eps)

            def f(a):
                dlt = a - o
                rho = 1.0 / (uniform_filter(dlt ** 2, win) + eps)
                return core.cosine(c.W_ext, c.ex(rho * dlt + o))
            return f
        raise ValueError(kind)

    out = {}
    for name in cfg["dets"]:
        rs = name.startswith("rs_")
        f = make(name[3:] if rs else name)
        if not rs:
            out[name] = (lambda f: lambda att: f(att.astype(np.float64)))(f)
        else:
            def g(att, f=f):
                s = f(att.astype(np.float64))
                for m in STRETCH:
                    s = max(s, f(destretch(att, m).astype(np.float64)))
                return s
            out[name] = g
    return out
