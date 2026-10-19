"""
Capture the Mark 2026 - script di attacco per il giorno della gara (gruppo fakebitinvestigation)

Per ogni immagine watermarkata di un altro gruppo cerca l'attacco PIU' LEGGERO che toglie il
watermark (cioe' quello con il WPSNR piu' alto), lo salva con il nome richiesto e scrive il log
(Image | Group | WPSNR | Attack(s) with parameters). Attacco riuscito (p.5): watermark assente secondo
la detection del gruppo attaccato E WPSNR >= 35.

Usa solo gli attacchi ammessi (attacks_fakebitinvestigation.py: AWGN, blur, sharpening, JPEG,
resize, median, anche in catena o localizzati). Le zone degli attacchi localizzati sono calcolate
SOLO dalla watermarkata.

Come cerca (per ogni immagine):
  1. FAMIGLIE: ogni famiglia e' un attacco (o una catena) con un parametro di forza e una griglia
     dal piu' leggero al piu' forte. Il WPSNR (veloce, via FFT, identico a quello dei lab) scarta
     subito i valori sotto 35 dB senza chiamare la detection. Poi:
       - si prova il valore piu' forte ancora sopra 35 dB: se non rompe, la famiglia e' inutile (1 chiamata);
       - ricerca binaria del valore piu' leggero che rompe (~4 chiamate);
       - raffinamento continuo tra l'ultimo che non rompe e il primo che rompe (~3 chiamate).
  2. LOCALIZZATI: le stesse famiglie applicate solo dentro una zona (MASKS). Fuori dalla zona
     l'immagine resta intatta, quindi dentro si puo' andare molto piu' forte restando sopra 35 dB.
  3. ZONE GUIDATE DALLA DETECTION (spento con --no-greedy): dal miglior attacco trovato si
     "restituiscono" alla watermarkata i blocchi 64x64 e poi 32x32 che non servono a rompere.
Si tiene il riuscito con il WPSNR piu' alto (p.7: se ne carica uno solo per immagine).
Le famiglie e le zone sono in ordine di efficacia (dalle nostre prove): se il tempo finisce, si
perdono solo i tentativi meno promettenti.

Uso della detection altrui: i TA hanno confermato (risposta alla domanda 6) che possiamo usare
l'esito della detection degli altri gruppi per guidare i nostri attacchi. Noi non usiamo mai
direttamente l'originale: lo riceve solo la loro detection, come input1.

Tempo: --total-min e' il tempo totale a disposizione (default 100 minuti, la fase di attacco dura
2h15). Il tempo per immagine viene calcolato da quello, dal numero di immagini e dai processi:
75% alla ricerca (punti 1-2), il resto al greedy (punto 3).

Protezioni contro le detection degli altri: timeout per chiamata (--timeout), errori catturati (dopo
3 errori di fila l'immagine viene saltata e segnalata), stampe soppresse, nessuna finestra.
Il log viene scritto man mano; con --resume si riparte saltando le immagini gia' nel log.

Cartelle attese:
  --images      immagini watermarkate scaricate:   <gruppo>_<imageName>.bmp
  --originals   immagini originali:                <imageName>.bmp
  --detections  detection degli altri gruppi:      detection_<gruppo>.py
  --out         dove salvare:                      fakebitinvestigation_<gruppo>_<imageName>.bmp, log_attacchi.csv
                                                   (da consegnare), tutti_i_riusciti.csv (alternative)

Uso (da v2.0_ibrida/):
  python attacchi_fakebitinvestigation.py --images DIR --originals DIR --detections DIR --out DIR
         [--only gruppo] [--procs N] [--total-min MINUTI] [--budget SECONDI] [--timeout SECONDI]
         [--no-greedy] [--resume]
"""
import os

# un thread per processo (i processi lavorano gia' in parallelo) e nessuna finestra di matplotlib:
# va fatto PRIMA di importare numpy/cv2
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse                                                          # noqa: E402
import contextlib                                                        # noqa: E402
import csv                                                               # noqa: E402
import importlib.util                                                    # noqa: E402
import signal                                                            # noqa: E402
import sys                                                               # noqa: E402
import tempfile                                                          # noqa: E402
import time                                                              # noqa: E402
import traceback                                                         # noqa: E402
from multiprocessing import Pool                                         # noqa: E402

import cv2                                                               # noqa: E402
import numpy as np                                                       # noqa: E402

cv2.setNumThreads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from attacks_fakebitinvestigation import MASKS, attacks, describe       # noqa: E402
from detection_fakebitinvestigation import _get_csf_filter, wpsnr        # noqa: E402  (codice dei lab)

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


def attack_one(job):
    """Attacca un'immagine. Non solleva mai eccezioni: in caso di problemi restituisce l'errore."""
    fname, det_path, orig_path, wm_path, out_dir, budget, use_greedy, timeout = job
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


LOG_HEADER = ["Image", "Group", "WPSNR", "Attack(s) with parameters"]


def check_one(job):
    """Controllo rapido di una detection: si carica? sulla watermarkata dice 'presente'? quanto ci mette?"""
    fname, det_path, orig_path, wm_path, timeout = job
    try:
        with tempfile.TemporaryDirectory() as tmp:
            o = Oracle(det_path, orig_path, wm_path, tmp, timeout)
            t = time.time()
            broken = o.broken(cv2.imread(wm_path, cv2.IMREAD_GRAYSCALE))
            dt = time.time() - t
            if o.errors:
                return fname, f"ERRORE: {o.last_error}"
            if broken:
                return fname, f"ATTENZIONE: dice 'assente' sulla watermarkata ({dt:.2f} s)"
            return fname, f"ok, {dt:.2f} s per chiamata"
    except Exception as e:                             # noqa: BLE001
        return fname, f"ERRORE: {type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser(description="Attacchi di gara del gruppo fakebitinvestigation")
    ap.add_argument("--images", required=True)
    ap.add_argument("--originals", required=True)
    ap.add_argument("--detections", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default=None, help="attacca solo questo gruppo")
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--total-min", type=float, default=100, help="minuti totali a disposizione (default 100)")
    ap.add_argument("--budget", type=float, default=None, help="secondi per immagine (default: da --total-min)")
    ap.add_argument("--timeout", type=float, default=10, help="secondi massimi per una chiamata alla detection")
    ap.add_argument("--no-greedy", action="store_true", help="salta la fase delle zone guidate dalla detection")
    ap.add_argument("--resume", action="store_true", help="salta le immagini gia' presenti nel log")
    ap.add_argument("--check", action="store_true",
                    help="solo controllo: ogni detection si carica, dice 'presente' sulla watermarkata, tempo")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    log_path = os.path.join(a.out, "log_attacchi.csv")
    all_path = os.path.join(a.out, "tutti_i_riusciti.csv")

    done = set()
    if a.check:
        pass                                           # il controllo non tocca i log
    elif a.resume and os.path.exists(log_path):
        with open(log_path, newline="") as f:
            done = {(r["Group"], r["Image"]) for r in csv.DictReader(f)}
    else:
        for path, header in ((log_path, LOG_HEADER), (all_path, ["Image", "Group", "WPSNR", "Attack"])):
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(header)

    jobs = []
    for fname in sorted(os.listdir(a.images)):
        if not fname.endswith(".bmp") or "_" not in fname:
            continue
        group, image = fname[:-4].split("_", 1)
        if group == OUR_GROUP or (a.only and group != a.only) or (group, image) in done:
            continue
        det_path = os.path.join(a.detections, f"detection_{group}.py")
        orig_path = os.path.join(a.originals, f"{image}.bmp")
        if not os.path.exists(det_path) or not os.path.exists(orig_path):
            print(f"[salto] {fname}: manca {det_path if not os.path.exists(det_path) else orig_path}")
            continue
        jobs.append([fname, det_path, orig_path, os.path.join(a.images, fname), a.out, 0, not a.no_greedy, a.timeout])
    if not jobs:
        sys.exit("nessuna immagine da attaccare")

    if a.check:
        print(f"controllo di {len(jobs)} immagini ({len({j[0].split('_', 1)[0] for j in jobs})} gruppi):")
        with Pool(max(1, min(a.procs, len(jobs)))) as p:
            for fname, msg in p.imap(check_one, [(j[0], j[1], j[2], j[3], a.timeout) for j in jobs]):
                print(f"  {fname:40s} {msg}", flush=True)
        return

    procs = max(1, min(a.procs, len(jobs)))
    rounds = -(-len(jobs) // procs)                    # giri di immagini in parallelo (arrotondato per eccesso)
    budget = a.budget if a.budget else 0.95 * a.total_min * 60 / rounds
    for j in jobs:
        j[5] = budget
    print(f"{len(jobs)} immagini, {procs} processi ({rounds} giri), budget {budget / 60:.1f} min per immagine, "
          f"tempo massimo previsto ~{rounds * budget / 60:.0f} min", flush=True)

    t0, broken, failed = time.time(), 0, []
    with Pool(procs) as p:
        for k, r in enumerate(p.imap_unordered(attack_one, [tuple(j) for j in jobs]), 1):
            tag = f"[{k}/{len(jobs)} {(time.time() - t0) / 60:.0f} min]"
            info = f"({r['calls']} chiamate, {r['errors']} errori, {r['time']:.0f} s)"
            if r["error"]:
                failed.append(f"{r['group']}_{r['image']}: {r['error']}")
                print(f"{tag} [ERRORE] {r['group']}_{r['image']}: {r['error']} {info}", flush=True)
            elif r["wpsnr"] is None:
                print(f"{tag} [non rotta] {r['group']}_{r['image']} {info}", flush=True)
            else:
                broken += 1
                print(f"{tag} [ROTTA] {r['group']}_{r['image']}: {r['wpsnr']:.2f} dB con {r['desc']} {info}",
                      flush=True)
                with open(log_path, "a", newline="") as f:
                    csv.writer(f).writerow([r["image"], r["group"], f"{r['wpsnr']:.2f}", r["desc"]])
            with open(all_path, "a", newline="") as f:
                csv.writer(f).writerows([[r["image"], r["group"], w, d] for w, d in r["found"]])

    print(f"\nrotte {broken}/{len(jobs)} in {(time.time() - t0) / 60:.1f} min; log: {log_path}")
    if failed:
        print("immagini con errori (da guardare a mano):\n  " + "\n  ".join(failed))


if __name__ == "__main__":
    main()
