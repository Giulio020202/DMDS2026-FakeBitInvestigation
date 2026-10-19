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

Uso della detection altrui: i TA hanno confermato (risposta alla domanda 6) che possiamo usare
l'esito della detection degli altri gruppi per guidare i nostri attacchi. Noi non usiamo mai
direttamente l'originale: lo riceve solo la loro detection, come input1.

Cartelle attese:
  --images      immagini watermarkate scaricate:   <gruppo>_<imageName>.bmp
  --originals   immagini originali:                <imageName>.bmp
  --detections  detection degli altri gruppi:      detection_<gruppo>.py
  --out         dove salvare:                      fakebitinvestigation_<gruppo>_<imageName>.bmp + log

Uso (da v2.0_ibrida/):
  python gara_fakebitinvestigation.py --images DIR --originals DIR --detections DIR --out DIR
         [--only gruppo] [--procs N] [--budget SECONDI] [--no-greedy]
"""
import argparse
import csv
import importlib.util
import os
import sys
import tempfile
import time
from multiprocessing import Pool

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from attacks_fakebitinvestigation import MASKS, attacks, describe       # noqa: E402
from detection_fakebitinvestigation import _get_csf_filter, wpsnr        # noqa: E402  (codice dei lab)

OUR_GROUP = "fakebitinvestigation"
MIN_WPSNR = 35.0

QF = [95, 90, 80, 70, 60, 50, 40, 30, 25, 20, 16, 13, 10, 8, 6, 5, 4, 3, 2, 1]
# famiglia -> (catena di nomi, funzione forza -> param_array, griglia dal piu' LEGGERO al piu' FORTE, tipo)
# tipo: 'int' / 'float' = si raffina tra due valori della griglia; 'none' = solo i valori della griglia
FAMILIES = {
    # --- singoli
    "jpeg": (["jpeg"], lambda p: [[p]], QF, "int"),
    "blur": (["blur"], lambda p: [[p]], [0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.6, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0], "float"),
    "blur-oriz": (["blur"], lambda p: [[(p / 3, p)]], [0.5, 0.8, 1.2, 1.6, 2.0, 3.0, 4.0, 6.0], "float"),
    "blur-vert": (["blur"], lambda p: [[(p, p / 3)]], [0.5, 0.8, 1.2, 1.6, 2.0, 3.0, 4.0, 6.0], "float"),
    "awgn": (["awgn"], lambda p: [[p, 7]], [2, 4, 6, 8, 10, 13, 16, 20, 25, 30, 40, 55, 70], "float"),
    "median": (["median"], lambda p: [[p]], [3, 5, 7, 9, 11, 13, 15], "none"),
    "median-oriz": (["median"], lambda p: [[(1, p)]], [3, 5, 7, 9, 11, 15], "none"),
    "median-vert": (["median"], lambda p: [[(p, 1)]], [3, 5, 7, 9, 11, 15], "none"),
    "resize-lab": (["resize"], lambda p: [[p, "lab"]], [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5,
                                                      0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15], "float-down"),
    "resize-cubic": (["resize"], lambda p: [[p, "cubic"]], [0.9, 0.8, 0.7, 0.6, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2],
                     "float-down"),
    "resize-area": (["resize"], lambda p: [[p, "area"]], [0.9, 0.8, 0.7, 0.6, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2],
                    "float-down"),
    # --- catene: il secondo attacco "rifinisce" il primo
    "blur0.6+jpeg": (["blur", "jpeg"], lambda p: [[0.6], [p]], QF[:-1], "int"),
    "blur1.0+jpeg": (["blur", "jpeg"], lambda p: [[1.0], [p]], QF[:-1], "int"),
    "jpeg+blur0.8": (["jpeg", "blur"], lambda p: [[p], [0.8]], QF[:-1], "int"),
    "median3+jpeg": (["median", "jpeg"], lambda p: [[3], [p]], QF[:-1], "int"),
    "awgn2+jpeg": (["awgn", "jpeg"], lambda p: [[2, 7], [p]], QF[:-1], "int"),
    "awgn4+jpeg": (["awgn", "jpeg"], lambda p: [[4, 7], [p]], QF[:-1], "int"),
    "jpeg50+jpeg": (["jpeg", "jpeg"], lambda p: [[50], [p]], QF[:-1], "int"),
    "resize0.5+jpeg": (["resize", "jpeg"], lambda p: [[0.5], [p]], [90, 70, 50, 40, 30, 20, 13, 8, 5, 3], "int"),
    "awgn+blur1": (["awgn", "blur"], lambda p: [[p, 7], [1.0]], [4, 8, 12, 16, 20, 25, 30, 40, 55], "float"),
    "awgn+median3": (["awgn", "median"], lambda p: [[p, 7], [3]], [4, 8, 12, 16, 20, 25, 30], "float"),
    # blur + sharpening a sigma piu' larga = passa-banda: le basse frequenze tornano su (WPSNR alto),
    # le medie (dove di solito sta il watermark) restano attenuate
    "blur+sharp4": (["blur", "sharp"], lambda p: [[p], [4, 1.5]], [0.5, 0.8, 1.0, 1.3, 1.6, 2.0, 2.5, 3.0, 4.0], "float"),
    "blur+sharp10": (["blur", "sharp"], lambda p: [[p], [10, 2]], [0.5, 0.8, 1.0, 1.3, 1.6, 2.0, 2.5, 3.0, 4.0], "float"),
    "median5+sharp": (["median", "sharp"], lambda p: [[5], [4, p]], [0.3, 0.6, 1.0, 1.5, 2.0], "float"),
    "blur0.8+jpeg+blur0.6": (["blur", "jpeg", "blur"], lambda p: [[0.8], [p], [0.6]], QF[:-1], "int"),
}
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
# ORACOLO: la detection del gruppo attaccato
# ----------------------------------------------------------------------------
class Oracle:
    def __init__(self, det_path, orig_path, wm_path, tmpdir):
        spec = importlib.util.spec_from_file_location(f"det_{os.getpid()}_{time.time_ns()}", det_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.det, self.orig, self.wm = mod.detection, orig_path, wm_path
        self.tmp = os.path.join(tmpdir, "att.bmp")
        self.calls = 0

    def broken(self, att):
        """True se la detection dice 'assente'. L'immagine passa per il BMP come quella caricata."""
        cv2.imwrite(self.tmp, att)
        self.calls += 1
        present, _ = self.det(self.orig, self.wm, self.tmp)
        return int(present) == 0


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
        else:                                      # 'float' (forza crescente) o 'float-down' (resize: forza decrescente)
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
    """Restituisce alla watermarkata i blocchi che non servono (prima 64x64, poi 32x32)."""
    cur = att.copy()
    for b in GREEDY_BLOCKS:
        diff = np.abs(cur.astype(int) - wm.astype(int))
        blocks = [(i, j) for i in range(0, 512, b) for j in range(0, 512, b) if diff[i:i + b, j:j + b].any()]
        # prima i blocchi che "costano" di piu' in WPSNR (i piu' modificati)
        blocks.sort(key=lambda ij: -diff[ij[0]:ij[0] + b, ij[1]:ij[1] + b].sum())
        for i, j in blocks:
            if time.time() > deadline:
                return cur
            trial = cur.copy()
            trial[i:i + b, j:j + b] = wm[i:i + b, j:j + b]
            if oracle.broken(trial):
                cur = trial
    return cur


def attack_one(job):
    fname, det_path, orig_path, wm_path, out_dir, budget, use_greedy = job
    group, image = fname[:-4].split("_", 1)
    t0 = time.time()
    deadline = t0 + budget
    wm = cv2.imread(wm_path, cv2.IMREAD_GRAYSCALE)
    with tempfile.TemporaryDirectory() as tmp:
        oracle = Oracle(det_path, orig_path, wm_path, tmp)
        found = []                                  # (wpsnr, descrizione, immagine)

        tries = [(fam, None) for fam in FAMILIES] + [(fam, m) for m in MASKS for fam in FAMILIES]
        for fam, mname in tries:
            if time.time() > deadline:
                break
            mask = MASKS[mname](wm) if mname else None
            r = search(oracle, wm, fam, mask, deadline)
            if r is not None:
                w, names, params = r
                found.append((w, describe(names, params, mname), attacks(wm, names, params, mask=mask)))
        if use_greedy and found:
            w, desc, att = max(found, key=lambda x: x[0])
            g = greedy_blocks(oracle, wm, att, deadline + budget)       # il greedy ha un suo budget
            if oracle.broken(g):
                found.append((wpsnr(wm, g), desc + " + zone (greedy, detection)", g))

        # tra i riusciti, il WPSNR piu' alto (ricontrollato con la detection sull'immagine salvata)
        found.sort(key=lambda x: -x[0])
        for w, desc, att in found:
            if w < MIN_WPSNR:
                continue
            if oracle.broken(att):
                out = os.path.join(out_dir, f"{OUR_GROUP}_{group}_{image}.bmp")
                cv2.imwrite(out, att)
                return image, group, w, desc, oracle.calls, time.time() - t0
    return image, group, None, None, oracle.calls, time.time() - t0


def main():
    ap = argparse.ArgumentParser(description="Attacchi di gara del gruppo fakebitinvestigation")
    ap.add_argument("--images", required=True)
    ap.add_argument("--originals", required=True)
    ap.add_argument("--detections", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default=None, help="attacca solo questo gruppo")
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--budget", type=float, default=900, help="secondi massimi per immagine (default 900)")
    ap.add_argument("--no-greedy", action="store_true", help="salta la fase delle zone guidate dalla detection")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    jobs = []
    for fname in sorted(os.listdir(a.images)):
        if not fname.endswith(".bmp") or "_" not in fname:
            continue
        group, image = fname[:-4].split("_", 1)
        if group == OUR_GROUP or (a.only and group != a.only):
            continue
        det_path = os.path.join(a.detections, f"detection_{group}.py")
        orig_path = os.path.join(a.originals, f"{image}.bmp")
        if not os.path.exists(det_path) or not os.path.exists(orig_path):
            print(f"[salto] {fname}: manca {det_path if not os.path.exists(det_path) else orig_path}")
            continue
        jobs.append((fname, det_path, orig_path, os.path.join(a.images, fname), a.out, a.budget,
                     not a.no_greedy))
    print(f"{len(jobs)} immagini da attaccare, {a.procs} processi, budget {a.budget:.0f} s per immagine")

    rows = []
    with Pool(min(a.procs, max(1, len(jobs)))) as p:
        for image, group, w, desc, calls, dt in p.imap_unordered(attack_one, jobs):
            if w is None:
                print(f"[non rotta] {group}_{image}  ({calls} chiamate, {dt:.0f} s)")
            else:
                print(f"[ROTTA] {group}_{image}: {w:.2f} dB con {desc}  ({calls} chiamate, {dt:.0f} s)")
                rows.append((image, group, f"{w:.2f}", desc))

    rows.sort(key=lambda r: (r[1], r[0]))
    log_path = os.path.join(a.out, "log_attacchi.csv")          # si apre con Excel
    with open(log_path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["Image", "Group", "WPSNR", "Attack(s) with parameters"])
        wr.writerows(rows)
    print(f"rotte {len(rows)}/{len(jobs)}; log: {log_path}")


if __name__ == "__main__":
    main()
