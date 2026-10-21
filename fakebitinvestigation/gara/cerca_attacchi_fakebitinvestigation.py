"""
Capture the Mark 2026 - HELPER per la fase di attacco della gara (gruppo fakebitinvestigation). NON si consegna.

Non contiene nessun attacco: tutta la logica (attacchi ammessi, zone, famiglie, ricerca dell'attacco piu'
leggero con la detection altrui come oracolo, greedy a blocchi) e' in ../consegna/attacks_fakebitinvestigation.py,
funzione attack_image(). Questo file fa solo il "giro" su tutte le immagini scaricate:
  - abbina ogni immagine <gruppo>_<imageName>.bmp alla detection_<gruppo>.py e all'originale <imageName>.bmp;
  - divide il tempo totale tra le immagini e le attacca in parallelo (piu' processi);
  - scrive il log per i TA (Image | Group | WPSNR | Attack(s) with parameters) man mano, con --resume;
  - --check: controlla in un minuto che ogni detection si carichi e dica "presente" sulla watermarkata.

Cartelle attese:
  --images      immagini watermarkate scaricate:   <gruppo>_<imageName>.bmp
  --originals   immagini originali:                <imageName>.bmp (vanno SOLO alla loro detection, come input1)
  --detections  detection degli altri gruppi:      detection_<gruppo>.py
  --out         dove salvare:                      fakebitinvestigation_<gruppo>_<imageName>.bmp, log_attacchi.csv
                                                   (da consegnare), tutti_i_riusciti.csv (alternative)

Uso (da consegna/, come tutti i comandi della gara):
  python ../gara/cerca_attacchi_fakebitinvestigation.py --images DIR --originals DIR --detections DIR --out DIR
         [--only gruppo] [--procs N] [--total-min MINUTI] [--budget SECONDI] [--timeout SECONDI]
         [--no-greedy] [--resume] [--check]
"""
import os

# un thread per processo (i processi lavorano gia' in parallelo) e nessuna finestra di matplotlib:
# va fatto PRIMA di importare numpy/cv2
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse                                                          # noqa: E402
import csv                                                               # noqa: E402
import sys                                                               # noqa: E402
import tempfile                                                          # noqa: E402
import time                                                              # noqa: E402
from multiprocessing import Pool                                         # noqa: E402

import cv2                                                               # noqa: E402

cv2.setNumThreads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "consegna"))                # il codice degli attacchi e' quello consegnato
from attacks_fakebitinvestigation import OUR_GROUP, Oracle, attack_image  # noqa: E402


def attack_one(job):
    """Un'immagine: tutto il lavoro lo fa attack_image() del file degli attacchi."""
    fname, det_path, orig_path, wm_path, out_dir, budget, use_greedy, timeout = job
    return attack_image(wm_path, orig_path, det_path, out_dir, budget, use_greedy, timeout)


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
    ap.add_argument("--no-greedy", action="store_true",
                    help="salta le zone scelte con la detection altrui (punto 3; ammesso dai TA il 19/10)")
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
