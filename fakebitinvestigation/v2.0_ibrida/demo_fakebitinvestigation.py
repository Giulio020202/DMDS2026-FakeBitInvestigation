"""
Capture the Mark 2026 - DEMO del gruppo fakebitinvestigation (versione 2.0 ibrida DCT + wavelet)

Mostra come si usa ogni file consegnato:
  1. embedding_fakebitinvestigation.py   embedding(originale, watermark) -> watermarkata
  2. detection_fakebitinvestigation.py   detection(originale, watermarkata, attaccata) -> (presente?, WPSNR)
  3. attacks_fakebitinvestigation.py     attacks(watermarkata, nome, parametri) -> attaccata
  4. attacchi_fakebitinvestigation.py        ricerca automatica dell'attacco piu' leggero (con --attacchi)
  5. roc_fakebitinvestigation.py         ROC e soglia tau (con --roc, versione veloce)

Uso (da v2.0_ibrida/):
  python demo_fakebitinvestigation.py [--image FILE.bmp] [--watermark FILE.npy] [--out DIR] [--attacchi] [--roc]
Default: ../sample-images/0000.bmp, ../fakebitinvestigation.npy, cartella ./demo_output
"""
import argparse
import os
import shutil
import subprocess
import sys
import time

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from embedding_fakebitinvestigation import embedding                    # noqa: E402
from detection_fakebitinvestigation import TAU, detection, wpsnr        # noqa: E402
from attacks_fakebitinvestigation import MASKS, attacks, describe       # noqa: E402

GROUP = "fakebitinvestigation"

# (catena, parametri, maschera): dai piu' leggeri ai piu' pesanti
DEMO_ATTACKS = [
    ("jpeg", [50], None),
    ("jpeg", [10], None),
    ("blur", [0.8], None),
    ("awgn", [8], None),
    ("sharp", [1, 1], None),
    ("median", [5], None),
    ("resize", [0.5], None),
    (["blur", "jpeg"], [[0.6], [15]], None),
    ("jpeg", [5], "lisce50"),
    ("awgn", [40], "lisce70"),
    ("jpeg", [2], None),            # attacco distruttivo: WPSNR < 35, non conta come riuscito
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=os.path.join(HERE, "..", "sample-images", "0000.bmp"))
    ap.add_argument("--watermark", default=os.path.join(HERE, "..", "fakebitinvestigation.npy"))
    ap.add_argument("--out", default=os.path.join(HERE, "demo_output"))
    ap.add_argument("--attacchi", action="store_true", help="prova anche lo script di attacco contro la nostra detection")
    ap.add_argument("--roc", action="store_true", help="lancia anche la ROC veloce (--quick)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    name = os.path.splitext(os.path.basename(a.image))[0]

    # ---------------------------------------------------------------- 1. embedding
    print("=== 1. EMBEDDING ===")
    t = time.time()
    wm = embedding(a.image, a.watermark)
    original = cv2.imread(a.image, cv2.IMREAD_GRAYSCALE)
    p_wm = os.path.join(a.out, f"{GROUP}_{name}.bmp")
    cv2.imwrite(p_wm, wm)
    print(f"embedding({os.path.basename(a.image)}, {os.path.basename(a.watermark)}) in {time.time() - t:.2f} s")
    print(f"WPSNR(originale, watermarkata) = {wpsnr(original, wm):.2f} dB   -> salvata in {p_wm}")

    # ---------------------------------------------------------------- 2. detection
    print(f"\n=== 2. DETECTION (tau = {TAU}) ===")
    print(f"{'immagine data come attaccata':66s} {'output1':>7s} {'output2 (WPSNR)':>16s} {'tempo':>7s}  esito")

    def run(label, path, is_original=False):
        t = time.time()
        o1, o2 = detection(a.image, p_wm, path)
        dt = time.time() - t
        if is_original:
            esito = "corretto: sull'originale il watermark e' assente" if o1 == 0 else "ERRORE: falso positivo"
        elif o1 == 0 and o2 >= 35:
            esito = "ATTACCO RIUSCITO"
        elif o1 == 0:
            esito = "assente, ma WPSNR < 35: attacco non valido"
        else:
            esito = "watermark presente"
        print(f"{label:66s} {o1:7d} {o2:16.2f} {dt:6.2f}s  {esito}")

    run("watermarkata (nessun attacco)", p_wm)
    run("originale (deve dare 0)", a.image, is_original=True)

    # ---------------------------------------------------------------- 3. attacchi
    print("\n=== 3. ATTACCHI (attacks_fakebitinvestigation.py) + detection ===")
    for k, (names, params, mname) in enumerate(DEMO_ATTACKS):
        mask = MASKS[mname](wm) if mname else None
        att = attacks(p_wm, names, params, mask=mask)
        p_att = os.path.join(a.out, f"{GROUP}_demo_{name}_{k:02d}.bmp")
        cv2.imwrite(p_att, att)
        run(describe(names, params, mname), p_att)

    # ---------------------------------------------------------------- 4. attacchi (facoltativo)
    if a.attacchi:
        print("\n=== 4. SCRIPT DI ATTACCO contro la nostra detection (come se fossimo un altro gruppo 'demo') ===")
        g = os.path.join(a.out, "attacchi")
        for sub in ("img", "orig", "det", "out"):
            os.makedirs(os.path.join(g, sub), exist_ok=True)
        shutil.copy(p_wm, os.path.join(g, "img", f"demo_{name}.bmp"))
        shutil.copy(a.image, os.path.join(g, "orig", f"{name}.bmp"))
        shutil.copy(os.path.join(HERE, "detection_fakebitinvestigation.py"), os.path.join(g, "det", "detection_demo.py"))
        sys.stdout.flush()
        subprocess.run([sys.executable, os.path.join(HERE, "attacchi_fakebitinvestigation.py"),
                        "--images", os.path.join(g, "img"), "--originals", os.path.join(g, "orig"),
                        "--detections", os.path.join(g, "det"), "--out", os.path.join(g, "out"),
                        "--procs", "1", "--budget", "300"], check=True)

    # ---------------------------------------------------------------- 5. ROC (facoltativo)
    if a.roc:
        print("\n=== 5. ROC veloce (la tau vera viene dalla run completa, vedi README) ===")
        sys.stdout.flush()
        subprocess.run([sys.executable, os.path.join(HERE, "roc_fakebitinvestigation.py"), "--quick",
                        "--watermark", a.watermark, "--out", os.path.join(a.out, "roc_quick.png")], check=True)

    print(f"\nTutti i file della demo sono in {a.out}")


if __name__ == "__main__":
    main()
