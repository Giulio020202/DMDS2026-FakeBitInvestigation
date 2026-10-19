"""
Capture the Mark 2026 - aiuto per la FASE DI DIFESA del giorno della gara (gruppo fakebitinvestigation)

Serve a scegliere il target di WPSNR di ciascuna delle 3 immagini (es. 58.05 o 62.05 dB) e a produrre
le immagini da caricare. Lavora solo sulle NOSTRE immagini con la NOSTRA detection.

Modalita' 1 - PROVA (default):
  per ogni immagine e ogni target:
    - embedding a quel target;
    - controlli: watermarkata -> 1, originale -> 0, tempo della detection;
    - lo stesso attacco dello script di attacco (attacchi_fakebitinvestigation.py) contro la nostra detection:
      e' un avversario forte che cerca l'attacco piu' leggero che ci rompe.
  Poi prova tutte le combinazioni di target e stima i punti di difesa con la tabella del regolamento:
    - qualita' sulla MEDIA dei WPSNR delle 3 immagini (risposta dei TA);
    - robustezza sul WPSNR medio degli attacchi riusciti; 6 + 2 di bonus se nessuna viene rotta.
  Stampa la combinazione migliore e la riga TARGET_BY_IMAGE da copiare nell'embedding.
  ATTENZIONE: e' UN solo avversario; in gara ci attaccano 9 gruppi, quindi il bonus e' meno probabile
  di quanto sembra qui.

Modalita' 2 - FINALE (--final):
  dopo aver scritto TARGET_BY_IMAGE in embedding_fakebitinvestigation.py, rifa' l'embedding con il codice
  cosi' com'e' (nessun parametro passato da qui), controlla WPSNR e detection e salva in <out>/upload/
  le immagini da caricare con il nome del regolamento: fakebitinvestigation_<imageName>.bmp.

Uso (da v2.0_ibrida/):
  python difesa_fakebitinvestigation.py --images DIR_CON_LE_3_IMMAGINI --out DIR [--targets 58.05 62.05]
                                        [--budget SECONDI] [--procs N]
  python difesa_fakebitinvestigation.py --images DIR_CON_LE_3_IMMAGINI --out DIR --final
"""
import argparse
import itertools
import os
import sys
import time
from multiprocessing import Pool

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import embedding_fakebitinvestigation as emb                             # noqa: E402
from detection_fakebitinvestigation import TAU, detection, wpsnr        # noqa: E402
from attacchi_fakebitinvestigation import attack_one                        # noqa: E402

GROUP = "fakebitinvestigation"
DETECTION = os.path.join(HERE, "detection_fakebitinvestigation.py")


def quality_points(w):
    for lo, pts in ((66, 6), (62, 5), (58, 4), (54, 3), (50, 2), (35, 1)):
        if w >= lo:
            return pts
    return 0


def robustness_points(w):
    for hi, pts in ((38, 6), (41, 5), (44, 4), (47, 3), (50, 2), (53, 1)):
        if w < hi:
            return pts
    return 0


def checks(orig_path, wm_path):
    """watermarkata -> 1, originale -> 0, tempo della detection."""
    t = time.time()
    on_wm, _ = detection(orig_path, wm_path, wm_path)
    dt = time.time() - t
    on_orig, _ = detection(orig_path, wm_path, orig_path)
    ok = on_wm == 1 and on_orig == 0 and dt < 5
    return ok, f"watermarkata->{on_wm}, originale->{on_orig}, detection {dt:.2f} s"


def embed_to(orig_path, wm_file, target, out_path):
    emb.TARGET_BY_IMAGE = {}
    emb.TARGET_WPSNR = target
    wm = emb.embedding(orig_path, wm_file)
    cv2.imwrite(out_path, wm)
    return wpsnr(cv2.imread(orig_path, cv2.IMREAD_GRAYSCALE), wm)


def main():
    ap = argparse.ArgumentParser(description="Fase di difesa: scelta dei target e immagini da caricare")
    ap.add_argument("--images", required=True, help="cartella con le 3 immagini originali scaricate")
    ap.add_argument("--watermark", default=os.path.join(HERE, "..", "fakebitinvestigation.npy"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--targets", type=float, nargs="+", default=[58.05, 62.05])
    ap.add_argument("--budget", type=float, default=900, help="secondi di attacco per immagine e target (default 15 min)")
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--final", action="store_true", help="produce le immagini da caricare con il codice attuale")
    a = ap.parse_args()
    files = sorted(f for f in os.listdir(a.images) if f.lower().endswith(".bmp"))
    if not files:
        sys.exit(f"nessuna immagine .bmp in {a.images}")
    os.makedirs(a.out, exist_ok=True)

    # ------------------------------------------------------------ modalita' FINALE
    if a.final:
        up = os.path.join(a.out, "upload")
        os.makedirs(up, exist_ok=True)
        print(f"TARGET_WPSNR = {emb.TARGET_WPSNR}, TARGET_BY_IMAGE = {emb.TARGET_BY_IMAGE}, TAU = {TAU}")
        ws, all_ok = [], True
        for f in files:
            name = os.path.splitext(f)[0]
            orig = os.path.join(a.images, f)
            out = os.path.join(up, f"{GROUP}_{name}.bmp")
            wm = emb.embedding(orig, a.watermark)            # codice cosi' com'e', nessun parametro da qui
            cv2.imwrite(out, wm)
            w = wpsnr(cv2.imread(orig, cv2.IMREAD_GRAYSCALE), wm)
            ok, msg = checks(orig, out)
            all_ok &= ok
            ws.append(w)
            print(f"  {name:20s} WPSNR {w:.3f} dB  {msg}  {'OK' if ok else '!! CONTROLLARE !!'}")
        print(f"media WPSNR {np.mean(ws):.3f} dB -> {quality_points(np.mean(ws))} punti qualita'")
        print(f"\nimmagini da caricare in {up}" if all_ok else "\n!! qualche controllo NON e' passato: non caricare !!")
        return

    # ------------------------------------------------------------ modalita' PROVA
    print(f"=== embedding a {a.targets} dB e controlli (tau = {TAU}) ===")
    jobs, wps = [], {}
    for t in a.targets:
        d_img = os.path.join(a.out, f"target_{t:.2f}", "img")
        os.makedirs(d_img, exist_ok=True)
        for f in files:
            name = os.path.splitext(f)[0]
            orig = os.path.join(a.images, f)
            # nome "<gruppo>_<immagine>" come lo vuole lo script di attacco; il "gruppo" qui e' il target
            wm_path = os.path.join(d_img, f"t{t:.2f}_{name}.bmp")
            wps[(name, t)] = embed_to(orig, a.watermark, t, wm_path)
            ok, msg = checks(orig, wm_path)
            print(f"  {name:20s} target {t:.2f}: WPSNR {wps[(name, t)]:.3f} dB, {msg}  {'' if ok else '!! CONTROLLARE !!'}")
            d_out = os.path.join(a.out, f"target_{t:.2f}", "attacchi")
            os.makedirs(d_out, exist_ok=True)
            jobs.append((os.path.basename(wm_path), DETECTION, orig, wm_path, d_out, a.budget, True, 10.0))

    print(f"\n=== attacco contro di noi: {len(jobs)} prove, {a.budget / 60:.0f} min ciascuna al massimo ===", flush=True)
    res = {}
    with Pool(max(1, min(a.procs, len(jobs)))) as p:
        for r in p.imap_unordered(attack_one, jobs):
            name, t = r["image"], float(r["group"][1:])
            res[(name, t)] = r["wpsnr"]                       # None = non rotta
            esito = f"ROTTA a {r['wpsnr']:.2f} dB con {r['desc']}" if r["wpsnr"] else "non rotta"
            if r["error"]:
                esito = f"ERRORE: {r['error']}"
            print(f"  {name:20s} target {t:.2f}: {esito}  ({r['calls']} chiamate, {r['time']:.0f} s)", flush=True)

    names = [os.path.splitext(f)[0] for f in files]
    print("\n=== combinazioni di target (punti di difesa contro questo avversario) ===")
    rows = []
    for combo in itertools.product(a.targets, repeat=len(names)):
        q = quality_points(np.mean([wps[(n, t)] for n, t in zip(names, combo)]))
        att = [res[(n, t)] for n, t in zip(names, combo) if res.get((n, t))]
        r = 6 + 2 if not att else robustness_points(np.mean(att))
        rows.append((q + r, q, r, len(att), combo))
    rows.sort(key=lambda x: (-x[0], -x[1]))
    print(f"  {'totale':>6s} {'qualita':>7s} {'robust.':>7s} {'rotte':>5s}  target per immagine")
    for tot, q, r, n_br, combo in rows[:8]:
        print(f"  {tot:6d} {q:7d} {r:7d} {n_br:5d}  " + ", ".join(f"{n}={t:.2f}" for n, t in zip(names, combo)))
    mapping = {n: t for n, t in zip(names, rows[0][4])}
    print("\nProposta (da valutare: un solo avversario simulato, in gara sono 9). Riga per l'embedding:")
    print(f"  TARGET_BY_IMAGE = {mapping}")
    print("La detection NON va toccata (tau non dipende dal target). Poi:\n"
          f"  python difesa_fakebitinvestigation.py --images {a.images} --out {a.out} --final")


if __name__ == "__main__":
    main()
