"""Controlli di conformita' di detection_fakebitinvestigation.py della versione consegnata (v2.0 ibrida),
regolamento aggiornato p.5 e p.8: originale -> 0, watermarkata -> 1, nessuna stampa, tempo, e stesse decisioni
degli esperimenti (multi.py, coseno concatenato sulle varianti).
Uso (da embedding/): python esperimenti/check_detection.py"""
import contextlib, io, os, sys, tempfile, time
import cv2, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import multi, battery as B                            # noqa: E402
import importlib.util                                 # noqa: E402
spec = importlib.util.spec_from_file_location("det", os.path.join(ROOT, "v2.0_ibrida", "detection_fakebitinvestigation.py"))
det = importlib.util.module_from_spec(spec); spec.loader.exec_module(det)

w = 2.0 * np.load(os.path.join(ROOT, "fakebitinvestigation.npy")).astype(float) - 1
tmp = tempfile.mkdtemp()
times, mism, n = [], 0, 0
for name in ["0000", "0070", "0042", "0090"]:
    o = cv2.imread(os.path.join(ROOT, "sample-images", name + ".bmp"), 0)
    parts = multi.make_parts(o, multi.CONFIGS["dct_dwt34"])
    wm, _ = multi.embed(o, w, parts)
    ref = [p(wm.astype(np.float64))[None, :] for p, _ in parts]
    po, pw = os.path.join(tmp, "o.bmp"), os.path.join(tmp, "w.bmp")
    cv2.imwrite(po, o); cv2.imwrite(pw, wm)
    cases = {"originale": o, "watermarkata": wm, "jpeg 6": B.jpeg(wm, 6), "jpeg 20": B.jpeg(wm, 20),
             "resize lab .55": B.resizing(wm, .55), "blur 5 lisce70": B.localize(wm, B.blur(wm, 5), B.masks(wm)["lisce70"]),
             "awgn 60 (distrutta)": B.awgn(wm, 60), "altra immagine": cv2.imread(os.path.join(ROOT, "gara2022", "tree.bmp"), 0)}
    for k, att in cases.items():
        pa = os.path.join(tmp, "a.bmp"); cv2.imwrite(pa, att)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            t = time.time(); o1, o2 = det.detection(po, pw, pa); times.append(time.time() - t)
        assert buf.getvalue() == "", "la detection ha stampato qualcosa!"
        V = multi.variants(o.astype(np.float64), att)
        s = float(multi.scores(ref, [np.stack([p(v) for v in V]) for p, _ in parts])[0][0])
        exp = int(s >= det.TAU)
        mism += exp != o1; n += 1
        print(f"{name} {k:20s} output1={o1} output2={o2:9.2f}  sim_esperimenti={s:+.3f}  {'OK' if exp == o1 else 'DIVERSO'}")
        if k == "originale": assert o1 == 0
        if k == "watermarkata": assert o1 == 1
print(f"\ndecisioni diverse dagli esperimenti: {mism}/{n}; tempo detection: medio {np.mean(times):.2f}s, max {np.max(times):.2f}s")
