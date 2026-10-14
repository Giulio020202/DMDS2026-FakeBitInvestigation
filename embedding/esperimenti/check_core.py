"""Verifica che gli strumenti degli esperimenti diano gli stessi numeri del codice consegnato (v2.0 ibrida):
- WPSNR veloce (core.wpsnr_fast) = WPSNR dei lab;
- embedding di multi.py ('dct_dwt34') = embedding_fakebitinvestigation.embedding, bit per bit.
Uso (da embedding/): python esperimenti/check_core.py"""
import os, sys, time
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core, multi
from core import emb

imgs = ["sample-images/0000.bmp", "sample-images/0070.bmp", "gara2022/tree.bmp"]
w = 2.0 * np.load("fakebitinvestigation.npy").astype(float) - 1
rng = np.random.default_rng(0)
maxdiff = 0.0
for p in imgs:
    o = cv2.imread(p, 0)
    t = time.time(); ref = emb.embedding(p, "fakebitinvestigation.npy"); t_ref = time.time() - t
    t = time.time(); mine, s = multi.embed(o, w, multi.make_parts(o, multi.CONFIGS["dct_dwt34"])); t_mine = time.time() - t
    for _ in range(5):
        b = np.clip(ref.astype(float) + rng.normal(0, rng.uniform(1, 20), ref.shape), 0, 255).astype(np.uint8)
        maxdiff = max(maxdiff, abs(emb.wpsnr(ref, b) - core.wpsnr_fast(ref, b)))
    print(f"{p}: embedding identico={np.array_equal(ref, mine)} scala={s:.3f} WPSNR lab={emb.wpsnr(o, ref):.4f} "
          f"tempo ufficiale {t_ref:.1f}s, veloce {t_mine:.1f}s")
print(f"max |WPSNR lab - WPSNR veloce| su 15 coppie = {maxdiff:.2e} dB")
