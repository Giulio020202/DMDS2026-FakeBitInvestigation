"""Verifica che core.py dia gli stessi numeri del codice ufficiale (WPSNR dei lab, embedding)."""
import sys, time
import cv2, numpy as np
import core
from core import emb

imgs = ["sample-images/0000.bmp", "sample-images/0070.bmp", "tree.bmp"]
w = 2.0 * np.load("fakebitinvestigation.npy").astype(float) - 1
rng = np.random.default_rng(0)
maxdiff = 0.0
for p in imgs:
    o = cv2.imread(p, 0)
    t = time.time(); ref = emb.embedding(p, "fakebitinvestigation.npy"); t_ref = time.time() - t
    t = time.time(); mine, a = core.embed(o, w); t_mine = time.time() - t
    same = np.array_equal(ref, mine)
    for _ in range(5):
        b = np.clip(ref.astype(float) + rng.normal(0, rng.uniform(1, 20), ref.shape), 0, 255).astype(np.uint8)
        maxdiff = max(maxdiff, abs(emb.wpsnr(ref, b) - core.wpsnr_fast(ref, b)))
    print(f"{p}: embedding identico={same} a={a:.3f} WPSNR lab={emb.wpsnr(o, ref):.4f} "
          f"tempo ufficiale {t_ref:.1f}s, veloce {t_mine:.2f}s")
print(f"max |WPSNR lab - WPSNR veloce| su 15 coppie = {maxdiff:.2e} dB")
