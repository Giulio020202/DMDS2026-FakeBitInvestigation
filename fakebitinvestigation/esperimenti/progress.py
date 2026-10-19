"""Barra di avanzamento comune agli script degli esperimenti (scrive su stderr, quindi finisce anche nei .log)."""
import sys
import time


def pmap(pool, func, items, label=""):
    """Come pool.map (stesso ordine dei risultati), ma con barra di avanzamento e tempo stimato."""
    items = list(items)
    n, t0, out = len(items), time.time(), []
    for k, r in enumerate(pool.imap(func, items, chunksize=1), 1):
        out.append(r)
        el = time.time() - t0
        sys.stderr.write(f"\r{label} [{'#' * (30 * k // n):30s}] {k}/{n}  {el / 60:.1f} min, "
                         f"mancano ~{el / k * (n - k) / 60:.1f} min ")
        sys.stderr.flush()
    sys.stderr.write("\n")
    return out
