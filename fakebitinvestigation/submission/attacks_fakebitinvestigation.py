"""
Capture the Mark 2026 - attacks, group fakebitinvestigation

All our attack code is in this file, which does not import any other file of ours:
  1. the allowed attacks (AWGN, blur, sharpening, JPEG, resize, median), alone, in a chain or only
     on a zone of the image, through attacks(), which follows the example structure of the rules;
  2. the zones for the localized attacks, computed only from the watermarked image;
  3. attack_image(), the automatic search of the lightest attack that removes the watermark from an image:
     it tries families of attacks from the lightest to the strongest, using the outcome of the attacked
     group's detection to check each try, and then gives back to the watermarked image the blocks
     that are not needed to break it. The TAs confirmed by email that both steps are allowed.
The original image is never read or used to build the attack: we only pass its path to the
attacked group's detection, as its input1.
The only "external" function is the lab WPSNR, copied below.

Interface (example structure of the rules, p.6):
    output1 = attacks(input1, attack_name, param_array)
  input1      = path of the watermarked image (or already a uint8 512x512 array)
  attack_name = name of an attack ('awgn', 'blur', 'sharp', 'jpeg', 'resize', 'median')
                or a LIST of names = chain, applied in the given order
  param_array = parameters of the attack (list); for a chain, a list of lists
  mask        = (optional) boolean 512x512 mask: the attack is applied only there,
                outside the watermarked image stays. Masks are computed ONLY from the watermarked image.
  output1     = attacked image, uint8 512x512 array

Parameters of each attack:
  'awgn'   [std, seed]        gaussian noise, standard deviation in gray levels (seed optional)
  'blur'   [sigma]            gaussian filter; sigma can be a pair (sy, sx) = anisotropic blur
  'sharp'  [sigma, alpha]     unsharp masking: img + alpha * (img - blur(img, sigma))
  'jpeg'   [QF]               JPEG compression with quality factor 1..100
  'resize' [scale, method]    shrinks by 'scale' and goes back to 512x512; method = 'lab' (default,
                              skimage as in the labs), 'cubic', 'linear' or 'area' (OpenCV)
  'median' [k]                k x k median filter (or pair (ky, kx)), reflected borders

Examples from Python:
    attacks("groupX_lena.bmp", "jpeg", [20])
    attacks("groupX_lena.bmp", ["blur", "jpeg"], [[0.6], [12]])
    attacks(wm, "awgn", [30], mask=smooth_mask(wm, 0.5))     # strong noise only on the smoothest 50%
    attack_image("groupX_lena.bmp", "lena.bmp", "detection_groupX.py", "out/", budget=600)

From the command line:
    # one attack (or a chain) on an image, saved in OUT.bmp; prints the WPSNR
    python attacks_fakebitinvestigation.py apply WATERMARKED.bmp OUT.bmp jpeg 20
    python attacks_fakebitinvestigation.py apply WATERMARKED.bmp OUT.bmp blur,jpeg 0.6 12
    # automatic search of the lightest attack that breaks ONE image, with the detection of the attacked group
    python attacks_fakebitinvestigation.py search --image groupX_lena.bmp --original lena.bmp \
           --detection detection_groupX.py --out OUTDIR [--budget SECONDS] [--no-greedy]

The base functions are the lab ones (LAB-2 / LAB3-ROC) with two differences, needed to work
on uint8 like a real BMP: the input becomes float before blur and sharpening (with uint8 the
lab sharpening overflows) and the output of every attack is rounded and clipped to uint8.
"""
import argparse
import contextlib
import importlib.util
import io
import os
import signal
import sys
import tempfile
import time
import traceback

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, median_filter, uniform_filter
from scipy.signal import convolve2d
from skimage.transform import rescale


# ----------------------------------------------------------------------------
# WPSNR: identical copy of the code provided in the labs
# ----------------------------------------------------------------------------
def _csffun(u, v):
    f = np.sqrt(u**2 + v**2)
    w = 2 * np.pi * f / 60
    sigma = 2
    Sw = 1.5 * np.exp(-sigma**2 * w**2 / 2) - np.exp(-2 * sigma**2 * w**2 / 2)
    sita = np.arctan2(v, u)
    bita = 8
    f0 = 11.13
    w0 = 2 * np.pi * f0 / 60
    exp_term = np.exp(bita * (w - w0))
    Ow = (1 + exp_term * (np.cos(2 * sita))**4) / (1 + exp_term)
    return Sw * Ow


def _csfmat():
    min_f, max_f, step_f = -20, 20, 1
    freq_range = np.arange(min_f, max_f + step_f, step_f)
    u, v = np.meshgrid(freq_range, freq_range, indexing='xy')
    return _csffun(u, v)


def _get_csf_filter():
    Fmat = _csfmat()
    filter_coeffs = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(Fmat)))
    return np.real(filter_coeffs)


def wpsnr(image_a, image_b):
    if image_a.shape != image_b.shape:
        raise ValueError("Input images must have the same dimensions.")
    if image_a.dtype != np.float64 and image_a.dtype != np.float32:
        A = image_a.astype(np.float64) / 255.0
    else:
        A = image_a.copy()
    if image_b.dtype != np.float64 and image_b.dtype != np.float32:
        B = image_b.astype(np.float64) / 255.0
    else:
        B = image_b.copy()
    if A.max() > 1.0 or A.min() < 0.0 or B.max() > 1.0 or B.min() < 0.0:
        raise ValueError("Input image values must be in the interval [0, 1] for floats or [0, 255] for integers.")
    if np.array_equal(A, B):
        return 9999999.0
    error_image = A - B
    csf_filter = _get_csf_filter()
    weighted_error = convolve2d(error_image, csf_filter, mode='same', boundary='wrap')
    wmse = np.mean(weighted_error**2)
    if wmse == 0:
        return 9999999.0
    return 20 * np.log10(1.0 / np.sqrt(wmse))


def _u8(x):
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------
# BASE ATTACKS
# ----------------------------------------------------------------------------
def awgn(img, std, seed=123):
    rng = np.random.default_rng(seed)
    return _u8(img.astype(float) + rng.normal(0.0, std, img.shape))


def blur(img, sigma):
    return _u8(gaussian_filter(img.astype(float), sigma))


def sharpening(img, sigma, alpha):
    f = img.astype(float)
    return _u8(f + alpha * (f - gaussian_filter(f, sigma)))


def jpeg_compression(img, qf):
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, "JPEG", quality=int(qf))
    buf.seek(0)
    return np.asarray(Image.open(buf), dtype=np.uint8)


def resizing(img, scale):
    """Lab resize: rescale down and up with skimage, then crop to 512x512.
    For some scales (e.g. 0.7) the result is a bit smaller: we repeat the last row/column,
    so that the WPSNR can always be computed."""
    x, y = img.shape
    a = rescale(img.astype(float), scale)
    a = rescale(a, 1 / scale)[:x, :y]
    if a.shape != (x, y):
        a = np.pad(a, ((0, x - a.shape[0]), (0, y - a.shape[1])), mode="edge")
    return _u8(a)


_CV_INTERP = {"cubic": cv2.INTER_CUBIC, "linear": cv2.INTER_LINEAR, "area": cv2.INTER_AREA}


def resizing_cv(img, scale, method="cubic"):
    """Resize with OpenCV: down with INTER_AREA, up with the chosen method."""
    h, w = img.shape
    s = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    return _u8(cv2.resize(s, (w, h), interpolation=_CV_INTERP[method]))


def median(img, k):
    return median_filter(img, size=k)


ATTACKS = {
    "awgn": lambda img, p: awgn(img, *p),
    "blur": lambda img, p: blur(img, p[0]),
    "sharp": lambda img, p: sharpening(img, p[0], p[1]),
    "jpeg": lambda img, p: jpeg_compression(img, p[0]),
    "resize": lambda img, p: resizing(img, p[0]) if len(p) < 2 or p[1] == "lab" else resizing_cv(img, p[0], p[1]),
    "median": lambda img, p: median(img, p[0]),
}


# ----------------------------------------------------------------------------
# MASKS (computed ONLY from the watermarked image)
# ----------------------------------------------------------------------------
def _block_std(img, b):
    f = img.astype(float)
    m = uniform_filter(f, b)
    return np.sqrt(np.maximum(uniform_filter(f * f, b) - m * m, 0))


def smooth_mask(wm, q=0.5, b=16):
    """The fraction q of the image that is smoothest (lowest local standard deviation).
    In our tests the best attacks against our own scheme were localized on these zones."""
    sd = _block_std(wm, b)
    return sd <= np.quantile(sd, q)


def textured_mask(wm, q=0.5, b=16):
    """The fraction q of the image that is most textured."""
    sd = _block_std(wm, b)
    return sd >= np.quantile(sd, 1 - q)


def edge_mask(wm, q=0.3):
    """The fraction q of the image with the strongest edges (Sobel gradient)."""
    f = wm.astype(float)
    g = gaussian_filter(np.hypot(cv2.Sobel(f, cv2.CV_64F, 1, 0), cv2.Sobel(f, cv2.CV_64F, 0, 1)), 2)
    return g >= np.quantile(g, 1 - q)


def smooth_blocks8_mask(wm, q=0.5):
    """Like smooth_mask but made of whole 8x8 blocks, aligned to the JPEG grid
    (a JPEG localized this way leaves no "half" blocks)."""
    sd8 = _block_std(wm, 8)[4::8, 4::8]
    return np.kron(sd8 <= np.quantile(sd8, q), np.ones((8, 8), bool))


MASKS = {
    "smooth30": lambda wm: smooth_mask(wm, 0.3),
    "smooth50": lambda wm: smooth_mask(wm, 0.5),
    "smooth70": lambda wm: smooth_mask(wm, 0.7),
    "smoothblk50": lambda wm: smooth_blocks8_mask(wm, 0.5),
    "text30": lambda wm: textured_mask(wm, 0.3),
    "text50": lambda wm: textured_mask(wm, 0.5),
    "edges30": lambda wm: edge_mask(wm, 0.3),
}


# ----------------------------------------------------------------------------
# MAIN FUNCTION
# ----------------------------------------------------------------------------
def attacks(input1, attack_name, param_array, mask=None):
    wm = cv2.imread(input1, 0) if isinstance(input1, str) else np.asarray(input1, dtype=np.uint8)
    if isinstance(attack_name, str):
        names, params = [attack_name], [param_array]
    else:
        names, params = list(attack_name), list(param_array)
    if len(names) != len(params):
        raise ValueError("one list of parameters is needed for every attack of the chain")

    out = wm
    for name, p in zip(names, params):
        if name not in ATTACKS:
            raise ValueError(f"attack not allowed: {name!r} (allowed: {', '.join(ATTACKS)})")
        out = ATTACKS[name](out, list(p))

    if mask is not None:
        loc = wm.copy()
        loc[mask] = out[mask]
        out = loc
    return out


def _fmt(v):
    """Number or pair in readable form: 0.8 -> '0.8', (1, 7) -> '1x7'."""
    if isinstance(v, (tuple, list)):
        return "x".join(_fmt(x) for x in v)
    return f"{v:g}" if isinstance(v, float) else str(v)


def _describe_one(name, p):
    if name == "jpeg":
        return f"JPEG QF={p[0]}"
    if name == "awgn":
        return f"AWGN std={_fmt(p[0])}" + (f" (seed {p[1]})" if len(p) > 1 else "")
    if name == "blur":
        return f"Blur sigma={_fmt(p[0])}"
    if name == "sharp":
        return f"Sharpening sigma={_fmt(p[0])} alpha={_fmt(p[1])}"
    if name == "resize":
        return f"Resize scale={_fmt(p[0])} ({p[1] if len(p) > 1 else 'lab'})"
    if name == "median":
        k = p[0]
        return f"Median {_fmt(k)}" if isinstance(k, (tuple, list)) else f"Median {k}x{k}"
    return f"{name} {list(p)}"


def describe(attack_name, param_array, mask_name=None):
    """Text for the challenge log, in the style of the example of the rules ('JPEG QF=90, Median 5x5'),
    with the attacks in the order they are applied, e.g. 'Blur sigma=0.6, JPEG QF=12 (only zone smooth50)'."""
    if isinstance(attack_name, str):
        attack_name, param_array = [attack_name], [param_array]
    s = ", ".join(_describe_one(n, list(p)) for n, p in zip(attack_name, param_array))
    return f"{s} (only zone {mask_name}, computed from the watermarked image)" if mask_name else s


# ----------------------------------------------------------------------------
# AUTOMATIC SEARCH OF THE LIGHTEST ATTACK (attack phase)
# ----------------------------------------------------------------------------
OUR_GROUP = "fakebitinvestigation"
MIN_WPSNR = 35.0
MAX_ERRORS = 3                 # consecutive errors of the other group's detection before skipping the image
SEARCH_SHARE = 0.75            # share of the budget for the search; the rest goes to the greedy

QF = [95, 90, 80, 70, 60, 50, 40, 30, 25, 20, 16, 13, 10, 8, 6, 5, 4, 3, 2, 1]
# family -> (chain of names, strength function -> param_array, grid from the LIGHTEST to the STRONGEST, kind)
# kind: 'int' (QF) / 'float' / 'float-down' (resize: growing strength = decreasing scale) are refined;
# 'none' = only the values of the grid.
# The order is by effectiveness: first the chains with JPEG, which were almost always the winning attacks
# when we attacked our own scheme.
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
    "blur-horiz": (["blur"], lambda p: [[(p / 3, p)]], [0.5, 0.8, 1.2, 1.6, 2.0, 3.0, 4.0, 6.0], "float"),
    "blur-vert": (["blur"], lambda p: [[(p, p / 3)]], [0.5, 0.8, 1.2, 1.6, 2.0, 3.0, 4.0, 6.0], "float"),
    "median-horiz": (["median"], lambda p: [[(1, p)]], [3, 5, 7, 9, 11, 15], "none"),
    "median-vert": (["median"], lambda p: [[(p, 1)]], [3, 5, 7, 9, 11, 15], "none"),
}
# zones in order of effectiveness (against us the best localized attacks were on the smooth zones)
MASK_ORDER = ["smooth50", "smooth70", "smoothblk50", "smooth30", "text50", "text30", "edges30"]
GREEDY_BLOCKS = (64, 32)


# ----------------------------------------------------------------------------
# Fast WPSNR: same formula as the labs (convolve2d 'same' with 'wrap' border = circular
# convolution), computed with the FFT. Used only to discard candidates: the value reported in the
# log is always recomputed with the lab wpsnr().
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
# ORACLE: the detection of the attacked group, with timeout and error handling
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
            raise OracleError(f"{os.path.basename(det_path)} does not define detection()")
        self.det, self.orig, self.wm = mod.detection, orig_path, wm_path
        self.tmp = os.path.join(tmpdir, "att.bmp")
        self.timeout = timeout
        self.calls = self.errors = self.consecutive = 0
        self.last_error = ""
        self.cache = {}                                # same image -> same outcome, no call
        signal.signal(signal.SIGALRM, _on_alarm)

    def broken(self, att):
        """True if the detection says 'absent'. The image is saved as BMP first, like the one we would upload.
        To be safe, an error or a timeout of the detection counts as 'not broken'."""
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
        except Exception as e:                       # noqa: BLE001  (other groups' code can fail in any way)
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
            raise OracleError(f"detection failed {MAX_ERRORS} times in a row, last: {msg}")
        return False


def search(oracle, wm, fam, mask=None, deadline=None):
    """Lightest attack of the family that breaks. Returns (wpsnr, names, params) or None."""
    names, to_params, grid, kind = FAMILIES[fam]
    make = lambda p: attacks(wm, names, to_params(p), mask=mask)      # noqa: E731

    # 1. only the values above 35 dB (the grid goes from light to strong: we stop at the first one below)
    valid = []
    for p in grid:
        if wpsnr_fast(wm, make(p)) < MIN_WPSNR:
            break
        valid.append(p)
    if not valid or not oracle.broken(make(valid[-1])):
        return None
    # 2. binary search of the first value that breaks (assuming: stronger -> easier to break)
    lo_i, hi_i = -1, len(valid) - 1                # valid[hi_i] breaks; valid[lo_i] does not (-1 = none)
    while hi_i - lo_i > 1:
        if deadline and time.time() > deadline:
            break
        mid = (lo_i + hi_i) // 2
        if oracle.broken(make(valid[mid])):
            hi_i = mid
        else:
            lo_i = mid
    best = valid[hi_i]
    # 3. refinement between the last one that does not break and the first one that breaks
    if lo_i >= 0 and kind != "none" and not (deadline and time.time() > deadline):
        lo = valid[lo_i]
        if kind == "int":                          # JPEG: the integer QFs between the two, from the lightest (high QF)
            for p in range(lo - 1, best, -1):
                if oracle.broken(make(p)):
                    best = p
                    break
        else:                                      # 'float' or 'float-down': bisection
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
    """Gives back to the watermarked image the blocks that are not needed (first 64x64, then 32x32).
    Returns (image, keep): keep = mask of the pixels where the attack stayed, so
    the attack is reproducible (base attack, then watermarked image outside keep)."""
    cur = att.copy()
    keep = np.ones(wm.shape, bool)
    for b in GREEDY_BLOCKS:
        diff = np.abs(cur.astype(int) - wm.astype(int))
        blocks = [(i, j) for i in range(0, 512, b) for j in range(0, 512, b) if diff[i:i + b, j:j + b].any()]
        # first the blocks that "cost" the most in WPSNR (the most modified ones)
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


def attack_image(wm_path, orig_path, det_path, out_dir, budget=600, use_greedy=True, timeout=10):
    """Attacks ONE watermarked image of another group (<group>_<imageName>.bmp) and saves in out_dir
    the successful one with the highest WPSNR as fakebitinvestigation_<group>_<imageName>.bmp (+ PNG of the zones
    if the greedy was used). The original (orig_path) is NEVER read here: only the
    detection of the attacked group receives it, as input1. budget = seconds available.
    Returns a dictionary with the outcome; never raises exceptions (errors end up in res['error'])."""
    fname = os.path.basename(wm_path)
    group, image = fname[:-4].split("_", 1)
    t0 = time.time()
    res = {"image": image, "group": group, "wpsnr": None, "desc": None, "found": [], "calls": 0,
           "errors": 0, "error": None, "time": 0.0}
    oracle = None
    try:
        wm = cv2.imread(wm_path, cv2.IMREAD_GRAYSCALE)
        if wm is None or wm.shape != (512, 512):
            raise OracleError(f"image not readable or not 512x512: {fname}")
        with tempfile.TemporaryDirectory() as tmp:
            oracle = Oracle(det_path, orig_path, wm_path, tmp, timeout)
            # check: on the non-attacked watermarked image their detection must say "present"
            if oracle.broken(wm) and oracle.consecutive == 0:
                raise OracleError("their detection already says 'absent' on the watermarked image: we do not attack")
            found = []                                 # (wpsnr, description, image)
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
                    zfile = f"zones_{OUR_GROUP}_{group}_{image}.png"
                    gdesc = f"{desc}; then only inside the white blocks of {zfile} (outside: watermarked image unchanged)"
                    found.append((wpsnr(wm, g), gdesc, g))
                    keep_masks[gdesc] = (zfile, keep)

            # among the successful ones, the highest WPSNR, checked again with the detection
            found.sort(key=lambda x: -x[0])
            res["found"] = [(round(w, 2), d) for w, d, _ in found]
            for w, desc, att in found:
                if w >= MIN_WPSNR and oracle.broken(att):
                    cv2.imwrite(os.path.join(out_dir, f"{OUR_GROUP}_{group}_{image}.bmp"), att)
                    if desc in keep_masks:                  # zone mask, so that the attack can be reproduced
                        zfile, keep = keep_masks[desc]
                        cv2.imwrite(os.path.join(out_dir, zfile), keep.astype(np.uint8) * 255)
                    res["wpsnr"], res["desc"] = w, desc
                    break
    except OracleError as e:
        res["error"] = str(e)
    except Exception:                                  # noqa: BLE001  (one bad image must not stop the whole run)
        res["error"] = traceback.format_exc(limit=2).strip().splitlines()[-1]
    if oracle is not None:
        res["calls"], res["errors"] = oracle.calls, oracle.errors
    res["time"] = time.time() - t0
    return res


# ----------------------------------------------------------------------------
# COMMAND LINE
# ----------------------------------------------------------------------------
def _num(x):
    return int(x) if x.lstrip("-").isdigit() else float(x)


def main():
    ap = argparse.ArgumentParser(description="Attacks of group fakebitinvestigation")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("apply", help="applies an attack (or a chain) and saves the image")
    a1.add_argument("image")
    a1.add_argument("out")
    a1.add_argument("attack", help="name or comma-separated chain, e.g. blur,jpeg")
    a1.add_argument("params", nargs="+", help="parameters in order; for 'resize' the method is optional")
    a2 = sub.add_parser("search", help="searches the lightest attack that breaks ONE image")
    a2.add_argument("--image", required=True, help="watermarked image of the other group: <group>_<imageName>.bmp")
    a2.add_argument("--original", required=True, help="original: goes ONLY to their detection, as input1")
    a2.add_argument("--detection", required=True, help="detection_<group>.py of the attacked group")
    a2.add_argument("--out", required=True)
    a2.add_argument("--budget", type=float, default=600, help="seconds available (default 600)")
    a2.add_argument("--timeout", type=float, default=10, help="maximum seconds for one call to the detection")
    a2.add_argument("--no-greedy", action="store_true")
    a = ap.parse_args()

    if a.cmd == "apply":
        names = a.attack.split(",")
        n_par = {"awgn": 1, "blur": 1, "sharp": 2, "jpeg": 1, "resize": 1, "median": 1}
        vals, params = list(a.params), []
        for n in names:
            if n not in n_par:
                sys.exit(f"attack not allowed: {n!r} (allowed: {', '.join(n_par)})")
            p = [_num(vals.pop(0)) for _ in range(n_par[n])]
            if n == "resize" and vals and not vals[0].lstrip("-").replace(".", "", 1).isdigit():
                p.append(vals.pop(0))
            params.append(p)
        if vals:
            sys.exit(f"extra parameters: {vals}")
        wm = cv2.imread(a.image, cv2.IMREAD_GRAYSCALE)
        out = attacks(wm, names, params)
        cv2.imwrite(a.out, out)
        print(f"{describe(names, params)} -> {a.out}, WPSNR {wpsnr(wm, out):.2f} dB")
        return

    os.makedirs(a.out, exist_ok=True)
    r = attack_image(a.image, a.original, a.detection, a.out, a.budget, not a.no_greedy, a.timeout)
    info = f"({r['calls']} calls to the detection, {r['errors']} errors, {r['time']:.0f} s)"
    if r["error"]:
        print(f"ERROR: {r['error']} {info}")
    elif r["wpsnr"] is None:
        print(f"not broken {info}")
    else:
        print(f"BROKEN at {r['wpsnr']:.2f} dB with {r['desc']} {info}")
        saved = os.path.join(a.out, f"{OUR_GROUP}_{r['group']}_{r['image']}.bmp")
        print(f"saved in {saved}")


if __name__ == "__main__":
    main()
