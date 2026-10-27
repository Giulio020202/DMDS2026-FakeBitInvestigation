"""
Capture the Mark 2026 - embedding, group fakebitinvestigation

Our scheme is spread spectrum on pairs of coefficients: each bit of the watermark (mapped to -1/+1)
increases the magnitude of one coefficient and decreases the magnitude of its neighbour by the same
amount. The same watermark is embedded twice:
  1) in the global DCT of the image, inside the radial band R_MIN..R_MAX;
  2) in a db4 wavelet decomposition (DWT_LEVELS levels), in the detail subbands of the levels in
     DWT_USE. Levels 3 and 4 cover about the same frequencies as the DCT part (0.03-0.125 cycles/pixel),
     but every wavelet coefficient only covers a small area of the image.
The two parts have relative amplitudes K_DCT and K_DWT. A single scale factor is then found by
bisection, so that the WPSNR (computed with the lab code) ends up just above TARGET_WPSNR.

embedding(input1, input2) -> output1
  input1  = path of the original image (512x512 grayscale BMP)
  input2  = path of the watermark (.npy, 1024 bits 0/1)
  output1 = watermarked image (numpy uint8 512x512)
Images are read with OpenCV; the function prints nothing, opens no windows and writes no files.
Everything we may fine-tune after the 27/10 deadline (parameters and embedding locations) is in
the constants below.
"""
import os

import cv2
import numpy as np
import pywt
from scipy.fft import dctn, idctn
from scipy.signal import convolve2d

# ----------------------------------------------------------------------------
# PARAMETERS (the ones describing where the watermark goes must be the same in the detection)
# ----------------------------------------------------------------------------
TARGET_WPSNR = 62.05   # dB: just above 62, the lower limit of the 5-point quality band
TARGET_BY_IMAGE = {}   # optional per-image target, e.g. {"buildings": 58.05} (file name without extension -> dB).
                       # Empty = TARGET_WPSNR for every image. We may fill it on the challenge day, so that
                       # this code gives exactly the images we upload.
R_MIN = 32.0           # radial band of the DCT pairs
R_MAX = 128.0
N_BITS = 1024          # length of the assigned watermark
WAVELET = "db4"        # wavelet of the second part
DWT_LEVELS = 4         # levels of the decomposition
DWT_USE = (3, 4)       # levels whose detail subbands host the watermark
K_DCT = 1.0            # relative amplitude of the DCT part
K_DWT = 1.0            # relative amplitude of the wavelet part
AMP_MIN, AMP_MAX = 0.2, 40.0   # search interval of the scale
N_ITER = 14                    # bisection steps


# ----------------------------------------------------------------------------
# WPSNR: identical copy of the code provided in the labs
# ----------------------------------------------------------------------------
# Contrast Sensitivity Function (CSF) of the lab WPSNR, evaluated at the spatial frequencies (u, v).
#
# @param u  horizontal frequencies (array)
# @param v  vertical frequencies (array, same shape as u)
# @return   sensitivity of the eye at each frequency (array, same shape as u)
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


# Samples the CSF on the integer frequency grid -20..20 x -20..20.
#
# @return  41x41 matrix of the CSF in the frequency domain
def _csfmat():
    min_f, max_f, step_f = -20, 20, 1
    freq_range = np.arange(min_f, max_f + step_f, step_f)
    u, v = np.meshgrid(freq_range, freq_range, indexing='xy')
    return _csffun(u, v)


# Turns the CSF matrix into the spatial filter used by the WPSNR (inverse FFT, real part).
#
# @return  41x41 spatial filter (float array)
def _get_csf_filter():
    Fmat = _csfmat()
    filter_coeffs = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(Fmat)))
    return np.real(filter_coeffs)


# WPSNR of the labs: PSNR of the difference between the two images, filtered with the CSF.
#
# @param image_a  first image (integers in 0..255, or floats in 0..1)
# @param image_b  second image, same shape as image_a
# @return         WPSNR in dB (9999999.0 if the two images are identical)
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


# ----------------------------------------------------------------------------
# PART 1: global DCT
# ----------------------------------------------------------------------------
# Global 2D DCT of the whole image (orthonormal).
#
# @param x  image as a float array
# @return   DCT coefficients, same shape as x
def _dct2(x):
    return dctn(x, norm='ortho')


# Inverse of _dct2: from the DCT coefficients back to the pixels.
#
# @param c  DCT coefficients (float array)
# @return   image in the pixel domain (float array, not rounded)
def _idct2(c):
    return idctn(c, norm='ortho')


# Chooses where the DCT part of the watermark goes: the N_BITS pairs (u,v),(u,v+1), with v even,
# in the radial band R_MIN <= r <= R_MAX that have the largest min(|C1|, |C2|), sorted by position.
#
# @param coeffs  DCT coefficients of the original image
# @return        (uu, vv): row and column of the first coefficient of each pair (arrays of N_BITS ints)
def _select_pairs(coeffs):
    H, W = coeffs.shape
    u = np.arange(H)[:, None]
    v = np.arange(0, W - 1, 2)[None, :]
    uu = np.broadcast_to(u, (H, v.shape[1])).ravel()
    vv = np.broadcast_to(v, (H, v.shape[1])).ravel()
    r = np.sqrt(uu**2 + (vv + 0.5)**2)
    band = (r >= R_MIN) & (r <= R_MAX)
    uu, vv = uu[band], vv[band]
    strength = np.minimum(np.abs(coeffs[uu, vv]), np.abs(coeffs[uu, vv + 1]))
    best = np.argsort(-strength, kind='stable')[:N_BITS]
    best = np.sort(best)
    return uu[best], vv[best]


# ----------------------------------------------------------------------------
# PART 2: db4 wavelet (same differential pairs, inside a subband)
# ----------------------------------------------------------------------------
# Wavelet decomposition (WAVELET, DWT_LEVELS levels, orthogonal with 'periodization', so Parseval
# holds) with all the coefficients put in a single 512x512 array.
#
# @param x  image as a float array
# @return   (arr, slices): array of the coefficients and the slices of each subband inside it
def _dwt2(x):
    arr, slices = pywt.coeffs_to_array(pywt.wavedec2(x, WAVELET, mode='periodization', level=DWT_LEVELS))
    return arr, slices


# Inverse of _dwt2: from the array of wavelet coefficients back to the pixels.
#
# @param arr     array of the coefficients (same layout as the output of _dwt2)
# @param slices  slices of the subbands, as returned by _dwt2
# @return        image in the pixel domain (float array, not rounded)
def _idwt2(arr, slices):
    return pywt.waverec2(pywt.array_to_coeffs(arr, slices, output_format='wavedec2'), WAVELET, mode='periodization')


# Chooses where the wavelet part of the watermark goes: the N_BITS horizontal pairs (i,j),(i,j+1),
# with j even, inside the detail subbands of the levels in DWT_USE, that have the largest
# min(|C1|, |C2|), sorted by position.
#
# @param arr     wavelet coefficients of the original image (output of _dwt2)
# @param slices  slices of the subbands (output of _dwt2)
# @return        (ii, jj): row and column of the first coefficient of each pair (arrays of N_BITS ints)
def _select_pairs_dwt(arr, slices):
    m = np.zeros(arr.shape, bool)
    for lev in DWT_USE:
        k = DWT_LEVELS - lev + 1           # in wavedec2 index 1 is the coarsest level
        for key in ('ad', 'da', 'dd'):
            m[slices[k][key]] = True
    ii, jj = np.nonzero(m[:, :-1] & m[:, 1:])
    keep = jj % 2 == 0
    ii, jj = ii[keep], jj[keep]
    strength = np.minimum(np.abs(arr[ii, jj]), np.abs(arr[ii, jj + 1]))
    best = np.sort(np.argsort(-strength, kind='stable')[:N_BITS])
    return ii[best], jj[best]


# Builds the watermark pattern in the coefficient domain, with amplitude 1: +sgn(C1)*w on the first
# coefficient of each pair (its magnitude grows by w) and -sgn(C2)*w on the second one (its
# magnitude shrinks by w). All the other coefficients are 0.
#
# @param c   coefficients of the original image (DCT or wavelet array)
# @param ii  rows of the first coefficient of each pair
# @param jj  columns of the first coefficient of each pair (the second one is at jj + 1)
# @param w   watermark in -1/+1 form, one value per pair
# @return    array with the same shape as c that contains only the pushes
def _pair_pattern(c, ii, jj, w):
    s1 = np.where(c[ii, jj] < 0, -1.0, 1.0)
    s2 = np.where(c[ii, jj + 1] < 0, -1.0, 1.0)
    P = np.zeros_like(c)
    P[ii, jj] = s1 * w
    P[ii, jj + 1] = -s2 * w
    return P


# ----------------------------------------------------------------------------
# EMBEDDING (signature required by the rules)
# ----------------------------------------------------------------------------
# Embeds the watermark in the image: DCT part + wavelet part, with the single scale found by
# bisection so that the WPSNR stays just above the target (TARGET_WPSNR, or TARGET_BY_IMAGE).
#
# @param input1  path of the original image (512x512 grayscale BMP)
# @param input2  path of the watermark (.npy, N_BITS bits 0/1)
# @return        watermarked image (numpy uint8 512x512)
def embedding(input1, input2):
    original = cv2.imread(input1, cv2.IMREAD_GRAYSCALE)
    bits = np.load(input2).astype(np.float64).ravel()[:N_BITS]
    w = 2.0 * bits - 1.0                          # 0/1 -> -1/+1
    o = original.astype(np.float64)
    target = TARGET_BY_IMAGE.get(os.path.splitext(os.path.basename(input1))[0], TARGET_WPSNR)

    c = _dct2(o)
    uu, vv = _select_pairs(c)
    arr, slices = _dwt2(o)
    ii, jj = _select_pairs_dwt(arr, slices)
    # watermark pattern in the pixels (sum of the two parts, both linear)
    pattern = K_DCT * _idct2(_pair_pattern(c, uu, vv, w)) + K_DWT * _idwt2(_pair_pattern(arr, ii, jj, w), slices)

    # Watermarked image for a given scale of the pattern.
    #
    # @param s  scale factor of the pattern
    # @return   original + s * pattern, rounded and clipped to uint8
    def marked(s):
        return np.clip(np.rint(o + s * pattern), 0, 255).astype(np.uint8)

    # Scale calibration: the largest one that keeps WPSNR >= TARGET_WPSNR (logarithmic bisection).
    lo, hi = AMP_MIN, AMP_MAX
    best = marked(lo)
    for _ in range(N_ITER):
        mid = np.sqrt(lo * hi)
        candidate = marked(mid)
        if wpsnr(original, candidate) >= target:
            lo, best = mid, candidate
        else:
            hi = mid

    output1 = best
    return output1
