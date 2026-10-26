# Capture the Mark 2026 — group `fakebitinvestigation`

Submitted version: **2.0 hybrid (DCT + wavelet)**. The full explanation of the algorithm is in `../documentation/project_explanation.md` (PDF: `../documentation/project_explanation.pdf`).

## Files

| File | What it does | Signature |
|---|---|---|
| `embedding_fakebitinvestigation.py` | embeds the watermark | `embedding(input1, input2) -> output1` |
| `detection_fakebitinvestigation.py` | tells whether the watermark is still there | `detection(input1, input2, input3) -> output1, output2` |
| `attacks_fakebitinvestigation.py` | **all** our attack code (allowed attacks only): one attack, a chain or a localized attack; the automatic search of the lightest attack that breaks an image | `attacks(input1, attack_name, param_array, mask=None) -> output1`, `attack_image(...)`, command line |
| `roc_fakebitinvestigation.py` | ROC and threshold τ of the detection | command line |
| `demo_fakebitinvestigation.py` | shows how to use all the files above | command line |
| `roc_fakebitinvestigation.png` | ROC curve of the full run | — |
| `roc_fakebitinvestigation_output.txt` | numbers printed by the full run of the ROC | — |

This folder also contains `fakebitinvestigation.npy` (our watermark, used by the embedding as `input2`; the detection does not read it) and `demo_images/` (10 BMP 512×512 grayscale images used by the demo), so **the folder works on its own**. The full ROC needs the 101 sample images of the course: by default the script looks for them in `../sample-images/` (otherwise it uses `demo_images/`), or they can be given with `--images`.

## Requirements

Python 3.10 with `numpy`, `scipy`, `opencv-python`, `PyWavelets`.
Attacks, ROC and demo also need `pillow`, `scikit-image`, `scikit-learn`, `matplotlib`.
All of them are in the requirements of lab 1 and lab 2.

```bash
pip install numpy scipy opencv-python PyWavelets pillow scikit-image scikit-learn matplotlib
```

All the commands below are run **from this folder (`submission/`)**.

## Demo (the best place to start)

```bash
python demo_fakebitinvestigation.py                       # embedding + detection + 11 example attacks (~15 s)
python demo_fakebitinvestigation.py --attacks             # + automatic attack search (attack_image) against our detection (~5 min)
python demo_fakebitinvestigation.py --roc                 # + quick ROC (only to see that it runs)
python demo_fakebitinvestigation.py --image demo_images/0004.bmp --out /tmp/demo
```

For each case it prints `output1`, `output2` (WPSNR), the detection time and the outcome. All the images it produces go to `demo_output/`.

## Embedding

```python
from embedding_fakebitinvestigation import embedding
wm = embedding("demo_images/0000.bmp", "fakebitinvestigation.npy")   # numpy uint8 512x512
```

- `input1` = path of the original image; `input2` = path of the `.npy` watermark (1024 bits).
- It does not print, open windows or write files. The WPSNR of the watermarked image is calibrated to `TARGET_WPSNR` (62.05 dB).

## Detection

```python
from detection_fakebitinvestigation import detection
output1, output2 = detection("original.bmp", "watermarked.bmp", "attacked.bmp")
```

- `output1` = 1 if the watermark is present, 0 if not; `output2` = WPSNR(watermarked, attacked) with the lab code.
- Single file: no relative paths, no external scripts, the watermark is never read; the only "external" function is the lab WPSNR, copied into the file.
- Decision: present if similarity ≥ `TAU` (constant at the top of the file, obtained from the ROC).
- Measured time: about 0.6 s per call (limit 5 s).

## Attacks (one single file: `attacks_fakebitinvestigation.py`)

All our attack code is in this file, which works on its own: it only uses libraries (numpy, scipy, opencv, pillow, scikit-image) and the lab WPSNR, copied inside. It contains three things.

**1. `attacks()`: one attack, a chain or a localized attack** (example structure of the rules, p.6).

```python
from attacks_fakebitinvestigation import attacks, smooth_mask
att = attacks("groupX_lena.bmp", "jpeg", [20])                          # single
att = attacks("groupX_lena.bmp", ["blur", "jpeg"], [[0.6], [12]])       # chain
att = attacks(wm, "awgn", [30], mask=smooth_mask(wm, 0.5))              # localized
```

| `attack_name` | `param_array` | Meaning |
|---|---|---|
| `'awgn'` | `[std]` or `[std, seed]` | additive gaussian noise |
| `'blur'` | `[sigma]` (or `[(sy, sx)]`) | gaussian filter (also anisotropic) |
| `'sharp'` | `[sigma, alpha]` | unsharp masking |
| `'jpeg'` | `[QF]` | JPEG compression, QF 1–100 |
| `'resize'` | `[scale]` or `[scale, method]` | down and back up to 512×512; method `'lab'`, `'cubic'`, `'linear'`, `'area'` |
| `'median'` | `[k]` (or `[(ky, kx)]`) | median filter |

- Only the allowed attacks (p.6): AWGN, blur, sharpening, JPEG, resize, median, also combined or localized.
- The zones of the localized attacks (`MASKS`: smooth, textured and edge zones) are computed **only from the watermarked image**, never from the original.

From the command line (saves the attacked image and prints the WPSNR):

```bash
python attacks_fakebitinvestigation.py apply groupX_lena.bmp out.bmp jpeg 20
python attacks_fakebitinvestigation.py apply groupX_lena.bmp out.bmp blur,jpeg 0.6 12
```

**2. `attack_image()`: the automatic search of the lightest attack that breaks an image** (this is what we use in the challenge).

```bash
python attacks_fakebitinvestigation.py search --image groupX_lena.bmp --original lena.bmp \
       --detection detection_groupX.py --out OUTDIR [--budget SECONDS] [--timeout SECONDS] [--no-greedy]
```

```python
from attacks_fakebitinvestigation import attack_image
r = attack_image("groupX_lena.bmp", "lena.bmp", "detection_groupX.py", "OUTDIR", budget=600)
# r["wpsnr"], r["desc"]: the successful attack with the highest WPSNR (None = not broken); r["error"] if something goes wrong
```

For the given image it:
1. tries 25 attack families (single attacks and chains, from the lightest to the strongest), both on the whole image and localized on 7 zones;
2. uses a fast WPSNR (FFT, same result as the lab code) to discard, without calling the detection, everything that goes below 35 dB;
3. for each family, finds with a binary search the lightest parameter that breaks the image, using the outcome of the attacked group's detection as the check (the TAs confirmed this is allowed);
4. gives back to the watermarked image the 64×64 and 32×32 blocks that are not needed to break it, to raise the WPSNR, again checking with the attacked group's detection (the TAs confirmed this is allowed too). The zone mask is saved as a PNG and mentioned in the attack description. It can be skipped with `--no-greedy`;
5. keeps the successful attack with the highest WPSNR (p.7: only one is uploaded per image) and saves it as `fakebitinvestigation_<group>_<imageName>.bmp`.

**The original image is never read or used to build the attack**: its path is only passed to the attacked group's detection, as its `input1` (rules p.6; the TAs confirmed that the rule is about using the original, for example `watermarked − original`, not about calling their detection).

Protections against the other groups' detections: a timeout for each call (`--timeout`, 10 s); errors are caught (after 3 in a row the image is skipped and the error is returned); prints are suppressed and no windows are opened.

**3. The description for the log** (`describe()`), in the style of the example in the rules: `Blur sigma=0.6, JPEG QF=12`.

On the challenge day we use a small helper to attack all the images of all the groups in parallel and write the Excel log. It is not part of the submission and contains no attacks: `../challenge/search_attacks_fakebitinvestigation.py`, which calls `attack_image()` for every image.

## ROC and threshold τ

```bash
python roc_fakebitinvestigation.py               # full run: 101 images x 40 attacks (~40 min on 14 cores)
python roc_fakebitinvestigation.py --quick       # quick test (10 images x 5 attacks): the τ it gives is NOT the real one
python roc_fakebitinvestigation.py --images DIR --watermark FILE.npy --attacks N --out curve.png
```

Procedure of p.4 of the rules plus the checks of Lab 4:
- **H1**: the same watermarked image, attacked;
- **H0-R**: random watermarks, the procedure of the rules;
- **H0-A**: the same attack applied to the original;
- **H0-B**: the same attack on an image watermarked by "another group".

The images are split into calibration (70%) and test (30%, never used to choose τ). τ is the lowest threshold with FPR ≤ 0.1% on each kind of H0.

**Result of the full run:** thresholds for FPR ≤ 0.1%:
- H0-R 0.0835;
- H0-A 0.0841;
- H0-B 0.0867.

The maximum is 0.0867, rounded up to **τ = 0.087** (`TAU` in the detection).
- Calibration: TPR 99.1%.
- Test, with τ frozen: FPR 0.058% / 0% / 0%, TPR 97.6% (99.6% on the attacks with WPSNR ≥ 35).

The full output is in `roc_fakebitinvestigation_output.txt`, the curve in `roc_fakebitinvestigation.png`.

## Fine-tuning

After 27/10 only parameters and embedding locations can change (the constants at the top of `embedding_…` and `detection_…`). If we change one of them, we:

1. change it **in both** `embedding_fakebitinvestigation.py` and `detection_fakebitinvestigation.py`: they stay identical, except `TARGET_WPSNR` and `TARGET_BY_IMAGE`, which are only in the embedding;
2. run the full ROC again (`python roc_fakebitinvestigation.py`) and copy the printed τ into `TAU` in the detection;
3. run `python demo_fakebitinvestigation.py` again: watermarked → 1, original → 0, times under 5 s.
