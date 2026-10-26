# The challenge day (2/11): what to do, step by step

Operational guide of group `fakebitinvestigation` for the defense (8:30–10:00) and the attack (10:15–12:30), always within the rules (`../documentation/document.pdf`). All the commands are run **from the folder `fakebitinvestigation/submission/`** (it holds the only copy of the code: embedding, detection, attacks; this file and `defense_fakebitinvestigation.py` are in `../challenge/`) with the environment active:

```bash
cd fakebitinvestigation/submission
source ../../OFF_MDS_ENV/bin/activate
```

> 🗣 **In plain words**:
> - **in defense** we try to attack ourselves to decide how "strong" to make the watermark on each image, then we upload;
> - **in attack** we let the helper `../challenge/search_attacks_fakebitinvestigation.py` work: for every image of the other groups it calls the automatic search of our attack file (`attack_image()` in `attacks_fakebitinvestigation.py`) and writes the log, and we upload as we go.

---

## 0. Before 2/11 (at home)

1. **Only one PC accesses the website; at most 3 laptops; no servers** (rules). The "computing" PC must have many cores: the script uses `cores − 1` processes.
2. On the computing PC: updated repo (`git pull`), working `OFF_MDS_ENV` environment (Python 3.10 with the requirements of lab 1 and lab 2), and a full test:
   ```bash
   python demo_fakebitinvestigation.py --attacks      # ~5 min: it must finish without errors
   ```
3. **Libraries that the other groups' detections might use**, to be installed beforehand (there is no time during the challenge):
   ```bash
   pip install numpy scipy opencv-python PyWavelets scikit-image scikit-learn matplotlib pillow
   ```
4. Prepare the empty folders:
   ```bash
   mkdir -p ~/ctm/defense/originals ~/ctm/defense/out \
            ~/ctm/attack/img ~/ctm/attack/orig ~/ctm/attack/det ~/ctm/attack/out
   ```
5. **Roles** (3 people):
   - **A** = computing PC, runs the scripts;
   - **B** = website, that is download and upload;
   - **C** = checks file names, rules and times, and keeps this guide open.

---

## 1. DEFENSE PHASE (8:30–10:00, 90 minutes)

Goal: upload the 3 watermarked images and the detection. The WPSNR target can be chosen **image by image**, because the quality points are computed on the **mean** of the 3 (answer of the TAs).

| when | who | what |
|---|---|---|
| 8:30 | B | downloads the 3 original images into `~/ctm/defense/originals/` (original names, e.g. `buildings.bmp`) |
| 8:32 | A | **safety version**: `python ../challenge/defense_fakebitinvestigation.py --images ~/ctm/defense/originals --out ~/ctm/defense/safety --final` (all at 62.05, ~30 s) |
| 8:35 | B | immediately uploads the 3 images and the encrypted detection from `~/ctm/defense/safety/upload/`. Images can be uploaded several times during the defense (rules), so we are covered even if something goes wrong later |
| 8:35 | A | **step 1.1** (trial of the targets) |
| ~8:55 | everyone | **step 1.2** (choice) |
| ~9:00 | A | **step 1.3** (final files) |
| by 9:30 | B | **step 1.4** (final upload, replaces the safety version), with half an hour of margin |

### 1.1 Trial of the targets (~17 min)

```bash
python ../challenge/defense_fakebitinvestigation.py --images ~/ctm/defense/originals --out ~/ctm/defense/out
```

For every image and every target (58.05 and 62.05 dB) it:
1. does the embedding;
2. checks that watermarked → 1, original → 0 and that the detection stays under 5 s;
3. **attacks us** with the same engine we use in the attack phase (`attack_image()` of `attacks_fakebitinvestigation.py`), at most 15 minutes per trial, 6 trials in parallel.

At the end it prints the ranking of the 8 combinations of targets with the estimated defense points, and the line to copy, for example:

```
TARGET_BY_IMAGE = {'buildings': 58.05, 'rollercoaster': 62.05, 'tree': 62.05}
```

If a line says `!! CHECK !!` (watermarked image not detected, original detected, slow detection): **stop and look**, do not upload anything.

### 1.2 The choice (5 minutes, all together)

The simulation uses **one** opponent only, however strong; in the challenge **9 groups** attack us. Practical rules:

- **The +2 bonus (never broken) is fragile.** It is enough that one of the 9 groups breaks us once. If a combination wins only thanks to the bonus, ask whether it would hold against 9 opponents.
- **Against a strong attacker v2.0 is often broken at 62 dB** (70% of the images in our tests), but almost always at 35–38 dB, so with 6 robustness points out of 6. Going down to 58 dB is mostly useful to aim at the bonus.
- From our simulations (§A8.3 of `../documentation/project_explanation.md`): 11.10 expected points at 58.05 against 10.88 at 62.05. It is a small difference.
- **Do not go below a mean of 58.05**, otherwise another quality point is lost.

If time is short: **all at 62.05** (default of the code, nothing to change).

### 1.3 Final files (2 minutes)

1. Open `embedding_fakebitinvestigation.py` and paste the chosen line in place of `TARGET_BY_IMAGE = {}`. It is a **parameter**, so it is allowed as fine-tuning. **The detection is NOT touched**: τ does not depend on the target.
2. Generate the files to upload **with the code as it is**, so that code and uploaded images match:
   ```bash
   python ../challenge/defense_fakebitinvestigation.py --images ~/ctm/defense/originals --out ~/ctm/defense/out --final
   ```
   It must print `OK` on all 3 and the mean WPSNR. The files are in `~/ctm/defense/out/upload/`: the images, already named `fakebitinvestigation_<imageName>.bmp`, and the encrypted detection `detection_fakebitinvestigation.pyc`, compiled from the current `detection_fakebitinvestigation.py`.
3. Save the state in the repo, so that we keep track of what we uploaded:
   ```bash
   git add -A && git commit -m "Challenge: targets chosen for the 3 images" && git push
   ```

### 1.4 Upload (B)

- The 3 images of `~/ctm/defense/out/upload/`.
- The **encrypted detection** `detection_fakebitinvestigation.pyc` of `~/ctm/defense/out/upload/` (the professors asked for it encrypted with `py_compile`, as shown in lab 1). A copy compiled from the submitted code is also in `../challenge/detection_fakebitinvestigation.pyc`. If we changed a parameter of the detection, the `.pyc` must be the one produced by `--final` (or compiled again with `python -c "import py_compile; py_compile.compile('detection_fakebitinvestigation.py', cfile='../challenge/detection_fakebitinvestigation.pyc', dfile='detection_fakebitinvestigation.py')"`), always with Python 3.10.
- **After the upload no uploaded file is changed anymore**: 2-point penalty.

---

## 2. ATTACK PHASE (10:15–12:30, 135 minutes)

Goal: break as many images as possible, of **all** the groups, with the highest possible WPSNR. How points are made in attack:

- **activity**, that is the % of groups successfully attacked: >90% → 6 points. With 9 opposing groups, 8 out of 9 is 89%, so **we need at least one broken image for EVERY group**;
- **quality**: number of broken images with WPSNR above the mean of all the attacks (>15 → 4 points);
- **+2 bonus** if we break a group that nobody else breaks.

### 2.1 Preparation (10:15–10:25)

B downloads from the website, A puts the files in the folders **with these names** (the script recognizes them from the name):

| folder | content | name |
|---|---|---|
| `~/ctm/attack/img/` | the watermarked images of the other groups (27 = 9 groups × 3) | `<group>_<imageName>.bmp` (the website format) |
| `~/ctm/attack/orig/` | the originals | `<imageName>.bmp` |
| `~/ctm/attack/det/` | the detections of the others | `detection_<group>.py` or `detection_<group>.pyc` (rename if needed) |

The originals are needed **only** by the other groups' detections, which want them as `input1`. We never use them to choose or tune an attack: the script just passes them on. Using the outcome of the other groups' detection to guide the attack is **allowed by the TAs**. The encrypted `.pyc` detections only work with the same Python version they were compiled with (3.10, as required).

### 2.2 Check of the detections (1 minute)

```bash
python ../challenge/search_attacks_fakebitinvestigation.py --images ~/ctm/attack/img --originals ~/ctm/attack/orig \
       --detections ~/ctm/attack/det --out ~/ctm/attack/out --check
```

For every image it prints `ok, X s per call`, or:
- `ERROR: ModuleNotFoundError: …`: a library is missing → `pip install <library>` and repeat the check;
- `ERROR: …` (other): that detection is broken. The script will skip it; take note and, if there is time, look at it by hand (a `bad magic number` error means a `.pyc` compiled with another Python version);
- `WARNING: it says 'absent' on the watermarked image`: wrong file name (original and watermarked image do not match?) or faulty detection. Check the names.

### 2.3 Attack (10:30 → at most ~12:10)

```bash
python ../challenge/search_attacks_fakebitinvestigation.py --images ~/ctm/attack/img --originals ~/ctm/attack/orig \
       --detections ~/ctm/attack/det --out ~/ctm/attack/out --total-min 100
```

- **Measured time** (simulation of 19/10: 27 images, 9 groups × 3, 13 processes, detection of about 0.6 s, with the zones guided by the detection, as by default): **32 minutes** in total, with 18 broken images out of 27, at least one for each of the 9 groups, at 35.6–38.0 dB. The images that do not break finish earlier, in about 7 minutes, because they run out of tries. With slower detections the time grows, but the ceiling stays `--total-min`.
- `--total-min 100`: the script gives itself 100 minutes in total and computes by itself the time per image. **Starting at 10:30 it finishes by 12:10**, with 20 minutes of margin for the last uploads. If we start later, lower it, for example `--total-min 80` starting at 10:50.
- It prints one line per image as it finishes:
  ```
  [5/27 14 min] [BROKEN] groupX_buildings: 37.12 dB with Median 3x3, JPEG QF=7 (480 calls, 0 errors, 320 s)
  ```
- **Upload as we go (B)**: every `[BROKEN]` already has the image ready in `~/ctm/attack/out/` with the right name, `fakebitinvestigation_<group>_<imageName>.bmp`. The script prints only one line per image, and it is already the best attack found, so we upload **only one image per attacked image** as the rules require.
- **If the script stops** (crash, PC turned off): run the same command again with `--resume` and a smaller `--total-min`. It restarts skipping the images already done.

### 2.4 Second pass (if there is time left)

At the end the script prints `broken X/27` and the list of the images with errors. Then:

- **A group without any broken image** is worth activity points (and maybe the +2 bonus): run again only on that group, with more time and on a different output folder, so that the main log is not overwritten:
  ```bash
  python ../challenge/search_attacks_fakebitinvestigation.py ... --out ~/ctm/attack/out_groupX --only groupX --budget 900
  ```
- The alternatives for every image (all the successful attacks, not only the best one) are in `~/ctm/attack/out/all_successful.csv`.

### 2.5 After the challenge: the log

The rules ask for an **Excel log** with columns Image | Group | WPSNR | Attack(s) with parameters. It is already ready in `~/ctm/attack/out/attack_log.csv`: open it with Excel or LibreOffice and save it as `.xlsx`. The descriptions are already in the style of the example in the rules (`JPEG QF=6, Blur sigma=0.8`). When an attack was restricted to zones chosen with the detection, the description mentions a file `zones_fakebitinvestigation_<group>_<image>.png`, that is the mask of the attacked zones. **It must be sent together with the log**, because without it the attack cannot be reproduced. If there was a second pass, merge the rows of the various `attack_log.csv`, keeping for every image **only** the uploaded one.

---

## 3. What NOT to do (rules)

| forbidden | why / penalty |
|---|---|
| changing an uploaded file after the defense phase | −2 points |
| changing in the embedding anything that is not a parameter or a location (after 27/10) | only fine-tuning is allowed |
| changing a "location" parameter (R_MIN, R_MAX, DWT_USE, …) only in the embedding | it must be changed **in both** codes before the upload (and the `.pyc` compiled again); better not to touch them during the challenge |
| using the original to choose, tune or localize an attack | the script only passes it to the other groups' detection; the zones come from the watermarked image only |
| attacks other than AWGN, blur, sharpening, JPEG, resize, median (also combined) | only these are allowed; the script only uses these |
| external software (Photoshop, GIMP, …) | forbidden |
| uploading several attacks of the same image | only one is uploaded, the best one |
| servers or more than 3 laptops, more than one PC on the website | forbidden |

## 4. Common problems

| problem | solution |
|---|---|
| `!! CHECK !!` in defense | do not upload; check that `TARGET_BY_IMAGE` is written correctly (image names **without** `.bmp`) and redo `--final` |
| the check gives `ModuleNotFoundError` | `pip install` the library, then repeat the check |
| the check gives `bad magic number` | that `.pyc` was compiled with another Python version: ask the group for one compiled with Python 3.10 |
| a detection of another group is very slow | the timeout (10 s) stops it: after 3 timeouts in a row the image is skipped. We can try `--only group --timeout 20` at the end |
| the script stopped | run it again with `--resume` |
| we are late | reduce `--total-min` |
| we are very late | add `--no-greedy`: faster, slightly lower WPSNR (about −0.2/−0.4 dB) |
