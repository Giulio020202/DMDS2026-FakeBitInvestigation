# DMDS2026 — FakeBitInvestigation

Repo of group `fakebitinvestigation` for the **Capture the Mark 2026** challenge (Multimedia Data Security, University of Trento).

The submitted version is **2.0 hybrid (DCT + wavelet)**.

- [`fakebitinvestigation/submission/`](fakebitinvestigation/submission/): **the files to submit**, 100% of the work (embedding, detection, ROC, attacks in a single file, demo) with the [`README.md`](fakebitinvestigation/submission/README.md) to run them;
- [`fakebitinvestigation/challenge/`](fakebitinvestigation/challenge/): **the challenge day** (not submitted): [`CHALLENGE_INSTRUCTIONS.md`](fakebitinvestigation/challenge/CHALLENGE_INSTRUCTIONS.md) (step by step), the encrypted detection to upload (`detection_fakebitinvestigation.pyc`, compiled with `py_compile` as shown in lab 1) and two helpers, `defense_fakebitinvestigation.py` and `search_attacks_fakebitinvestigation.py`, which contain no embedding or attack logic: they only use the files of `submission/`;
- [`fakebitinvestigation/documentation/`](fakebitinvestigation/documentation/): full explanation of the project ([`project_explanation.md`](fakebitinvestigation/documentation/project_explanation.md)) with the charts, and all the PDFs;
- [`project_explanation.pdf`](fakebitinvestigation/documentation/project_explanation.pdf): the explanation in PDF;
- [`test_group2022.pdf`](fakebitinvestigation/documentation/test_group2022.pdf): one page with the results of the test against a real group of the 2022 challenge.

Other folders:
- `fakebitinvestigation/documentation/document.pdf`: the updated rules (all the PDFs are in `documentation/`);
- `fakebitinvestigation/sample-images/`: the 101 sample images; `fakebitinvestigation/set_c/`: set C, 14 images never used to decide anything (a final "clean" test);
- `fakebitinvestigation/submission/fakebitinvestigation.npy`: our watermark (it is in the submission because the embedding and the demo need it);
- the experiments folder (measurement scripts, images and code of the 2022 group used for the test) was deleted on 21/10: nothing in the submission or in the challenge uses it. It can be recovered with `git checkout cc4c11d -- fakebitinvestigation/esperimenti fakebitinvestigation/gara2022`.
