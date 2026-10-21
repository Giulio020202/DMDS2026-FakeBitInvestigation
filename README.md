# DMDS2026 — FakeBitInvestigation

Repo del gruppo `fakebitinvestigation` per la challenge **Capture the Mark 2026** (Multimedia Data Security, Università di Trento).

La versione consegnata è la **2.0 ibrida (DCT + wavelet)**.

- [`fakebitinvestigation/consegna/`](fakebitinvestigation/consegna/): **i file da consegnare**, il 100% del lavoro (embedding, detection, ROC, attacchi in un unico file, demo) con il [`README.md`](fakebitinvestigation/consegna/README.md) per eseguirli;
- [`fakebitinvestigation/gara/`](fakebitinvestigation/gara/): **il giorno della challenge** (non si consegna): [`ISTRUZIONI_GARA.md`](fakebitinvestigation/gara/ISTRUZIONI_GARA.md) (passo passo) e due helper, `difesa_fakebitinvestigation.py` e `cerca_attacchi_fakebitinvestigation.py`, che non contengono logica di embedding o di attacco: usano solo i file di `consegna/`;
- [`fakebitinvestigation/documentazione/`](fakebitinvestigation/documentazione/): spiegazione completa del progetto ([`spiegazione_progetto.md`](fakebitinvestigation/documentazione/spiegazione_progetto.md)) con i grafici, e tutti i PDF;
- [`spiegazione_progetto.pdf`](fakebitinvestigation/documentazione/spiegazione_progetto.pdf): la spiegazione in PDF;
- [`test_gruppo2022.pdf`](fakebitinvestigation/documentazione/test_gruppo2022.pdf): una pagina con i risultati del test contro un gruppo vero della gara 2022.

Altre cartelle:
- `fakebitinvestigation/documentazione/document.pdf`: il regolamento aggiornato (tutti i PDF sono in `documentazione/`);
- `fakebitinvestigation/sample-images/`: le 101 immagini di prova; `fakebitinvestigation/set_c/`: il set C, 14 immagini mai usate per decidere (prova finale "pulita", per esempio per il test delle maschere);
- `fakebitinvestigation/consegna/fakebitinvestigation.npy`: il nostro watermark (sta nella consegna perché serve all'embedding e alla demo);
- la cartella degli esperimenti (script di misura, immagini e codice del gruppo 2022 usati per il test) è stata eliminata il 21/10: niente della consegna né della gara la usa. Si recupera con `git checkout cc4c11d -- fakebitinvestigation/esperimenti fakebitinvestigation/gara2022`.
