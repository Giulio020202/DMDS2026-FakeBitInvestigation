# Capture the Mark 2026 — gruppo `fakebitinvestigation`

Versione consegnata: **2.0 ibrida (DCT + wavelet)**. La spiegazione completa dell'algoritmo è in `spiegazione_embedding.md`.

## File

| File | Cosa fa | Firma |
|---|---|---|
| `embedding_fakebitinvestigation.py` | inserisce il watermark | `embedding(input1, input2) -> output1` |
| `detection_fakebitinvestigation.py` | dice se il watermark c'è ancora | `detection(input1, input2, input3) -> output1, output2` |
| `attacks_fakebitinvestigation.py` | i nostri attacchi (solo quelli ammessi) | `attacks(input1, attack_name, param_array, mask=None) -> output1` |
| `attacchi_fakebitinvestigation.py` | script di attacco: cerca l'attacco più leggero che rompe ogni immagine | riga di comando |
| `roc_fakebitinvestigation.py` | ROC e soglia τ della detection | riga di comando |
| `demo_fakebitinvestigation.py` | mostra come usare tutti i file qui sopra | riga di comando |
| `difesa_fakebitinvestigation.py` | fase di difesa: prova dei target contro un attaccante forte, immagini da caricare | riga di comando |
| `ISTRUZIONI_GARA.md` | cosa fare passo passo il giorno della gara (difesa e attacco) | — |
| `roc_fakebitinvestigation.png` | curva ROC della run completa | — |
| `roc_fakebitinvestigation_output.txt` | numeri stampati dalla run completa della ROC | — |

Risorse nella cartella superiore (`../`): `fakebitinvestigation.npy` (il nostro watermark), `sample-images/` (101 immagini BMP 512×512 in grigi).

## Requisiti

Python 3.10 con `numpy`, `scipy`, `opencv-python`, `PyWavelets`.
Per attacchi, ROC e demo servono anche `pillow`, `scikit-image`, `scikit-learn`, `matplotlib`.

```bash
pip install numpy scipy opencv-python PyWavelets pillow scikit-image scikit-learn matplotlib
```

Tutti i comandi qui sotto si lanciano **dalla cartella `v2.0_ibrida/`**.

## Demo (consigliata come primo passo)

```bash
python demo_fakebitinvestigation.py                 # embedding + detection + 11 attacchi d'esempio (~15 s)
python demo_fakebitinvestigation.py --attacchi          # + script di attacco contro la nostra detection (~6 min)
python demo_fakebitinvestigation.py --roc           # + ROC veloce (solo per vedere che gira)
python demo_fakebitinvestigation.py --image ../sample-images/0042.bmp --out /tmp/demo
```

Stampa, per ogni caso, `output1`, `output2` (WPSNR), il tempo della detection e l'esito. Tutte le immagini prodotte finiscono in `demo_output/`.

## Embedding

```python
from embedding_fakebitinvestigation import embedding
wm = embedding("../sample-images/0000.bmp", "../fakebitinvestigation.npy")   # numpy uint8 512x512
```

- `input1` = percorso dell'immagine originale; `input2` = percorso del watermark `.npy` (1024 bit).
- Non stampa, non apre finestre, non scrive file. Il WPSNR della watermarkata è calibrato a `TARGET_WPSNR` (62,05 dB).

## Detection

```python
from detection_fakebitinvestigation import detection
output1, output2 = detection("originale.bmp", "watermarkata.bmp", "attaccata.bmp")
```

- `output1` = 1 se il watermark è presente, 0 se no; `output2` = WPSNR(watermarkata, attaccata) col codice dei lab.
- File unico: nessun percorso relativo, nessuno script esterno, nessuna lettura del watermark; l'unica funzione "esterna" è il WPSNR dei lab, copiato nel file.
- Decisione: presente se similarità ≥ `TAU` (costante in cima al file, ricavata dalla ROC).
- Tempo misurato: circa 0,6 s per chiamata (limite 5 s).

## Attacchi

```python
from attacks_fakebitinvestigation import attacks, smooth_mask
att = attacks("gruppoX_lena.bmp", "jpeg", [20])                          # singolo
att = attacks("gruppoX_lena.bmp", ["blur", "jpeg"], [[0.6], [12]])       # catena
att = attacks(wm, "awgn", [30], mask=smooth_mask(wm, 0.5))               # localizzato
```

| `attack_name` | `param_array` | Significato |
|---|---|---|
| `'awgn'` | `[std]` o `[std, seed]` | rumore gaussiano additivo |
| `'blur'` | `[sigma]` (o `[(sy, sx)]`) | filtro gaussiano (anche anisotropo) |
| `'sharp'` | `[sigma, alpha]` | unsharp masking |
| `'jpeg'` | `[QF]` | compressione JPEG, QF 1–100 |
| `'resize'` | `[scale]` o `[scale, metodo]` | giù e su a 512×512; metodo `'lab'`, `'cubic'`, `'linear'`, `'area'` |
| `'median'` | `[k]` (o `[(ky, kx)]`) | filtro mediano |

- Solo gli attacchi ammessi (p.6): AWGN, blur, sharpening, JPEG, resize, median, anche combinati o localizzati.
- Le zone degli attacchi localizzati (`MASKS`: zone lisce, testurizzate, bordi) sono calcolate **solo dalla watermarkata**, mai dall'originale.

## Script di attacco

```bash
python attacchi_fakebitinvestigation.py --images DIR --originals DIR --detections DIR --out DIR \
       [--only gruppo] [--procs N] [--total-min MINUTI] [--budget SECONDI] [--timeout SECONDI] [--no-greedy] [--resume]
```

| Opzione | Default | Significato |
|---|---|---|
| `--total-min` | 100 | minuti totali a disposizione; il tempo per immagine si calcola da questo, dal numero di immagini e dai processi |
| `--budget` | da `--total-min` | secondi per immagine (sovrascrive il calcolo automatico) |
| `--procs` | n. core − 1 | immagini attaccate in parallelo |
| `--timeout` | 10 | secondi massimi per una chiamata alla detection di un altro gruppo |
| `--resume` | — | riparte dopo un'interruzione, saltando le immagini già nel log |

Protezioni contro le detection degli altri gruppi:
- timeout per ogni chiamata;
- errori catturati: dopo 3 errori di fila l'immagine viene saltata e segnalata a fine run;
- stampe soppresse, nessuna finestra.

Il log `log_attacchi.csv` viene scritto man mano. `tutti_i_riusciti.csv` contiene tutti gli attacchi riusciti per ogni immagine, come alternative.

- `--images`: le watermarkate scaricate, `<gruppo>_<imageName>.bmp`;
- `--originals`: le originali, `<imageName>.bmp`;
- `--detections`: le detection degli altri gruppi, `detection_<gruppo>.py`;
- `--out`: le attaccate, `fakebitinvestigation_<gruppo>_<imageName>.bmp`, e il log `log_attacchi.csv` (Image | Group | WPSNR | Attack(s) with parameters).

Per ogni immagine:
1. prova 25 famiglie di attacchi, sia globali sia localizzate su 7 zone;
2. per ognuna cerca con ricerca binaria il parametro più leggero che rompe, usando come verifica l'esito della detection del gruppo attaccato (uso confermato dai TA);
3. "restituisce" alla watermarkata i blocchi 64×64 e 32×32 che non servono a rompere, per alzare il WPSNR, usando l'esito della detection del gruppo attaccato (ammesso dai TA il 19/10). La maschera delle zone viene salvata come PNG e citata nel log. Si salta con `--no-greedy`;
4. tiene il riuscito con il WPSNR più alto: se ne carica uno solo per immagine (p.7).

Noi non usiamo mai direttamente l'originale: lo riceve solo la detection altrui, come `input1`.

## ROC e soglia τ

```bash
python roc_fakebitinvestigation.py               # run completa: 101 immagini x 40 attacchi (~40 min su 14 core)
python roc_fakebitinvestigation.py --quick       # prova veloce (10 immagini x 5 attacchi): la τ NON è quella vera
python roc_fakebitinvestigation.py --images DIR --watermark FILE.npy --attacks N --out curva.png
```

Procedura di p.4 del regolamento più i controlli del Lab 4:
- **H1**: la stessa immagine watermarkata, attaccata;
- **H0-R**: watermark casuali, la procedura del regolamento;
- **H0-A**: lo stesso attacco applicato all'originale;
- **H0-B**: lo stesso attacco su un'immagine watermarkata da un "altro gruppo".

Le immagini sono divise in calibrazione (70%) e test (30%, mai usate per scegliere τ). τ è la soglia più bassa con FPR ≤ 0,1% su ciascun tipo di H0.

**Risultato della run completa (19/10):** soglie per FPR ≤ 0,1%:
- H0-R 0,0835;
- H0-A 0,0841;
- H0-B 0,0867.

Il massimo è 0,0867, arrotondato a **τ = 0,087** (`TAU` nella detection).
- Calibrazione: TPR 99,1%.
- Test, con τ congelata: FPR 0,058% / 0% / 0%, TPR 97,6% (99,6% sugli attacchi con WPSNR ≥ 35).

L'output completo è in `roc_fakebitinvestigation_output.txt`, la curva in `roc_fakebitinvestigation.png`.

## Per noi: se modifichiamo qualcosa

Dopo il 27/10 si possono cambiare solo parametri e posizioni di embedding (le costanti in cima a `embedding_…`). Ordine da seguire:

1. Cambiare la costante **in entrambi** `embedding_fakebitinvestigation.py` e `detection_fakebitinvestigation.py`. Devono restare identiche, a parte `TARGET_WPSNR`, che sta solo nell'embedding.
2. Rifare la ROC completa (`python roc_fakebitinvestigation.py`) e copiare la τ stampata in `TAU` nella detection.
3. Rilanciare `python demo_fakebitinvestigation.py`. Watermarkata → 1, originale → 0, tempi sotto i 5 s.
4. Aggiornare la τ in questo README e nella §A7 di `spiegazione_embedding.md`.
