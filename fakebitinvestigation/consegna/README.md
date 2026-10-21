# Capture the Mark 2026 — gruppo `fakebitinvestigation`

Versione consegnata: **2.0 ibrida (DCT + wavelet)**. La spiegazione completa dell'algoritmo è in `../documentazione/spiegazione_progetto.md` (in PDF: `../documentazione/spiegazione_progetto.pdf`).

## File

| File | Cosa fa | Firma |
|---|---|---|
| `embedding_fakebitinvestigation.py` | inserisce il watermark | `embedding(input1, input2) -> output1` |
| `detection_fakebitinvestigation.py` | dice se il watermark c'è ancora | `detection(input1, input2, input3) -> output1, output2` |
| `attacks_fakebitinvestigation.py` | **tutto** il nostro codice di attacco (solo attacchi ammessi): un attacco, una catena o un attacco localizzato; la ricerca automatica dell'attacco più leggero che rompe un'immagine | `attacks(input1, attack_name, param_array, mask=None) -> output1`, `attack_image(...)`, riga di comando |
| `roc_fakebitinvestigation.py` | ROC e soglia τ della detection | riga di comando |
| `demo_fakebitinvestigation.py` | mostra come usare tutti i file qui sopra | riga di comando |
| `roc_fakebitinvestigation.png` | curva ROC della run completa | — |
| `roc_fakebitinvestigation_output.txt` | numeri stampati dalla run completa della ROC | — |

In questa cartella ci sono anche `fakebitinvestigation.npy` (il nostro watermark, serve all'embedding come `input2`; la detection non lo legge) e `immagini_demo/` (10 immagini BMP 512×512 in grigi, usate dalla demo): **la cartella funziona da sola**. Per la ROC completa servono le 101 immagini di esempio del corso: per default lo script le cerca in `../sample-images/` (altrimenti usa `immagini_demo/`), oppure si indicano con `--images`.

Per il giorno della gara (non fanno parte della consegna) ci sono `../gara/ISTRUZIONI_GARA.md` (cosa fare passo passo, difesa e attacco), `../gara/difesa_fakebitinvestigation.py` (prova dei target contro un attaccante forte e preparazione delle immagini da caricare) e `../gara/cerca_attacchi_fakebitinvestigation.py` (attacca in parallelo tutte le immagini scaricate e scrive il log). Sono solo helper: non contengono logica di embedding né di attacco, usano i file di questa cartella.

## Requisiti

Python 3.10 con `numpy`, `scipy`, `opencv-python`, `PyWavelets`.
Per attacchi, ROC e demo servono anche `pillow`, `scikit-image`, `scikit-learn`, `matplotlib`.

```bash
pip install numpy scipy opencv-python PyWavelets pillow scikit-image scikit-learn matplotlib
```

Tutti i comandi qui sotto si lanciano **da questa cartella (`consegna/`)**.

## Demo (consigliata come primo passo)

```bash
python demo_fakebitinvestigation.py                 # embedding + detection + 11 attacchi d'esempio (~15 s)
python demo_fakebitinvestigation.py --attacchi          # + ricerca automatica dell'attacco (attack_image) contro la nostra detection (~5 min)
python demo_fakebitinvestigation.py --roc           # + ROC veloce (solo per vedere che gira)
python demo_fakebitinvestigation.py --image immagini_demo/0004.bmp --out /tmp/demo
```

Stampa, per ogni caso, `output1`, `output2` (WPSNR), il tempo della detection e l'esito. Tutte le immagini prodotte finiscono in `demo_output/`.

## Embedding

```python
from embedding_fakebitinvestigation import embedding
wm = embedding("immagini_demo/0000.bmp", "fakebitinvestigation.npy")   # numpy uint8 512x512
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

## Attacchi (un unico file: `attacks_fakebitinvestigation.py`)

Tutto il nostro codice di attacco sta in questo file, che funziona da solo: usa solo librerie (numpy, scipy, opencv, pillow, scikit-image) e il WPSNR dei lab, copiato dentro. Contiene tre cose.

**1. `attacks()`: un attacco, una catena o un attacco localizzato** (struttura d'esempio del regolamento, p.6).

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

Da riga di comando (salva l'attaccata e stampa il WPSNR):

```bash
python attacks_fakebitinvestigation.py applica gruppoX_lena.bmp out.bmp jpeg 20
python attacks_fakebitinvestigation.py applica gruppoX_lena.bmp out.bmp blur,jpeg 0.6 12
```

**2. `attack_image()`: la ricerca automatica dell'attacco più leggero che rompe un'immagine** (è quella che usiamo in gara).

```bash
python attacks_fakebitinvestigation.py cerca --image gruppoX_lena.bmp --original lena.bmp \
       --detection detection_gruppoX.py --out OUTDIR [--budget SECONDI] [--timeout SECONDI] [--no-greedy]
```

```python
from attacks_fakebitinvestigation import attack_image
r = attack_image("gruppoX_lena.bmp", "lena.bmp", "detection_gruppoX.py", "OUTDIR", budget=600)
# r["wpsnr"], r["desc"]: il riuscito con il WPSNR più alto (None = non rotta); r["error"] se qualcosa va storto
```

Per l'immagine data:
1. prova 25 famiglie di attacchi (singoli e catene, dal più leggero al più forte), sia globali sia localizzate su 7 zone;
2. un WPSNR veloce (FFT, identico a quello dei lab) scarta senza chiamare la detection tutto ciò che va sotto i 35 dB;
3. per ogni famiglia cerca con ricerca binaria il parametro più leggero che rompe, usando come verifica l'esito della detection del gruppo attaccato (uso confermato dai TA);
4. "restituisce" alla watermarkata i blocchi 64×64 e 32×32 che non servono a rompere, per alzare il WPSNR, sempre con l'esito della detection del gruppo attaccato (ammesso dai TA il 19/10). La maschera delle zone viene salvata come PNG e citata nella descrizione dell'attacco. Si salta con `--no-greedy`;
5. tiene il riuscito con il WPSNR più alto (p.7: se ne carica uno solo per immagine) e lo salva come `fakebitinvestigation_<gruppo>_<imageName>.bmp`.

**L'originale non viene mai letto né usato per costruire l'attacco**: il suo percorso viene passato solo alla detection del gruppo attaccato, come suo `input1` (regolamento p.6; risposta dei TA del 20/10: il divieto riguarda usare l'originale, per esempio `watermarkata − originale`, non chiamare la loro detection).

Protezioni contro le detection degli altri gruppi: timeout per ogni chiamata (`--timeout`, 10 s); errori catturati (dopo 3 di fila l'immagine viene saltata e l'errore restituito); stampe soppresse, nessuna finestra.

**3. La descrizione per il log** (`describe()`), nello stile dell'esempio del regolamento: `Blur sigma=0.6, JPEG QF=12`.

Il giorno della gara, per attaccare tutte le immagini di tutti i gruppi in parallelo e scrivere il log Excel, usiamo un piccolo helper che **non si consegna** e non contiene attacchi: `../gara/cerca_attacchi_fakebitinvestigation.py`, che chiama `attack_image()` per ogni immagine (istruzioni in `../gara/ISTRUZIONI_GARA.md`).

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
4. Aggiornare la τ in questo README e nella §A7 di `../documentazione/spiegazione_progetto.md`.
