# Il giorno della gara (2/11): cosa fare, passo per passo

Guida operativa del gruppo `fakebitinvestigation` per la difesa (8:30–10:00) e l'attacco (10:15–12:30), sempre dentro il regolamento (`../document.pdf`). Tutti i comandi si lanciano **dalla cartella `fakebitinvestigation/v2.0_ibrida/`** con l'ambiente attivo:

```bash
cd fakebitinvestigation/v2.0_ibrida
source ../../OFF_MDS_ENV/bin/activate
```

> 🗣 **In parole povere**:
> - **in difesa** proviamo ad attaccare noi stessi per decidere quanto "forte" mettere il watermark su ogni immagine, poi carichiamo;
> - **in attacco** lasciamo lavorare `attacchi_fakebitinvestigation.py`, che per ogni immagine degli altri cerca l'attacco più leggero che toglie il watermark, e carichiamo man mano.

---

## 0. Prima del 2/11 (a casa)

1. **Un solo PC accede al sito; al massimo 3 laptop; niente server** (regolamento). Il PC "di calcolo" deve avere molti core: lo script usa `core − 1` processi.
2. Sul PC di calcolo: repo aggiornato (`git pull`), ambiente `OFF_MDS_ENV` funzionante, e prova completa:
   ```bash
   python demo_fakebitinvestigation.py --attacchi      # ~6 min: deve finire senza errori
   ```
3. **Librerie che le detection degli altri potrebbero usare**, da installare prima (in gara non c'è tempo):
   ```bash
   pip install numpy scipy opencv-python PyWavelets scikit-image scikit-learn matplotlib pillow
   ```
4. Preparare le cartelle vuote:
   ```bash
   mkdir -p ~/gara/difesa/originali ~/gara/difesa/out \
            ~/gara/attacco/img ~/gara/attacco/orig ~/gara/attacco/det ~/gara/attacco/out
   ```
5. **Ruoli** (3 persone):
   - **A** = PC di calcolo, lancia gli script;
   - **B** = sito, cioè download e upload;
   - **C** = controlla nomi dei file, regole e orari, e tiene questa guida aperta.

---

## 1. FASE DI DIFESA (8:30–10:00, 90 minuti)

Obiettivo: caricare le 3 immagini watermarkate e la detection. Il target di WPSNR si può scegliere **immagine per immagine**, perché i punti qualità si calcolano sulla **media** delle 3 (risposta dei TA).

| quando | chi | cosa |
|---|---|---|
| 8:30 | B | scarica le 3 immagini originali in `~/gara/difesa/originali/` (nomi originali, es. `buildings.bmp`) |
| 8:32 | A | **versione di sicurezza**: `python difesa_fakebitinvestigation.py --images ~/gara/difesa/originali --out ~/gara/difesa/sicurezza --final` (tutte a 62,05, ~30 s) |
| 8:35 | B | carica subito le 3 immagini di `~/gara/difesa/sicurezza/upload/` e la detection. Le immagini si possono ricaricare più volte in difesa (regolamento), quindi così siamo coperti anche se dopo qualcosa va storto |
| 8:35 | A | **passo 1.1** (prova dei target) |
| ~8:55 | tutti | **passo 1.2** (scelta) |
| ~9:00 | A | **passo 1.3** (immagini finali) |
| entro 9:30 | B | **passo 1.4** (upload definitivo, sostituisce la versione di sicurezza), con mezz'ora di margine |

### 1.1 Prova dei target (~17 min)

```bash
python difesa_fakebitinvestigation.py --images ~/gara/difesa/originali --out ~/gara/difesa/out
```

Per ogni immagine e ogni target (58,05 e 62,05 dB):
1. fa l'embedding;
2. controlla che watermarkata → 1, originale → 0 e che la detection stia sotto i 5 s;
3. **attacca noi stessi** con lo stesso motore dello script di attacco, al massimo 15 minuti per prova, 6 prove in parallelo.

Alla fine stampa la classifica delle 8 combinazioni di target con i punti di difesa stimati, e la riga da copiare, per esempio:

```
TARGET_BY_IMAGE = {'buildings': 58.05, 'rollercoaster': 62.05, 'tree': 62.05}
```

Se una riga dice `!! CONTROLLARE !!` (watermarkata non trovata, originale trovata, detection lenta): **fermarsi e guardare**, non caricare niente.

### 1.2 La scelta (5 minuti, tutti insieme)

La simulazione usa **un solo** avversario, per quanto forte; in gara ci attaccano **9 gruppi**. Regole pratiche:

- **Il bonus +2 (mai rotti) è fragile.** Basta che uno solo dei 9 gruppi ci rompa una volta. Se una combinazione vince solo grazie al bonus, chiedersi se reggerebbe contro 9 avversari.
- **Contro un attaccante forte la v2.0 viene rotta spesso a 62 dB** (70% delle immagini nelle nostre prove), ma quasi sempre a 35–38 dB, quindi con 6 punti di robustezza su 6. Scendere a 58 dB serve soprattutto a puntare al bonus.
- Dalle nostre simulazioni (§A8.3 di `spiegazione_embedding.md`): 11,10 punti attesi a 58,05 contro 10,88 a 62,05. È una differenza piccola.
- **Non scendere sotto 58,05 di media**, altrimenti si perde un altro punto di qualità.

Se il tempo stringe: **tutte a 62,05** (default del codice, nessuna modifica da fare).

### 1.3 Immagini finali (2 minuti)

1. Aprire `embedding_fakebitinvestigation.py` e incollare la riga scelta al posto di `TARGET_BY_IMAGE = {}`. È un **parametro**, quindi è ammesso come fine-tuning. **La detection NON si tocca**: τ non dipende dal target.
2. Generare le immagini da caricare **con il codice così com'è**, così codice e immagini caricate coincidono:
   ```bash
   python difesa_fakebitinvestigation.py --images ~/gara/difesa/originali --out ~/gara/difesa/out --final
   ```
   Deve stampare `OK` su tutte e 3 e la media WPSNR. Le immagini sono in `~/gara/difesa/out/upload/`, già con il nome `fakebitinvestigation_<imageName>.bmp`.
3. Salvare lo stato nel repo, così resta la prova di cosa abbiamo caricato:
   ```bash
   git add -A && git commit -m "Gara: target scelti per le 3 immagini" && git push
   ```

### 1.4 Upload (B)

- Le 3 immagini di `~/gara/difesa/out/upload/`.
- **`detection_fakebitinvestigation.py`** (questa cartella, file unico).
- **Dopo l'upload non si modifica più nessun file caricato**: penalità di 2 punti.

---

## 2. FASE DI ATTACCO (10:15–12:30, 135 minuti)

Obiettivo: rompere più immagini possibile, di **tutti** i gruppi, con il WPSNR più alto possibile. Come si fanno i punti in attacco:

- **attività**, cioè la % di gruppi attaccati con successo: >90% → 6 punti. Con 9 gruppi avversari, 8 su 9 fa l'89%, quindi **serve almeno un'immagine rotta per OGNI gruppo**;
- **qualità**: numero di immagini rotte con WPSNR sopra la media di tutti gli attacchi (>15 → 4 punti);
- **bonus +2** se rompiamo un gruppo che nessun altro rompe.

### 2.1 Preparazione (10:15–10:25)

B scarica dal sito, A mette i file nelle cartelle **con questi nomi** (lo script li riconosce dal nome):

| cartella | contenuto | nome |
|---|---|---|
| `~/gara/attacco/img/` | le watermarkate degli altri gruppi (27 = 9 gruppi × 3) | `<gruppo>_<imageName>.bmp` (il formato del sito) |
| `~/gara/attacco/orig/` | le originali | `<imageName>.bmp` |
| `~/gara/attacco/det/` | le detection degli altri | `detection_<gruppo>.py` (rinominare se serve) |

Le originali servono **solo** alla detection degli altri, che le vuole come `input1`. Noi non le usiamo mai per scegliere o regolare un attacco: lo script le passa e basta. Usare l'esito della detection altrui per guidare l'attacco è **ammesso dai TA**.

### 2.2 Controllo delle detection (1 minuto)

```bash
python attacchi_fakebitinvestigation.py --images ~/gara/attacco/img --originals ~/gara/attacco/orig \
       --detections ~/gara/attacco/det --out ~/gara/attacco/out --check
```

Per ogni immagine stampa `ok, X s per chiamata`, oppure:
- `ERRORE: ModuleNotFoundError: …`: manca una libreria → `pip install <libreria>` e ripetere il check;
- `ERRORE: …` (altro): quella detection è rotta. Lo script la salterà; segnarla e, se c'è tempo, guardarla a mano;
- `ATTENZIONE: dice 'assente' sulla watermarkata`: nome file sbagliato (originale e watermarkata non corrispondono?) o detection difettosa. Controllare i nomi.

### 2.3 Attacco (10:30 → al massimo ~12:10)

```bash
python attacchi_fakebitinvestigation.py --images ~/gara/attacco/img --originals ~/gara/attacco/orig \
       --detections ~/gara/attacco/det --out ~/gara/attacco/out --total-min 100
```

- **Tempo misurato** (simulazione del 19/10: 27 immagini, 9 gruppi × 3, 13 processi, detection da circa 0,6 s, con il greedy acceso; senza è più veloce): **32 minuti** in tutto, con 18 immagini rotte su 27, almeno una per ognuno dei 9 gruppi, a 35,6–38,0 dB. Le immagini che non si rompono finiscono prima, in circa 7 minuti, perché esauriscono i tentativi. Con detection più lente il tempo cresce, ma il tetto resta `--total-min`.
- `--total-min 100`: lo script si dà 100 minuti in tutto e calcola da solo il tempo per immagine. **Partendo alle 10:30 finisce entro le 12:10**, con 20 minuti di margine per gli ultimi upload. Se si parte più tardi, abbassarlo, per esempio `--total-min 80` partendo alle 10:50.
- Stampa una riga per immagine man mano che finisce:
  ```
  [5/27 14 min] [ROTTA] gruppoX_buildings: 37.12 dB con Median 3x3, JPEG QF=7 (480 chiamate, 0 errori, 320 s)
  ```
- **Upload man mano (B)**: ogni `[ROTTA]` ha già l'immagine pronta in `~/gara/attacco/out/` con il nome giusto, `fakebitinvestigation_<gruppo>_<imageName>.bmp`. Lo script stampa una sola riga per immagine, ed è già il miglior attacco trovato, quindi si carica **un'immagine sola per immagine attaccata** come vuole il regolamento.
- **Se lo script si interrompe** (crash, PC spento): rilanciare lo stesso comando con `--resume` e un `--total-min` ridotto. Riparte saltando le immagini già fatte.

### 2.4 Seconda passata (se avanza tempo)

Alla fine lo script stampa `rotte X/27` e l'elenco delle immagini con errori. Poi:

- **Un gruppo senza nessuna immagine rotta** vale punti di attività (e forse il bonus +2): rilanciare solo su quel gruppo, con più tempo e su una cartella di output diversa, così non si sovrascrive il log principale:
  ```bash
  python attacchi_fakebitinvestigation.py ... --out ~/gara/attacco/out_gruppoX --only gruppoX --budget 900
  ```
- Le alternative per ogni immagine (tutti gli attacchi riusciti, non solo il migliore) sono in `~/gara/attacco/out/tutti_i_riusciti.csv`.

### 2.5 Dopo la gara: il log

Il regolamento chiede un **log Excel** con colonne Image | Group | WPSNR | Attack(s) with parameters. È già pronto in `~/gara/attacco/out/log_attacchi.csv`: aprirlo con Excel o LibreOffice e salvarlo come `.xlsx`. Le descrizioni sono già nello stile dell'esempio del regolamento (`JPEG QF=6, Blur sigma=0.8`). Quando un attacco è stato ristretto a zone scelte con la detection, la descrizione cita un file `zone_fakebitinvestigation_<gruppo>_<immagine>.png`, cioè la maschera delle zone attaccate. **Va mandato insieme al log**, perché senza non si può riprodurre l'attacco. Se c'è stata una seconda passata, unire le righe dei vari `log_attacchi.csv`, tenendo per ogni immagine **solo** quella caricata.

---

## 3. Cosa NON fare (regolamento)

| vietato | perché / penalità |
|---|---|
| modificare un file caricato dopo la fase di difesa | −2 punti |
| cambiare nell'embedding qualcosa che non sia un parametro o una posizione (dopo il 27/10) | solo fine-tuning ammesso |
| cambiare un parametro "di posizione" (R_MIN, R_MAX, DWT_USE, …) solo nell'embedding | va cambiato **in entrambi** i codici prima dell'upload; meglio non toccarli in gara |
| usare l'originale per scegliere, regolare o localizzare un attacco | lo script la passa solo alla detection altrui; le zone vengono dalla sola watermarkata |
| attacchi diversi da AWGN, blur, sharpening, JPEG, resize, median (anche combinati) | solo questi sono ammessi; lo script usa solo questi |
| software esterni (Photoshop, GIMP, …) | vietati |
| caricare più attacchi della stessa immagine | se ne carica uno, il migliore |
| server o più di 3 laptop, più di un PC sul sito | vietato |

## 4. Problemi frequenti

| problema | soluzione |
|---|---|
| `!! CONTROLLARE !!` in difesa | non caricare; ricontrollare che `TARGET_BY_IMAGE` sia scritto bene (nomi delle immagini **senza** `.bmp`) e rifare `--final` |
| il check dà `ModuleNotFoundError` | `pip install` della libreria, poi ripetere il check |
| una detection altrui è lentissima | il timeout (10 s) la ferma: dopo 3 timeout di fila l'immagine viene saltata. Si può provare `--only gruppo --timeout 20` alla fine |
| lo script si è fermato | rilanciare con `--resume` |
| siamo in ritardo | ridurre `--total-min` |
| i TA hanno confermato che le zone scelte con la detection sono ammesse | aggiungere `--greedy` al comando dell'attacco (+0,2–0,4 dB per immagine); **senza conferma scritta non usarlo** |
