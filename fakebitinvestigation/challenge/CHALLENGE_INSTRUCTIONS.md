# Il giorno della gara (2/11): cosa fare, passo per passo

Guida operativa del gruppo `fakebitinvestigation` per la **fase di difesa** (8:30–10:00) e la **fase di attacco** (10:15–12:30). Tutto quello che c'è qui rispetta il regolamento (`../documentation/document.pdf`): se qualcosa non torna, vale il PDF, non questa guida.

> 🗣 **In parole povere**:
>
> - **in difesa** mettiamo il watermark nelle 3 immagini, proviamo ad attaccarci da soli per decidere quanto "forte" farlo su ciascuna, poi carichiamo immagini e detection;
> - **in attacco** lasciamo lavorare l'helper `search_attacks_fakebitinvestigation.py`: per ogni immagine degli altri gruppi chiama la ricerca automatica del nostro file degli attacchi (`attack_image()` in `attacks_fakebitinvestigation.py`), salva l'immagine attaccata con il nome giusto e scrive il log; noi carichiamo man mano.

**Indice**: Convenzioni · 0. Prima del 2/11 (a casa) · 1. Fase di difesa · 2. Fase di attacco · 3. Cosa NON fare · 4. Problemi frequenti · 5. Promemoria dei comandi.

---

## Convenzioni di questa guida (leggere prima di tutto)

1. **Tutti i comandi si lanciano dalla cartella `fakebitinvestigation/submission/`**, con l'ambiente Python attivo. È lì che c'è l'unica copia del codice (embedding, detection, attacchi); gli helper della gara stanno in `../challenge/` e si chiamano con `python ../challenge/...`. Per entrarci, partendo dalla radice della repo `DMDS2026-FakeBitInvestigation/`:
   ```bash
   cd fakebitinvestigation/submission
   source ../../OFF_MDS_ENV/bin/activate
   ```
   Il prompt deve mostrare `(OFF_MDS_ENV)` e `python --version` deve dire `Python 3.10.x`. In ogni nuovo terminale questi due comandi vanno rilanciati.
2. **Nei blocchi di comandi non ci sono commenti** (`# ...`): si possono copiare e incollare così come sono. Attenzione: zsh, di default, **non** tratta `#` come commento sulla riga di comando. Incollando `python demo_fakebitinvestigation.py --attacks   # ~5 min` si ottiene l'errore `zsh: not enough directory stack entries` (zsh legge `~5` come "quinta cartella dello stack"). Se si vuole incollare qualcosa con i commenti, prima: `setopt interactivecomments`.
3. Le cartelle di lavoro della gara sono tutte sotto **`~/ctm/`** (fuori dalla repo, così non finiscono su git per sbaglio).
4. I comandi lunghi sono spezzati su più righe con `\` alla fine: si incolla tutto il blocco insieme.
5. Gli output riportati come esempio servono a mostrare il **formato**: i numeri veri saranno diversi (tranne dove è scritto "misurato").
6. **Ruoli** (3 persone):
   - **A** = PC di calcolo: lancia gli script e legge l'output;
   - **B** = sito della gara (è **l'unico PC che accede al sito**: regolamento), cioè download e upload;
   - **C** = controlla nomi dei file, regolamento e orari, tiene aperta questa guida e spunta i passi.

---

## 0. Prima del 2/11 (a casa)

### 0.1 Regole sul materiale (regolamento, p.2)

- **Un solo PC per gruppo accede al sito della gara.**
- **Al massimo 3 laptop. Nessun server** (niente macchine remote, cluster, cloud).
- Il PC di calcolo (A) deve avere più core possibili: gli script usano `numero di core − 1` processi in parallelo (per esempio 13 processi su un PC da 14 core).

### 0.2 Preparare il PC di calcolo

1. Aggiornare la repo:
   ```bash
   cd ~/Scrivania/DMDS2026-FakeBitInvestigation
   git pull
   ```
2. Se l'ambiente `OFF_MDS_ENV` non c'è ancora, crearlo con Python 3.10 (dalla radice della repo):
   ```bash
   python3.10 -m venv OFF_MDS_ENV
   source OFF_MDS_ENV/bin/activate
   pip install numpy scipy==1.14.1 opencv-python PyWavelets pillow scikit-image scikit-learn matplotlib pandas
   ```
   Python **deve** essere il 3.10: la nostra detection cifrata (`.pyc`) e quelle degli altri gruppi funzionano solo con la stessa versione di Python con cui sono state compilate.
3. **Librerie che potrebbero servire alle detection degli altri gruppi**: vanno installate prima, il giorno della gara non c'è tempo. Sono quelle dei lab 1 e 2 (ambiente indicato dai TA nella mail del 26/10); con il comando del punto 2 ci sono già tutte. Controllo:
   ```bash
   python -c "import numpy, scipy, cv2, pywt, PIL, skimage, sklearn, matplotlib, pandas; print('ok', scipy.__version__)"
   ```
   Deve stampare `ok 1.14.1`.
4. **Prova completa** (dalla cartella `submission/`, ambiente attivo). Dura circa 5 minuti e deve finire senza errori, con watermarkata → 1 e originale → 0:
   ```bash
   python demo_fakebitinvestigation.py --attacks
   ```
5. **Prova della versione di sicurezza** (lo stesso comando del passo 1.1 del giorno della gara) sulle immagini della demo, per vedere che tutto il giro funzioni sul PC che useremo:
   ```bash
   mkdir -p ~/ctm/prova/originals
   cp demo_images/0000.bmp demo_images/0001.bmp demo_images/0002.bmp ~/ctm/prova/originals/
   python ../challenge/defense_fakebitinvestigation.py --images ~/ctm/prova/originals --out ~/ctm/prova/out --final
   ```
   Risultato **misurato il 27/10** (PC da 14 core, CPU libera): **26 s in tutto**, WPSNR 62,051 / 62,050 / 62,053 dB, ogni riga con `watermarked->1, original->0` e `OK`, media 62,051 dB → 5 punti di qualità, e in `~/ctm/prova/out/upload/` i file `fakebitinvestigation_0000.bmp`, `fakebitinvestigation_0001.bmp`, `fakebitinvestigation_0002.bmp` e `detection_fakebitinvestigation.pyc`. Alla fine si cancella la prova:
   ```bash
   rm -r ~/ctm/prova
   ```

### 0.3 Preparare le cartelle vuote della gara

```bash
mkdir -p ~/ctm/defense/originals ~/ctm/defense/out \
         ~/ctm/attack/img ~/ctm/attack/orig ~/ctm/attack/det ~/ctm/attack/out
ls ~/ctm/defense/originals ~/ctm/attack/img ~/ctm/attack/orig ~/ctm/attack/det
```

Le cartelle devono essere **vuote** (il secondo comando non deve elencare nessun file). Se c'è qualcosa di una prova precedente va tolto: gli script prendono **tutti** i `.bmp` della cartella.

### 0.4 Il giorno prima

- PC carichi, alimentatori, eventuale presa multipla.
- Password del sito (arrivata via mail dopo l'iscrizione: va tenuta segreta).
- Questa guida aperta (anche in PDF: `CHALLENGE_INSTRUCTIONS.pdf`, nella stessa cartella).
- Tutti e tre devono sapere cosa fa ogni script: alla presentazione del 4/11 ogni membro deve saper rispondere.

---

## 1. FASE DI DIFESA (8:30–10:00, 90 minuti)

**Obiettivo**: caricare sul sito le 3 immagini watermarkate e la detection. Cosa dice il regolamento (p.2):

1. si scaricano le 3 immagini della gara (512×512, grigi, BMP);
2. si scarica il codice del WPSNR (è lo stesso dei lab);
3. si fa l'embedding del nostro watermark nelle 3 immagini;
4. si caricano le immagini watermarkate: **si possono caricare più volte**, e ogni volta viene assegnato un punteggio di qualità;
5. si carica il codice della detection (gli altri gruppi la useranno in attacco).

In questa fase **si possono ritoccare i parametri di embedding e di detection**, ma ogni modifica va fatta **in entrambi i codici** prima dell'upload.

Il target di WPSNR si può scegliere **immagine per immagine**: i punti di qualità si calcolano sulla **media** del WPSNR delle 3 immagini (risposta dei TA del 14/10).

### Tabella di marcia

| ora | chi | cosa | passo |
|---|---|---|---|
| 8:30 | B | scarica le 3 immagini originali (e il codice del WPSNR) e le passa ad A | 1.0 |
| 8:31 | A | mette le 3 immagini in `~/ctm/defense/originals/` e controlla nomi e formato | 1.0 |
| 8:32 | A | **versione di sicurezza**: tutte a 62,05 dB (~30 s) | 1.1 |
| 8:34 | B | carica subito le 3 immagini e la detection della versione di sicurezza | 1.1 |
| 8:34 | A | lancia la **prova dei target** (~15–17 min) | 1.2 |
| 8:35 | C | confronta il WPSNR scaricato dal sito con il nostro | 1.0bis |
| ~8:52 | tutti | **scelta** dei target | 1.3 |
| ~8:57 | A | **file finali** (`--final`) | 1.4 |
| entro 9:30 | B | **upload finale** (sostituisce la versione di sicurezza) | 1.5 |
| 9:30–10:00 | C | margine e controlli finali sul sito | 1.6 |

Si punta a finire **entro le 9:30**, con mezz'ora di margine: all'ultimo minuto il sito può essere lento (regolamento, p.2).

### 1.0 Scaricare le immagini (B, poi A)

1. B scarica dal sito le 3 immagini e le passa ad A (chiavetta, cartella condivisa, …).
2. A le mette in `~/ctm/defense/originals/` **con il nome originale** (per esempio `buildings.bmp`, `rollercoaster.bmp`, `tree.bmp`). Il nome conta: diventa il nome del file da caricare (`fakebitinvestigation_buildings.bmp`) e la chiave di `TARGET_BY_IMAGE`.
3. A controlla che siano 3, BMP, 512×512:
   ```bash
   ls ~/ctm/defense/originals
   python -c "import cv2, glob; [print(f, cv2.imread(f, cv2.IMREAD_UNCHANGED).shape) for f in sorted(glob.glob('$HOME/ctm/defense/originals/*.bmp'))]"
   ```
   Ogni riga deve finire con `(512, 512)`. Se esce `(512, 512, 3)` l'immagine è salvata a colori: va bene lo stesso (il nostro embedding la legge in scala di grigi), ma conviene segnalarlo.

### 1.0bis Controllare il WPSNR del sito (C, appena è pronta la versione di sicurezza)

Il regolamento dice che il WPSNR della gara è "the same code provided during laboratories", e il nostro codice usa esattamente quello. Per esserne sicuri, C confronta il file scaricato con il nostro su un'immagine della versione di sicurezza (passo 1.1). Se il file scaricato è `~/Scaricati/wpsnr.py` e la funzione si chiama `wpsnr`:

```bash
python -c "
import importlib.util, glob, cv2
from detection_fakebitinvestigation import wpsnr as nostro
s = importlib.util.spec_from_file_location('sito', '$HOME/Scaricati/wpsnr.py'); m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
o = sorted(glob.glob('$HOME/ctm/defense/originals/*.bmp'))[0]; n = o.split('/')[-1]
a = cv2.imread(o, 0); b = cv2.imread('$HOME/ctm/defense/safety/upload/fakebitinvestigation_' + n, 0)
print('nostro', nostro(a, b), 'sito', m.wpsnr(a, b))
"
```

Se il file o la funzione hanno un altro nome, cambiare `wpsnr.py` e `m.wpsnr`. I due numeri devono essere **uguali**. Se sono diversi: **fermarsi e avvisare tutti**, perché tutta la calibrazione del target (62,05 dB) è fatta con il nostro WPSNR.

### 1.1 Versione di sicurezza (A, ~30 s; poi B carica)

```bash
python ../challenge/defense_fakebitinvestigation.py --images ~/ctm/defense/originals --out ~/ctm/defense/safety --final
```

Cosa fa: embedding delle 3 immagini con il codice così com'è (target 62,05 dB per tutte, perché `TARGET_BY_IMAGE = {}`), controlli (watermarkata → 1, originale → 0, tempo della detection), e prepara in `~/ctm/defense/safety/upload/` i file da caricare, compresa la detection cifrata compilata dal codice attuale.

Output atteso (formato; i nomi saranno quelli delle immagini vere):

```
TARGET_WPSNR = 62.05, TARGET_BY_IMAGE = {}, TAU = 0.087
  buildings            WPSNR 62.051 dB  watermarked->1, original->0, detection 0.07 s  OK
  rollercoaster        WPSNR 62.050 dB  watermarked->1, original->0, detection 0.06 s  OK
  tree                 WPSNR 62.053 dB  watermarked->1, original->0, detection 0.06 s  OK
mean WPSNR 62.051 dB -> 5 quality points
encrypted detection: /home/.../ctm/defense/safety/upload/detection_fakebitinvestigation.pyc (Python 3.10)

files to upload in /home/.../ctm/defense/safety/upload
```

Controlli di A, uno per uno:

- [ ] la prima riga dice `TARGET_BY_IMAGE = {}` e `TAU = 0.087`;
- [ ] 3 righe, tutte con WPSNR tra 62,050 e 62,055, `watermarked->1, original->0` e `OK` in fondo;
- [ ] la detection impiega meno di 5 s (di solito meno di 1 s);
- [ ] la riga della detection cifrata dice `(Python 3.10)`;
- [ ] l'ultima riga è `files to upload in ...` e **non** `!! some check did NOT pass: do not upload !!`;
- [ ] `ls ~/ctm/defense/safety/upload` mostra 4 file: le 3 `fakebitinvestigation_<nomeImmagine>.bmp` e `detection_fakebitinvestigation.pyc`.

Poi **B carica subito** sul sito i 4 file di `~/ctm/defense/safety/upload/`. Le immagini si possono caricare più volte durante la difesa, quindi da questo momento siamo coperti anche se dopo qualcosa va storto.

> 🗣 **In parole povere**: prima di tutto mettiamo al sicuro una consegna valida (5 punti di qualità, robustezza già misurata). Tutto il resto della difesa serve solo a migliorarla.

### 1.2 Prova dei target (A, ~15–17 min)

```bash
python ../challenge/defense_fakebitinvestigation.py --images ~/ctm/defense/originals --out ~/ctm/defense/out
```

Cosa fa, per ogni immagine e per ogni target (58,05 e 62,05 dB, cioè 6 prove):

1. fa l'embedding a quel target e salva l'immagine in `~/ctm/defense/out/target_<t>/img/`;
2. controlla che watermarkata → 1, originale → 0 e detection sotto i 5 s;
3. **ci attacca** con lo stesso motore che useremo nella fase di attacco (`attack_image()` di `attacks_fakebitinvestigation.py`), con la nostra detection come oracolo: al massimo 15 minuti per prova (`--budget 900` di default), le 6 prove in parallelo.

La prova dura quindi circa 15 minuti più l'embedding: va lanciata **subito** dopo la versione di sicurezza.

Durante la prova esce una riga per ciascuna delle 6 prove (formato):

```
  buildings            target 58.05: not broken  (612 calls, 900 s)
  buildings            target 62.05: BROKEN at 36.80 dB with Median 3x3, JPEG QF=7  (530 calls, 900 s)
```

Alla fine stampa la classifica delle 8 combinazioni di target (2 target per 3 immagini = 2³ = 8) con i punti di difesa stimati contro questo avversario, cioè qualità (sulla media dei WPSNR) + robustezza (sulla media dei WPSNR degli attacchi riusciti; 6 + 2 di bonus se nessuna immagine viene rotta), e la riga da copiare, per esempio:

```
  TARGET_BY_IMAGE = {'buildings': 58.05, 'rollercoaster': 62.05, 'tree': 62.05}
```

Se una riga dice **`!! CHECK !!`** (watermarkata non rilevata, originale rilevata o detection lenta): **fermarsi e guardare**, non caricare niente di questa prova (la versione di sicurezza resta valida).

### 1.3 La scelta (tutti, 5 minuti)

La simulazione usa **un solo** avversario, per quanto forte; in gara ci attaccano **tutti gli altri gruppi** (nelle nostre stime ne abbiamo ipotizzati 9; il numero vero si vede sul sito). Regole pratiche:

- **Il bonus di +2 (mai rotti) è fragile.** Basta che uno solo degli altri gruppi ci rompa una volta. Se una combinazione vince solo grazie al bonus, chiedersi se reggerebbe contro tutti.
- **Contro un attaccante forte la v2.0 viene rotta spesso a 62 dB** (70% delle immagini nei nostri test), ma quasi sempre tra 35 e 38 dB, quindi con 6 punti di robustezza su 6. Scendere a 58 dB serve soprattutto a puntare al bonus.
- Dalle nostre simulazioni (§A8.3 di `../documentation/project_explanation.md`): 11,10 punti attesi a 58,05 contro 10,88 a 62,05. È una differenza piccola.
- **Qualsiasi immagine a 58,05 costa un punto di qualità**: con una a 58,05 e due a 62,05 la media è circa 60,72, cioè fascia 58–62 → 4 punti invece di 5. Lo si fa solo se la prova dice che quel punto si recupera in robustezza.
- **La media non deve mai scendere sotto 58**, altrimenti si perde un altro punto (54–58 → 3 punti). Con i target 58,05 e 62,05 la media sta sempre sopra 58.

**Se il tempo stringe: tutte a 62,05.** È il default del codice, non si cambia niente e la versione di sicurezza già caricata è quella definitiva: si salta al passo 1.6.

### 1.4 File finali (A, 2–3 minuti)

1. Aprire `embedding_fakebitinvestigation.py` (in `submission/`) e alla **riga 34** sostituire
   ```
   TARGET_BY_IMAGE = {}
   ```
   con la riga scelta, per esempio
   ```
   TARGET_BY_IMAGE = {'buildings': 58.05, 'rollercoaster': 62.05, 'tree': 62.05}
   ```
   Le chiavi sono i nomi delle immagini **senza** `.bmp`, scritti esattamente come i file in `~/ctm/defense/originals/` (maiuscole comprese). Un nome sbagliato non dà errore: quell'immagine resta semplicemente a 62,05 (lo si vede dal WPSNR al punto 2).
   - È un **parametro**, quindi è un fine-tuning ammesso.
   - **La detection NON si tocca**: τ non dipende dal target e la detection non contiene il target.
   - **Nient'altro si tocca** nell'embedding (`R_MIN`, `R_MAX`, `DWT_USE`, … sono "posizioni": se cambiassero andrebbero cambiate anche nella detection, ricompilato il `.pyc` e rivista la soglia; il giorno della gara **non** si fa).
2. Generare i file da caricare **con il codice così com'è**, in modo che codice e immagini caricate coincidano:
   ```bash
   python ../challenge/defense_fakebitinvestigation.py --images ~/ctm/defense/originals --out ~/ctm/defense/out --final
   ```
   Controlli di A:
   - [ ] la prima riga mostra il `TARGET_BY_IMAGE` appena scritto e `TAU = 0.087`;
   - [ ] il WPSNR di ogni immagine corrisponde al suo target (58,05x oppure 62,05x);
   - [ ] tutte e 3 le righe finiscono con `OK`;
   - [ ] la media stampata dà i punti di qualità attesi;
   - [ ] l'ultima riga è `files to upload in .../ctm/defense/out/upload`.

   I file sono in `~/ctm/defense/out/upload/`: le 3 immagini, già con il nome `fakebitinvestigation_<nomeImmagine>.bmp`, e la detection cifrata `detection_fakebitinvestigation.pyc`, compilata dal `detection_fakebitinvestigation.py` attuale.
3. Salvare lo stato nella repo, così resta traccia di cosa abbiamo caricato (dalla cartella `submission/`):
   ```bash
   git add -A ..
   git commit -m "Gara: target scelti per le 3 immagini"
   git push
   ```

### 1.5 Upload finale (B)

- Le **3 immagini** di `~/ctm/defense/out/upload/` (sostituiscono quelle della versione di sicurezza).
- La **detection cifrata** `detection_fakebitinvestigation.pyc` di `~/ctm/defense/out/upload/` (i professori la vogliono cifrata con `py_compile`, come nel lab 1). Se è già stata caricata quella della versione di sicurezza e la detection non è cambiata, è equivalente: la detection non dipende dal target.
- Se per qualche motivo si fosse cambiato un parametro della detection, il `.pyc` deve essere quello prodotto da `--final` (oppure va ricompilato, sempre con Python 3.10, da `submission/`, con:
  `python -c "import py_compile; py_compile.compile('detection_fakebitinvestigation.py', cfile='../challenge/detection_fakebitinvestigation.pyc', dfile='detection_fakebitinvestigation.py')"`).
- Prima di caricare, C controlla i **nomi**: `fakebitinvestigation_<nomeImmagine>.bmp` (regolamento, p.7).

### 1.6 Controlli finali (C, entro le 10:00)

- [ ] Sul sito ci sono le 3 immagini giuste (quelle di `--final`, se abbiamo cambiato i target) e la detection.
- [ ] Il punteggio di qualità mostrato dal sito corrisponde a quello stampato da `--final`.
- [ ] **Dopo la fine della fase di difesa nessun file caricato si cambia più**: penalità di 2 punti.

---

## 2. FASE DI ATTACCO (10:15–12:30, 135 minuti)

**Obiettivo**: rompere più immagini possibile, di **tutti** i gruppi, con il WPSNR più alto possibile. Un attacco è riuscito se la detection del gruppo attaccato dice "assente" (`output1 = 0`) **e** il WPSNR tra watermarkata e attaccata è ≥ 35 dB (`output2 ≥ 35`).

Come si fanno punti in attacco (regolamento, p.7–8):

- **attività** = percentuale di gruppi attaccati con successo: > 30% → 2 punti, > 60% → 4, **> 90% → 6**. Con 9 gruppi avversari, 8 su 9 fa l'89%: **serve almeno un'immagine rotta per OGNI gruppo**;
- **qualità** = numero di immagini rotte con WPSNR sopra la media di tutti gli attacchi della gara: 1–5 → 1 punto, 6–10 → 2, 11–15 → 3, **> 15 → 4**;
- **bonus +2** se rompiamo un gruppo che nessun altro rompe.

### Tabella di marcia

| ora | chi | cosa | passo |
|---|---|---|---|
| 10:15 | B | scarica immagini, originali e detection degli altri gruppi | 2.1 |
| 10:15–10:25 | A | sistema i file nelle cartelle con i nomi giusti | 2.1 |
| 10:25 | A | **controllo delle detection** (~1 min) | 2.2 |
| 10:30 | A | **attacco** con `--total-min 100` | 2.3 |
| 10:30–12:10 | B | carica man mano ogni `[BROKEN]` | 2.3 |
| 10:30–12:10 | C | tiene la lista gruppo → immagini rotte → caricate | 2.3 |
| ~12:10 | A | secondo giro sui gruppi senza immagini rotte, se c'è tempo | 2.4 |
| entro 12:25 | B | ultimi upload (non all'ultimo minuto) | — |
| dopo la gara | tutti | log Excel + PNG delle zone | 2.5 |

### 2.1 Preparazione (10:15–10:25)

B scarica dal sito, A mette i file nelle cartelle **con questi nomi** (lo script riconosce gruppo e immagine dal nome del file):

| cartella | contenuto | nome dei file |
|---|---|---|
| `~/ctm/attack/img/` | le immagini watermarkate degli altri gruppi (per esempio 27 = 9 gruppi × 3) | `<gruppo>_<nomeImmagine>.bmp` (il formato del sito, regolamento p.7) |
| `~/ctm/attack/orig/` | le immagini originali | `<nomeImmagine>.bmp` |
| `~/ctm/attack/det/` | le detection degli altri gruppi | `detection_<gruppo>.py` oppure `detection_<gruppo>.pyc` (rinominarle se serve) |

Dettagli importanti:

- `<gruppo>` è tutto quello che sta **prima del primo `_`** nel nome dell'immagine, e deve essere identico al nome nella detection. Esempio: `pixelpirates_buildings.bmp` ↔ `detection_pixelpirates.pyc` ↔ originale `buildings.bmp`.
- Se di un gruppo ci sono sia il `.py` sia il `.pyc`, lo script usa il `.py`.
- Le immagini con il nostro nome (`fakebitinvestigation_...`) vengono saltate da sole: non serve toglierle.
- Se gli originali sono le stesse 3 immagini della difesa, si copiano da lì:
  ```bash
  cp ~/ctm/defense/originals/*.bmp ~/ctm/attack/orig/
  ```
- Controllo veloce dei nomi:
  ```bash
  ls ~/ctm/attack/img ~/ctm/attack/orig ~/ctm/attack/det
  ```

Perché ci servono gli originali: **solo** perché le detection degli altri gruppi li vogliono come `input1`. Noi non li usiamo mai per progettare, scegliere, regolare, localizzare o eseguire un attacco (vietato dal regolamento, p.3 e p.6): lo script li passa e basta, e le zone vengono calcolate solo dalla watermarkata. Usare l'esito della detection degli altri per guidare l'attacco (anche le zone a blocchi) è **ammesso dai TA** (risposte del 19/10 e del 20/10).

Le detection cifrate `.pyc` funzionano solo con la stessa versione di Python con cui sono state compilate (3.10, come richiesto dai TA).

### 2.2 Controllo delle detection (A, ~1 minuto)

```bash
python ../challenge/search_attacks_fakebitinvestigation.py --images ~/ctm/attack/img --originals ~/ctm/attack/orig \
       --detections ~/ctm/attack/det --out ~/ctm/attack/out --check
```

Stampa `check of N images (G groups):` (controllare che N e G siano quelli attesi), poi una riga per immagine:

- `ok, X s per call` → tutto a posto;
- `[skip] <file>: missing <percorso>` (stampato prima dell'elenco) → manca la detection o l'originale di quell'immagine: controllare i nomi del passo 2.1;
- `ERROR: ModuleNotFoundError: …` → manca una libreria: `pip install <libreria>` e rifare il controllo;
- `ERROR: …` (altro) → quella detection è rotta. Lo script la salterà; prenderne nota e, se avanza tempo, guardarla a mano. `bad magic number` vuol dire un `.pyc` compilato con un'altra versione di Python;
- `WARNING: it says 'absent' on the watermarked image` → nome sbagliato (originale e watermarkata non corrispondono?) oppure detection difettosa. Controllare i nomi. Lo script non attacca queste immagini.

Il controllo **non** tocca i log e non salva immagini.

### 2.3 Attacco (A, dalle 10:30 al massimo fino alle ~12:10)

```bash
python ../challenge/search_attacks_fakebitinvestigation.py --images ~/ctm/attack/img --originals ~/ctm/attack/orig \
       --detections ~/ctm/attack/det --out ~/ctm/attack/out --total-min 100
```

**Cosa fa**, per ogni immagine, in parallelo (un processo per immagine, `core − 1` processi):

1. controlla che la loro detection dica "presente" sulla watermarkata non attaccata (se no, non attacca);
2. per ognuna delle 25 famiglie di attacchi (JPEG, blur, AWGN, median, resize, sharpening e loro catene), prima su tutta l'immagine e poi su 7 zone calcolate dalla sola watermarkata, cerca il parametro **più leggero** che rompe, usando la loro detection come oracolo e scartando subito tutto quello che sta sotto i 35 dB (75% del tempo);
3. sul migliore trovato fa il **greedy a blocchi**: rimette la watermarkata nei blocchi 64×64 e poi 32×32 dove l'attacco non serve, finché la loro detection dice ancora "assente" (il restante 25% del tempo);
4. tra tutti gli attacchi riusciti tiene quello con il WPSNR più alto (ricalcolato con il codice dei lab), lo **ricontrolla** con la loro detection e lo salva in `~/ctm/attack/out/` come `fakebitinvestigation_<gruppo>_<nomeImmagine>.bmp`.

**Tempi**:

- `--total-min 100`: lo script si dà 100 minuti in tutto e calcola da solo il tempo per immagine. All'inizio stampa, per esempio (27 immagini, 13 processi), `27 images, 13 processes (3 rounds), budget 31.7 min per image, expected maximum time ~95 min`. **Partendo alle 10:30 finisce entro le 12:10**, con 20 minuti di margine per gli ultimi upload.
- **Se si parte più tardi, abbassare `--total-min`**: regola pratica = minuti che mancano alle 12:10. Per esempio alle 10:50 → `--total-min 80`.
- **Tempo misurato** (simulazione del 19/10: 27 immagini, 9 gruppi × 3, 13 processi, detection da circa 0,6 s, con le zone guidate dalla detection come di default): **32 minuti in tutto**, 18 immagini rotte su 27, almeno una per ognuno dei 9 gruppi, a 35,6–38,0 dB. Le immagini che non si rompono finiscono prima, in circa 7 minuti, perché esauriscono i tentativi. Con detection più lente il tempo cresce, ma il tetto resta `--total-min`.

**Cosa si vede**: una riga per immagine, man mano che finisce (formato):

```
[5/27 14 min] [BROKEN] groupX_buildings: 37.12 dB with Median 3x3, JPEG QF=7 (480 calls, 0 errors, 320 s)
[6/27 15 min] [not broken] groupY_tree (390 calls, 0 errors, 410 s)
[7/27 15 min] [ERROR] groupZ_tree: detection failed 3 times in a row, last: timeout (> 10 s) (3 calls, 3 errors, 31 s)
```

**Upload man mano (B)**:

- ogni `[BROKEN]` ha già l'immagine pronta in `~/ctm/attack/out/` con il nome giusto, `fakebitinvestigation_<gruppo>_<nomeImmagine>.bmp` (regolamento, p.7);
- sul sito, all'upload, va indicato **quale gruppo** è stato attaccato (è il `<gruppo>` del nome);
- lo script stampa **una sola riga per immagine** ed è già il miglior attacco trovato, quindi carichiamo **una sola immagine attaccata per ogni immagine**, come chiede il regolamento ("just upload the case in which you have the highest WPSNR");
- C spunta su una lista: gruppo → immagini rotte → caricate (serve al passo 2.4 e al log).

**Se lo script si ferma** (crash, PC spento, `Ctrl+C` per sbaglio): rilanciare **lo stesso comando** aggiungendo `--resume` e con un `--total-min` più piccolo (i minuti che mancano alle 12:10). Riparte saltando le immagini già nel log. ⚠ Senza `--resume` i due log vengono **riscritti da capo**.

### 2.4 Secondo giro (se avanza tempo)

Alla fine lo script stampa `broken X/27 in Y min; log: ...` e l'elenco delle immagini con errori. Poi:

- **Un gruppo senza nessuna immagine rotta** vale punti di attività (e forse il bonus +2): rilanciare solo su quel gruppo, con più tempo e su **un'altra cartella di uscita**, così il log principale non viene sovrascritto (al posto di `gruppoX` il nome vero, in tutti e due i punti):
  ```bash
  python ../challenge/search_attacks_fakebitinvestigation.py --images ~/ctm/attack/img --originals ~/ctm/attack/orig \
         --detections ~/ctm/attack/det --out ~/ctm/attack/out_gruppoX --only gruppoX --budget 900
  ```
  `--budget 900` = 15 minuti per immagine (le 3 immagini del gruppo vanno in parallelo). Se la loro detection andava in timeout, aggiungere `--timeout 20`.
- Se il secondo giro rompe un'immagine già rotta nel primo con un WPSNR **più alto**, si carica quella nuova (per ogni immagine deve restare solo l'attacco migliore) e nel log si tiene solo quella.
- Le alternative per ogni immagine (tutti gli attacchi riusciti, non solo il migliore) sono in `~/ctm/attack/out/all_successful.csv`.

### 2.5 Dopo la gara: il log

Il regolamento chiede un **log Excel** con le colonne Image | Group | WPSNR | Attack(s) with parameters. È già pronto in `~/ctm/attack/out/attack_log.csv`, con le descrizioni nello stile dell'esempio del regolamento (`JPEG QF=6, Blur sigma=0.8`).

1. Se c'è stato un secondo giro, unire le righe dei vari `attack_log.csv`, tenendo per ogni immagine **solo** quella caricata sul sito.
2. Convertirlo in `.xlsx`: aprirlo con LibreOffice Calc (o Excel) e salvarlo come `.xlsx`, oppure da terminale:
   ```bash
   cd ~/ctm/attack/out
   soffice --headless --convert-to xlsx attack_log.csv
   ```
3. Quando un attacco è stato limitato a zone scelte con la detection, la descrizione cita un file `zones_fakebitinvestigation_<gruppo>_<immagine>.png`, cioè la maschera dei blocchi attaccati (bianco = attaccato, nero = watermarkata lasciata com'era). **Va mandato insieme al log**: senza, l'attacco non è riproducibile. I PNG sono nella stessa cartella `~/ctm/attack/out/`.
4. Salvare tutto anche nella repo (log, PNG, immagini caricate), così il materiale è pronto per la presentazione del 4/11.

---

## 3. Cosa NON fare (regolamento)

| vietato | perché / penalità |
|---|---|
| cambiare un file caricato dopo la fine della fase di difesa | −2 punti |
| cambiare nell'embedding qualcosa che non sia un parametro o una posizione (dopo il 27/10) | è ammesso solo il fine-tuning: "adjust parameters and embedding locations. Other things are forbidden" |
| cambiare un parametro di "posizione" (`R_MIN`, `R_MAX`, `DWT_USE`, …) solo nell'embedding | va cambiato **in entrambi** i codici prima dell'upload (e ricompilato il `.pyc`, e rivista la soglia); meglio non toccarli in gara |
| usare l'originale per progettare, scegliere, regolare, localizzare o eseguire un attacco | lo script lo passa solo alla detection degli altri; le zone vengono dalla sola watermarkata |
| attacchi diversi da AWGN, blur, sharpening, JPEG, resize, median (anche combinati o localizzati) | sono ammessi solo questi; lo script usa solo questi |
| software esterni (Photoshop, GIMP, Paint, …) | vietati |
| caricare più attacchi della stessa immagine | se ne carica uno solo, il migliore |
| server, più di 3 laptop, più di un PC sul sito | vietati |
| detection che stampa, apre finestre o ci mette più di 5 s | −2 punti (la nostra non stampa e impiega meno di 1 s) |
| caricare all'ultimo minuto | il sito può bloccarsi (regolamento, p.2) |

## 4. Problemi frequenti

| problema | soluzione |
|---|---|
| `zsh: not enough directory stack entries` (o simili) incollando un comando | nel comando c'era un commento `# ...`: toglierlo, oppure `setopt interactivecomments` |
| `ModuleNotFoundError` lanciando i nostri script | ambiente non attivo: `source ../../OFF_MDS_ENV/bin/activate` (da `submission/`) |
| `no .bmp image in ...` | cartella sbagliata o vuota: controllare con `ls` |
| `!! CHECK !!` in difesa | non caricare; controllare che `TARGET_BY_IMAGE` sia scritto bene (nomi **senza** `.bmp`, tra apici, separati da virgole) e rifare `--final`; nel frattempo resta valida la versione di sicurezza |
| il WPSNR di un'immagine non è al target scelto | il nome in `TARGET_BY_IMAGE` non corrisponde al file: correggere e rifare `--final` |
| il controllo dà `ModuleNotFoundError` | `pip install` della libreria, poi ripetere il controllo |
| il controllo dà `bad magic number` | quel `.pyc` è compilato con un'altra versione di Python: chiedere al gruppo uno compilato con Python 3.10 (o il `.py`) |
| una detection di un altro gruppo è lentissima | il timeout (10 s) la ferma: dopo 3 errori o timeout di fila l'immagine viene saltata. Alla fine si può riprovare con `--only gruppo --timeout 20` |
| `no image to attack` | nessuna immagine con detection e originale corrispondenti (o tutte già nel log, con `--resume`): controllare i nomi |
| lo script si è fermato | rilanciarlo con `--resume` e meno `--total-min` |
| siamo in ritardo | abbassare `--total-min` |
| siamo molto in ritardo | aggiungere `--no-greedy`: più veloce, WPSNR un po' più basso (circa −0,2/−0,4 dB) |

## 5. Promemoria dei comandi

Tutti da `fakebitinvestigation/submission/`, dopo `source ../../OFF_MDS_ENV/bin/activate`.

| passo | comando |
|---|------------|
| 1.1 sicurezza | `python ../challenge/defense_fakebitinvestigation.py --images ~/ctm/defense/originals --out ~/ctm/defense/safety --final` |
| 1.2 prova target | `python ../challenge/defense_fakebitinvestigation.py --images ~/ctm/defense/originals --out ~/ctm/defense/out` |
| 1.4 finale | `python ../challenge/defense_fakebitinvestigation.py --images ~/ctm/defense/originals --out ~/ctm/defense/out --final` |
| 2.2 controllo | `python ../challenge/search_attacks_fakebitinvestigation.py --images ~/ctm/attack/img --originals ~/ctm/attack/orig --detections ~/ctm/attack/det --out ~/ctm/attack/out --check` |
| 2.3 attacco | `python ../challenge/search_attacks_fakebitinvestigation.py --images ~/ctm/attack/img --originals ~/ctm/attack/orig --detections ~/ctm/attack/det --out ~/ctm/attack/out --total-min 100` |
| 2.3 ripresa | come 2.3, aggiungendo `--resume` e con meno `--total-min` |
| 2.4 secondo giro | come 2.3, ma con `--out ~/ctm/attack/out_gruppoX --only gruppoX --budget 900` al posto di `--out ~/ctm/attack/out --total-min 100` |

Opzioni degli helper (quelle non scritte restano al default):

| script | opzione | significato | default |
|---|---|---|---|
| `defense_` | `--targets` | target di WPSNR da provare | `58.05 62.05` |
| `defense_` | `--budget` | secondi di attacco per immagine e target | `900` |
| `defense_` | `--procs` | processi in parallelo | core − 1 |
| `defense_` | `--final` | produce i file da caricare con il codice attuale | — |
| `search_attacks_` | `--total-min` | minuti totali a disposizione | `100` |
| `search_attacks_` | `--budget` | secondi per immagine (se dato, ignora `--total-min`) | da `--total-min` |
| `search_attacks_` | `--timeout` | secondi massimi per una chiamata alla loro detection | `10` |
| `search_attacks_` | `--only` | attacca solo questo gruppo | tutti |
| `search_attacks_` | `--procs` | processi in parallelo | core − 1 |
| `search_attacks_` | `--no-greedy` | salta il greedy a blocchi | greedy acceso |
| `search_attacks_` | `--resume` | salta le immagini già nel log | — |
| `search_attacks_` | `--check` | solo controllo delle detection | — |
