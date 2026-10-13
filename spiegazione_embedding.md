# Capture the Mark 2026: embedding del gruppo `fakebitinvestigation`

**Bozza v0.1: come funziona il nostro embedding e perché abbiamo fatto ogni scelta**

File collegati:

- `embedding_fakebitinvestigation.py`: la funzione `embedding(input1, input2)` da consegnare
- `test_embedding.py`: script di prova (embedding su una cartella di BMP + estrazione prototipo + attacchi). Non va consegnato come embedding/detection perché stampa a video

> Nomi dei file: seguono il nickname del gruppo, `fakebitinvestigation`, che è lo stesso nome del file del watermark assegnato dai TA (`fakebitinvestigation.npy`). Se il nickname cambia, basta rinominare i file e la costante `GROUP` nello script di test.

---

## 0. Riassunto in 10 punti

1. Lavoriamo nella **DCT globale** dell'immagine 512×512 (la stessa del lab sullo spread spectrum di Cox).
2. Il watermark (1024 bit 0/1) viene trasformato in **±1** (bipolare).
3. Ogni bit viene scritto su una **coppia di coefficienti DCT adiacenti**, (u,v) e (u,v+1): al primo aumentiamo il modulo di `a`, al secondo lo diminuiamo di `a` (il contrario se il bit è −1).
4. Le coppie stanno nella **banda radiale 32 ≤ r ≤ 128** (circa 0,03–0,125 cicli/pixel, cioè strutture con periodo tra 8 e 32 pixel) e sono le 1024 con i coefficienti più grandi dell'immagine originale.
5. L'ampiezza `a` è **uguale per tutte le coppie** e viene calibrata in automatico per ottenere esattamente il WPSNR che vogliamo (per esempio 58,05 o 62,05 dB, appena sopra una soglia della tabella punti).
6. In detection (non-blind) estraiamo il watermark con un **rapporto normalizzato**: così il watermark non risente dei cambi di guadagno (blur, sharpening, resize, contrasto) e sull'immagine originale l'estrazione dà esattamente **0**.
7. Risultati su 43 immagini (le 3 della gara 2022 + 40 estratte a caso tra le 101 di esempio) con 248 attacchi permessi: a **62,05 dB** il watermark viene rotto nel 63% delle immagini, ma solo da attacchi a **37,3 dB di media** (fascia da 6 punti di robustezza); a **58,05 dB** viene rotto solo nel **12%** delle immagini (quasi sempre immagini con molte zone lisce, e quasi sempre a 35–36 dB); a 56 dB mai (su 17 immagini).
8. A parità di WPSNR (60 dB) lo schema del lab così com'è (Cox moltiplicativo) viene rotto in **9 immagini su 9**, in media a **42 dB** (basta un blur con σ = 0,8 o uno sharpening leggero); il nostro in **1 su 9**.
9. Sulle **18 immagini reali watermarkate da altri gruppi nel 2022** la similarità resta tra −0,055 e +0,087: con soglia τ ≈ 0,10 nessun falso positivo.
10. Valore di default nel file: **target 62,05 dB** (5 punti qualità). Nella simulazione dei punti è la scelta che perde al massimo mezzo punto qualunque sia la regola, non scritta, sui gruppi mai attaccati; se i TA ci confermano che chi non viene mai rotto prende la robustezza piena più il bonus, conviene **58,05 dB** (§4.8). In ogni caso il valore va ricontrollato il giorno della gara sulle 3 immagini vere (§10).

---

## 1. Cosa impone il regolamento all'embedding (e come lo rispettiamo)

Riferimento: `1BattleRules2026.pdf` (tra parentesi la pagina).

| Regola (pagina) | Come la rispettiamo nel codice |
|---|---|
| Firma `def embedding(input1, input2): return output1` (p.4) | Esattamente questa firma |
| `input1` = stringa col nome dell'immagine **originale** (p.4) | `cv2.imread(input1, cv2.IMREAD_GRAYSCALE)` |
| `input2` = stringa col nome del **watermark** (p.4) | `np.load(input2)` → 1024 bit 0/1 |
| `output1` = immagine watermarkata (p.4) | array `uint8` 512×512, già arrotondato e saturato in [0,255] |
| Usare **OpenCV** per leggere/scrivere/gestire le immagini, BMP, 512×512 grigi (p.4) | lettura con `cv2`; il salvataggio (nello script di test) con `cv2.imwrite` in BMP |
| Il codice **non deve stampare nulla** né aprire finestre (p.4) | nessuna `print`, nessun `imshow`, nessun file scritto (verificato catturando lo stdout su tutte le 104 immagini: sempre vuoto) |
| Watermark = array NumPy di 1024 zeri e uni (p.7) | il file `fakebitinvestigation.npy` è `int64`, shape (1024,), 521 zeri e 503 uni |
| Fine-tuning dopo il 27/10 = **solo parametri e posizioni di embedding** (p.2) | tutti i parametri sono costanti in testa al file: `TARGET_WPSNR`, `R_MIN`, `R_MAX` |
| Nome immagini consegnate: `groupA_imageName.bmp` (p.7) | lo fa lo script di test: `fakebitinvestigation_<nome>.bmp` |

Regole della **detection** che condizionano già adesso il progetto dell'embedding (p.5, p.8):

| Regola | Conseguenza sul progetto |
|---|---|
| Detection **non-blind**: input = originale, watermarked, attaccata | possiamo usare l'originale per ritrovare le posizioni → le scegliamo in base all'originale |
| Si confronta W estratto dalla watermarked con W estratto dall'attaccata; **vietato** usare/hardcodare il watermark | l'estrazione non ha bisogno né del watermark né dell'ampiezza `a` (la similarità coseno è invariante alla scala) |
| Se si passa l'**originale** come attaccata → watermark **assente** | con la nostra estrazione l'originale dà W = 0 esatto → similarità 0 (dimostrato in §5.4) |
| Penalità se la detection trova il watermark in immagini **non watermarkate**, di **altri gruppi** o **distrutte** (WPSNR ≤ 25) | watermark **bipolare** e a media nulla → su immagini estranee la similarità ha media 0 e deviazione standard 1/32 (§4.2); per le distrutte servirà un controllo sul WPSNR nella detection (§10) |
| Detection entro **5 secondi** | misurato: due estrazioni + similarità ≈ 0,03 s, WPSNR dei lab ≈ 0,7 s → circa 0,8 s in tutto |

Punteggi (p.7–8), cioè quello che vogliamo massimizzare:

| WPSNR embedding | punti qualità |   | WPSNR medio attacchi riusciti su di noi | punti robustezza |
|---|---|---|---|---|
| 35–50 | 1 | | 35–38 | **6** |
| 50–54 | 2 | | 38–41 | 5 |
| 54–58 | 3 | | 41–44 | 4 |
| 58–62 | 4 | | 44–47 | 3 |
| 62–66 | 5 | | 47–50 | 2 |
| ≥ 66 | 6 | | 50–53 | 1 |
| | | | ≥ 53 | 0 |

In più c'è il **bonus +2** se nessuno riesce ad attaccarci (p.8; in fase di verifica il bonus viene tolto se dipende da errori nel codice o nelle immagini). Un attacco è riuscito solo se toglie il watermark **e** ha WPSNR ≥ 35 dB.

> 🗣 **In parole povere**: prendiamo punti in due modi. (1) Il watermark deve "sporcare" poco l'immagine (WPSNR alto). (2) Chi ci attacca deve essere costretto a rovinare parecchio l'immagine per togliere il watermark (WPSNR dell'attacco basso, vicino a 35). Le due cose tirano in direzioni opposte: più energia mettiamo, più il watermark è robusto ma meno è "pulito".

---

## 2. Come si legge la tabella punti (e perché miriamo ai bordi delle fasce)

- I punti qualità **salgono a gradini** ogni 4 dB. Dentro una fascia (es. 58–62) prendiamo sempre 4 punti: quindi conviene stare **appena sopra il bordo inferiore** (58,05), perché a parità di punti è il punto con più energia, cioè più robustezza. Stare a 61,5 non dà nessun punto in più rispetto a 58,05, ma rende il watermark più debole.
- Per questo l'ampiezza non è fissa: il codice **calibra** l'ampiezza per atterrare su `TARGET_WPSNR` (default 62,05; alternativa 58,05, vedi §4.8). Teniamo un margine di 0,05 dB: il WPSNR dei TA è lo stesso codice dei lab, quindi il valore è riproducibile.
- I punti robustezza scendono di 1 ogni 3 dB del WPSNR medio degli attacchi riusciti. Quindi alzare l'embedding di 4 dB (+1 punto qualità) conviene **solo se** gli attaccanti non guadagnano più di circa 3 dB.

---

## 3. Il punto chiave: come il WPSNR "pesa" le frequenze

Il WPSNR dei lab filtra l'errore (immagine A − immagine B) con un filtro CSF (Contrast Sensitivity Function) 41×41 e poi calcola un PSNR. Abbiamo misurato la risposta in frequenza di quel filtro (FFT del kernel portato a 512×512):

| frequenza (cicli/pixel) | 0 | 0,04 | 0,08 | 0,10 | 0,125 | 0,15 | 0,20 | 0,25 | 0,30 | 0,40 |
|---|---|---|---|---|---|---|---|---|---|---|
| \|guadagno CSF\| (lungo gli assi) | 0,500 | 0,509 | 0,491 | 0,453 | 0,373 | 0,271 | 0,092 | ≈0 | 0,016 | 0,003 |

Conseguenze numeriche:

- un errore a **bassa frequenza** "costa" come un PSNR + 6 dB (guadagno 0,5 → MSE diviso 4);
- un **rumore bianco** costa PSNR + 17,7 dB;
- le sottobande di **dettaglio DWT di livello 1** (0,25–0,5 cicli/pixel) costano **37 dB in meno** delle basse frequenze; quelle di livello 2 circa 10 dB in meno.

**Perché questo conta così tanto.** Sembrerebbe un regalo: nascondiamo il watermark nelle alte frequenze, dove il WPSNR quasi non lo vede, e prendiamo 6 punti qualità. Il problema è che **lo stesso regalo lo ricevono gli attaccanti**: blur, median, JPEG e resize cancellano proprio le alte frequenze, e quella cancellazione costa pochissimo WPSNR. Esempio misurato: il bit-plane del lab (che è un rumore quasi bianco, cioè soprattutto alte frequenze) viene tolto da attacchi a **53 dB in media** (§6.2) → 0 punti robustezza.

Il contrario vale per le **basse frequenze**: costano care a noi, ma costano care anche a chi le vuole rovinare, e nessuno degli attacchi permessi le toglie "gratis" (blur, resize, JPEG e median lasciano quasi intatte le basse frequenze). Lì dentro però c'è un'altra trappola (§4.5): è dove il JPEG a qualità bassissima e il rumore "colorato" scaricano il loro rumore.

> 🗣 **In parole povere**: il WPSNR è un "giudice" che guarda soprattutto i dettagli medio-grossi dell'immagine e quasi ignora la grana finissima. Se nascondiamo il watermark nella grana finissima il giudice non lo vede, ma un attaccante può spazzare via la grana finissima e il giudice non vede neanche quello. Bisogna nascondersi dove il giudice guarda, ma dove gli attacchi non riescono a "pulire" senza farsi notare.

---

## 4. Le scelte, una per una, con la loro giustificazione

### 4.1 Dominio: DCT globale (come Cox / lab sullo spread spectrum)

Tecniche viste a lezione e ai lab: LSB / bit-plane, spread spectrum in DCT (Cox), DWT (e DWT+SVD nei progetti degli anni passati).

- **LSB / bit-plane** (lab 3, esercizio 1): invisibile ma fragilissimo, perché attacchi leggerissimi (AWGN con σ = 4–5, JPEG QF 80) riscrivono i bit bassi e lo cancellano a 50–60 dB. Lo usiamo solo come riferimento (tabella §6.2).
- **DWT+SVD** (molto usato dai gruppi degli anni scorsi): funziona, ma nella variante più diffusa richiede di hardcodare U e V della SVD del watermark nella detection (permesso, p.5) ed è più complesso da difendere all'orale. Un dato utile sul livello della gara: il gruppo *howimetyourmark* del 2022 (repo pubblico, voto 30/30), **come attaccante**, ha rotto **il 100% dei gruppi e il 93% delle immagini**, con WPSNR medio degli attacchi riusciti **41,9 dB** (min 36,5, max 59,1). Cioè gli schemi "tipici" degli anni scorsi prendevano 4–5 punti robustezza, e gli attacchi più efficaci erano median (31%), JPEG (26%) e blur (19%).
- **DCT globale**: ogni coefficiente dipende da tutta l'immagine, quindi un bit è "spalmato" su tutti i 262.144 pixel. È la scelta di Cox (*Secure Spread Spectrum Watermarking for Multimedia*, IEEE TIP 1997, il PDF del corso): nascondere il watermark nelle componenti **percettivamente significative**, perché "la maggior parte delle elaborazioni lascia intatte le componenti percettivamente significative" e perché così l'energia si distribuisce uniformemente nei pixel.
- **DCT a blocchi** (64×64 e 32×32, stessa idea ma locale, così il watermark finisce nelle zone testurizzate): l'abbiamo provata (tabella in §4.5). Stessa robustezza media, ma più sensibile ai resize che "stirano" leggermente l'immagine e al median 11. Scartata.

### 4.2 Watermark bipolare: 0/1 → −1/+1

`w = 2·bit − 1`.

**Perché.** La similarità usata nei lab è il coseno: `sim(X, Y) = X·Y / (‖X‖‖Y‖)`.

- Con vettori **0/1**, due watermark **casuali e indipendenti** hanno coseno ≈ **0,5** (prodotto scalare ≈ n/4, norme ≈ √(n/2)). Quindi due marchi che non c'entrano niente sembrano "mezzi uguali": H0 e H1 si avvicinano e la soglia diventa delicata.
- Con vettori **±1** il coseno tra il nostro W (idealmente W = a·w) e **qualunque** vettore indipendente dal nostro watermark ha media 0 e deviazione standard esattamente **1/√1024 = 1/32 ≈ 0,031**; in più la coda è limitata (disuguaglianza di Hoeffding): `P(sim > t) ≤ exp(−1024·t²/2)`, cioè ≤ 0,6% per t = 0,10 e ≤ 0,06% per t = 0,12, **qualunque cosa abbiano fatto gli altri gruppi** sull'immagine.

**Verifica sperimentale.**

- 20.000 watermark ±1 casuali contro il nostro W estratto: deviazione standard 0,0309, quantile 99,9% = 0,094, massimo = 0,115.
- **18 immagini reali watermarkate da 6 gruppi del 2022** (stesse immagini originali *buildings*, *rollercoaster*, *tree*): similarità tra −0,055 e **+0,087** (media −0,004). Con τ = 0,10 nessun falso positivo. Con τ = 0,08 ce ne sarebbe stato uno, quindi la soglia non va scesa sotto circa 0,10 (la fisseremo con la ROC).

> 🗣 **In parole povere**: con 0/1 tutti i watermark "si somigliano un po'" perché sono tutti fatti di numeri positivi. Con ±1 due watermark diversi si annullano a vicenda in media, quindi la nostra detection dice "presente" solo se c'è davvero il nostro.

### 4.3 Coppie differenziali di coefficienti (la modifica più importante)

**Il problema dello schema del lab.** Nel lab (Cox moltiplicativo) si estrae `W_i = (|X_i| − |C_i|) / (α·|C_i|)`, dove C è l'originale e X l'immagine da testare. Supponiamo che un attacco moltiplichi quei coefficienti per un guadagno g vicino a 1 (lo fanno blur, sharpening, resize, cambi di contrasto):

`W_i = g·w_i + (g − 1)/α`

Con α = 0,01–0,02 (quello che serve per avere WPSNR ≈ 54–60), basta g = 0,95 per avere `(g−1)/α = −2,5…−5`: un termine enorme che **non c'entra con il watermark** e annega la parte utile. Infatti nei test lo schema del lab a WPSNR ≈ 54 dB viene rotto da uno **sharpening con σ grande (10–20) e α = 0,2** a soli 35,3–36,4 dB: con σ = 10 quello sharpening moltiplica le frequenze r = 10, 20, 32, 64 per g = 1,03 / 1,11 / 1,17 / 1,20, cioè proprio un guadagno di qualche percento sui coefficienti usati dal lab. (E siccome g cambia con la frequenza, non basterebbe nemmeno togliere la media a W.)

**La soluzione: due coefficienti per bit, con segni opposti.** Prendiamo due coefficienti vicinissimi in frequenza, (u,v) e (u,v+1). Qualunque filtro "liscio" (blur, sharpening, resize, filtri anisotropi tipo median 1×7) li moltiplica praticamente per lo **stesso** g. Scriviamo il bit così:

- modulo del primo: `|C₁| + a·w`
- modulo del secondo: `|C₂| − a·w`

In estrazione facciamo la **differenza** delle variazioni relative: il termine `(g − 1)` è uguale nei due e **si cancella**; resta solo `g·a·w`, e il coseno è insensibile al fattore g. Dimostrazione completa in §5.4.

**Verifica.** Prima prova (4 immagini, 105 attacchi singoli, α = 0,02, τ = 0,12): la versione del lab (WPSNR medio 54,1 dB) viene rotta in 3 immagini su 4, sempre dallo sharpening con σ = 10–20 e α = 0,2 (35,3–36,4 dB); la versione a coppie, alla stessa qualità (54,7 dB), in **nessuna**. Il confronto completo a parità di WPSNR è in §6.2.

> 🗣 **In parole povere**: invece di dire "questo coefficiente è più alto del normale = 1", diciamo "questo coefficiente è più alto del suo vicino = 1". Se un attacco alza o abbassa tutti e due insieme (come fanno blur e sharpening), il confronto tra i due non cambia.

### 4.4 Ampiezza additiva uniforme + estrazione normalizzata (invece che moltiplicativa)

Nello schema moltiplicativo la modifica è proporzionale al coefficiente (`α·|C|`): l'energia va quasi tutta sui coefficienti più grandi. Ma nell'estrazione a rapporto il rumore di un attacco viene diviso per `|C|`, quindi **pesa di più sui coefficienti piccoli**. Risultato: si spende energia (cioè WPSNR) dove serve meno e si è deboli dove il rumore entra di più. Nei test la versione "Cox a coppie" moltiplicativa a circa 60 dB veniva rotta in tutte e 6 le immagini di una delle prime prove (157 attacchi, τ = 0,12), anche da AWGN con σ = 20–25 (σ = 25 da solo, oppure σ = 20 seguito da un blur: il WPSNR li considera ancora ≥ 35 dB!), JPEG QF 10 e median 1×7.

Abbiamo quindi ricavato la combinazione ottima (per rumore additivo e mantenendo l'invarianza al guadagno):

- **embedding**: stessa ampiezza `a` (in valore assoluto) su ogni coefficiente scelto;
- **estrazione**: `W_i = [(X₁/C₁ − 1) − (X₂/C₂ − 1)] / (1/|C₁| + 1/|C₂|)`.

Con questa normalizzazione ogni bit ha **lo stesso rapporto segnale/rumore**, il massimo possibile per l'energia spesa (energia per bit = 2a², varianza del rumore per bit tra σ²/2 e σ²). E l'invarianza al guadagno resta esatta.

**Verifica** (9 immagini, 157 attacchi, τ = 0,10, tutti calibrati a WPSNR 60): moltiplicativo a coppie → rotto **7/9**, media degli attacchi riusciti 42,0 dB; additivo uniforme (stessa banda bassa) → rotto **3/9**, media 37,6 dB. Tabella completa in §6.2.

> 🗣 **In parole povere**: la detection guarda tutti i 1024 bit insieme; se alcuni bit sono "rumorosi" rovinano il voto di tutti. Conviene che tutti i bit siano ugualmente affidabili, e per farlo ogni coppia riceve la stessa spinta assoluta.

### 4.5 La banda di frequenze: 32 ≤ r ≤ 128

Per ogni coefficiente DCT (u,v) di un'immagine 512×512 la frequenza spaziale è `r/1024` cicli/pixel con `r = √(u² + v²)`. Quindi r = 32 → periodo 32 pixel, r = 128 → periodo 8 pixel. La banda scelta copre "strutture" grandi tra 8 e 32 pixel.

**Come l'abbiamo scelta: misurando, non a intuito.** Per i 7 attacchi più pericolosi abbiamo misurato, banda per banda, quanto segnale sopravvive (guadagno g) e quanto rumore entra (n, potenza per coefficiente dopo aver tolto il guadagno), su 12 immagini a caso del set di esempio. Il numero che conta per noi è `g²/n` (più alto = il watermark sopravvive meglio per unità di energia). Abbiamo anche segnato il "costo" WPSNR di ogni banda (le frequenze sopra r ≈ 82 costano meno):

| banda r | [1,16) | [16,32) | [32,64) | [64,128) | [128,200) |
|---|---|---|---|---|---|
| costo WPSNR relativo | 1,00 | 1,02 | 1,04 | 0,84 | 0,29 |
| JPEG QF 5 (34,7 dB) | **0,21** | 0,44 | 0,72 | 1,27 | 1,39 |
| JPEG QF 10 (39,5 dB) | 1,12 | 1,77 | 2,58 | 4,26 | 4,56 |
| Median 7×7 (34,1 dB) | 0,46 | 1,23 | 1,81 | 1,40 | **0,30** |
| Resize 0,3 con cv2 (37,1 dB) | 288 | 171 | 43,9 | 5,2 | 0,97 |
| AWGN σ=25 (35,5 dB) | 0,85 | 1,45 | 1,57 | 1,61 | 1,66 |
| AWGN σ=30 + Blur σ=2 (32,9 dB) | 0,51 | 1,04 | 1,04 | 0,95 | 0,89 |
| Blur σ=1,5 (37,0 dB) | 105 | 112 | 36,5 | 12,1 | 11,8 |

(valori ×1000; tra parentesi il WPSNR medio dell'attacco)

Lettura della tabella:

- **Le frequenze bassissime (r < 16) sono la zona peggiore**, contrariamente a quello che ci si aspetterebbe: lì cade il **rumore di blocking del JPEG** (le medie dei blocchi 8×8 quantizzate sono una "scalinata" a bassa frequenza) e il rumore "colorato" AWGN+blur. È uno dei motivi per cui lo schema del lab soffre il JPEG a QF bassa: prende i 1024 coefficienti più grandi in assoluto, e di questi (misurato su 14 immagini) il 22% sta sotto r = 20 e metà sotto r ≈ 35, comprese le frequenze più basse in assoluto.
- **Le frequenze alte (r > 128)** reggono bene il JPEG ma vengono fortemente attenuate da median e resize, e diventano sensibili ai resize "non esatti" (sotto).
- Se si guarda solo questa tabella, dividendo `g²/n` per il costo WPSNR della banda, il caso peggiore tra gli attacchi è più alto in [64,128) (1,13, AWGN+blur), poi in [128,200) (1,03, median 7×7), mentre [32,64) si ferma a 0,69 (JPEG QF 5) e le bande sotto r = 32 sono molto peggio. La tabella però non contiene il resize "stirato" del lab, che penalizza proprio le frequenze alte: per questo la scelta finale l'abbiamo fatta con la verifica completa qui sotto, che con tutti gli attacchi premia la fascia **32–128** rispetto a 48–128 e 64–128.

**Il problema dei resize "non esatti".** La funzione `resizing` dei lab fa `rescale(img, s)` e poi `rescale(…, 1/s)` e taglia a `[:512, :512]`. Per alcune scale l'immagine non torna a 512 px: con 0,55, 0,3 e 0,15 (tra quelle provate) esce 513×513 (es. 0,55 → 282 px → 513 px), quindi l'immagine è **stirata dello 0,2%** e tagliata; con 0,7, 0,45, 0,35 e 0,2 esce 510–511 px e il codice del lab da solo andrebbe in errore sul WPSNR (nella nostra batteria in quel caso replichiamo l'ultima riga e colonna, che è di nuovo uno stiramento). Uno stiramento sposta un po' le frequenze; per un coefficiente a frequenza r lo spostamento è ∝ r, quindi le coppie a r alto si "desincronizzano". È un attacco legale (è il codice del lab) che gli attaccanti useranno sicuramente; per questo non siamo saliti oltre r = 128.

**Verifica (banda per banda, 17 immagini, 248 attacchi, target 62 dB, τ = 0,10):**

| banda | immagini rotte | WPSNR medio attacchi riusciti | massimo |
|---|---|---|---|
| r ≥ 8 (senza limite superiore: in pratica coefficienti bassi) | 12/17 | 39,7 | 43,3 |
| 16–96 | 10/17 | 39,4 | 42,1 |
| 24–128 | 10/17 | 38,5 | 41,0 |
| **32–128 (scelta)** | **10/17** | **37,4** | 41,6 |
| DCT a blocchi 64×64, r 4–16 | 10/17 | 37,9 | 44,9 |
| DCT a blocchi 32×32, r 2–8 | 11/17 | 37,2 | 40,1 |
| DCT a blocchi 64×64, r 3–12 | 11/17 | 37,5 | 41,7 |
| 48–128 | 10/17 | 37,8 | 40,0 |
| 64–128 | 11/17 | 38,2 | 42,0 |

Con soglia τ = 0,08 la differenza è ancora più netta: 32–128 → 4/17 immagini rotte; 48–128, 64–128, 24–128 e 16–96 → 8/17; r ≥ 8 → 10/17 (solo la variante a blocchi 64×64 con r 3–12 arriva anche lei a 4/17).

### 4.6 Quali coefficienti dentro la banda: i 1024 più "forti"

Tra tutte le coppie candidate nella banda (v pari, coppia (u,v)–(u,v+1)), prendiamo le 1024 con il **minimo dei due moduli più grande**.

Perché:

1. **Stabilità dell'estrazione**: dividiamo per C₁ e C₂; coefficienti grandi = divisione stabile, nessun rischio di dividere per ~0 (sulle 104 immagini il più piccolo dei coefficienti scelti vale 13,2, la mediana 55,8).
2. **Significatività percettiva** (argomento di Cox): i coefficienti grandi sono quelli che "tengono in piedi" l'immagine; per cancellarli un attaccante deve rovinare l'immagine.
3. **Riproducibilità**: la scelta dipende **solo dall'originale**, che la detection riceve (non-blind), quindi la detection ritrova le stesse posizioni senza salvare nulla.
4. **Ordine per posizione**: dopo la selezione le coppie vengono ordinate per (u,v), non per ampiezza. Così l'assegnazione bit→coppia non dipende da differenze di ultima cifra decimale tra PC diversi (in ogni caso la detection estrae W_ext e W_att con la stessa lista, quindi anche uno scambio non cambierebbe la similarità).

Una sola coppia per bit: abbiamo provato 2 coppie per bit (2048 coppie, bit mediati) e non c'è **nessun miglioramento** (11/17 rotte invece di 10/17 a 62 dB). È coerente con la teoria: contro il rumore conta solo l'energia totale, non su quanti coefficienti è spalmata.

### 4.7 Calibrazione automatica dell'ampiezza

Il WPSNR scende quando `a` cresce, quindi facciamo una **bisezione** (in scala logaritmica, 14 passi tra 0,2 e 40) e teniamo la **più grande** `a` che rispetta `WPSNR ≥ TARGET_WPSNR`. Il WPSNR è calcolato con **il codice esatto dei lab** (incluso nel file), sull'immagine già arrotondata a interi: è esattamente il numero che calcoleranno i TA.

Dettagli osservati:

- Il WPSNR dipende quasi solo da `a` (tutte le coppie stanno dove il CSF è circa piatto): a 62 dB serve a ≈ 4,2 su tutte le immagini, a 58 dB a ≈ 6,5.
- **L'arrotondamento a interi è un effetto grosso**: a 62 dB il pattern del watermark nei pixel vale circa ±0,4 livelli di grigio, quindi la maggior parte dei pixel arrotonda allo stesso valore. Sull'immagine *tree*: **82% dei pixel invariati**, il resto cambia di ±1 (solo 57 pixel su 262.144 di ±2). Il watermark resta comunque perfettamente leggibile (correlazione con il watermark vero 0,99) perché l'errore di arrotondamento è sparso su tutte le frequenze, mentre il watermark è concentrato in 2048 coefficienti.
- Controllo su tutte le 104 immagini che abbiamo (101 di esempio + 3 della gara 2022), con target 62,05: WPSNR ottenuto sempre tra 62,0500 e 62,0545 dB, pixel cambiati tra il 16,7% e il 18,1%, mai più di ±2 livelli, correlazione col watermark vero tra 0,978 e 0,992.
- Tempo: circa 10–12 s per immagine (la bisezione chiama il WPSNR dei lab 14 volte; sulle 104 immagini, con due processi in parallelo, media 12 s e massimo 18,5 s). Sull'embedding non c'è nessun limite di tempo.

### 4.8 Quanto forte? Il target di WPSNR

Curva di compromesso misurata (banda 32–128, 17 immagini, 248 attacchi):

| target embedding | punti qualità | rotte (τ=0,10) | WPSNR medio attacchi riusciti | max | rotte (τ=0,08) |
|---|---|---|---|---|---|
| 56 dB | 3 | **0/17** | — | — | 0/17 |
| 58 dB | 4 | **1/17** | 38,4 | 38,4 | 1/17 |
| 60 dB | 4 | 2/17 | 37,3 | 39,6 | 1/17 |
| 62 dB | 5 | 10/17 | 37,4 | 41,6 | 4/17 |
| 64 dB | 5 | 15/17 | 38,5 | 43,1 | 12/17 |
| 66 dB | 6 | 16/17 | 40,7 | 45,2 | 16/17 |

Conferma su un campione più grande e **non scelto a mano** (3 immagini 2022 + 40 immagini estratte a caso, con seme fisso, dalle 101 di esempio), 248 attacchi:

| target | rotte τ=0,08 | rotte τ=0,10 | rotte τ=0,12 | WPSNR medio attacchi riusciti (τ=0,10) | mediana | max |
|---|---|---|---|---|---|---|
| **58,05 dB** | 2/43 (5%) | **5/43 (12%)** | 8/43 (19%) | 36,1 | 35,7 | 38,4 |
| **62,05 dB** | 19/43 (44%) | **27/43 (63%)** | 34/43 (79%) | 37,3 | 36,9 | 41,6 |

Le 3 immagini della gara 2022 (similarità più bassa ottenuta con un attacco ≥ 35 dB): a 58,05 dB *buildings* 0,163, *rollercoaster* 0,196, *tree* 0,321 (nessuna rotta); a 62,05 dB 0,093 / 0,131 / 0,189 (rotta solo *buildings*, e solo con τ ≥ 0,10).

**Punti attesi** (simulazione Monte Carlo): estraiamo a caso 3 immagini dal campione (20.000 volte), guardiamo se e con quale WPSNR vengono rotte e applichiamo le tabelle di p.7. Il regolamento **non dice** quanti punti robustezza prende chi non viene mai attaccato con successo, quindi consideriamo due casi:
- **caso A**: mai rotti ⇒ robustezza massima (6) + bonus 2 (secondo noi l'interpretazione più logica);
- **caso B**: mai rotti ⇒ 0 robustezza + bonus 2.

| target | P(rotta) per immagine | P(nessuna delle 3 rotta) | punti attesi caso A | punti attesi caso B |
|---|---|---|---|---|
| 56 dB (17 img) | 0,00 | 1,00 | 11,0 | 5,0 |
| **58,05 dB** (43 img) | 0,12 | 0,69 | **11,3** | 7,2 |
| 60 dB (17 img) | 0,12 | 0,68 | 11,2 | 7,1 |
| **62,05 dB** (43 img) | 0,63 | 0,05 | 10,8 | 10,5 |
| 64 dB (17 img) | 0,88 | 0,00 | 10,4 | 10,4 |
| 66 dB (17 img) | 0,94 | 0,00 | 10,6 | **10,6** |

**Decisione per la bozza: `TARGET_WPSNR = 62.05`.**

- Nel caso A il migliore è 58,05 (11,3), ma 62,05 perde solo 0,5 punti.
- Nel caso B il migliore è 66 (10,6), e 62,05 perde solo 0,1 punti; 58,05 invece crollerebbe a 7,2.
- Quindi 62,05 è la scelta "a rimpianto minimo" finché non conosciamo la regola. **Se i TA confermano il caso A, passiamo a 58,05.**
- Attenzione: gli attaccanti veri saranno un po' più forti della nostra batteria (ricerche più fini, combinazioni di 3 attacchi). Questo riduce soprattutto il vantaggio di 58,05 nel caso A (con probabilità di rottura 25% invece di 12% scende a circa 10,8 punti, cioè pari a 62,05).

> 🗣 **In parole povere**: fra 60 e 62 dB c'è un "gradino": a 58 quasi nessun attacco valido ci rompe, a 62 ci rompono spesso ma solo attacchi pesanti (≈37 dB, che ci danno comunque 6 punti robustezza). Quale convenga dipende da una regola che il PDF non scrive: va chiesta ai TA.

---

## 5. L'algoritmo passo per passo (con la matematica)

### 5.1 Notazione

- `I` immagine originale 512×512, `C = DCT2(I)` (DCT-II ortonormale, la stessa dei lab: `dct(dct(I, axis=0, norm='ortho'), axis=1, norm='ortho')`). Essendo ortonormale vale Parseval: l'energia nei pixel è uguale all'energia nei coefficienti.
- `b ∈ {0,1}¹⁰²⁴` il watermark, `w = 2b − 1 ∈ {−1,+1}¹⁰²⁴`.
- Coppia i: `p_i1 = (u_i, v_i)`, `p_i2 = (u_i, v_i + 1)`, con `v_i` pari.

### 5.2 Embedding

1. Leggi `I` (cv2, grigi) e `b` (np.load). Calcola `w = 2b − 1`.
2. `C = DCT2(I)`.
3. Coppie candidate: tutte le (u, v) con v pari e `R_MIN ≤ √(u² + (v + ½)²) ≤ R_MAX` (il +½ è il centro della coppia).
4. Forza di una coppia: `s = min(|C(u,v)|, |C(u,v+1)|)`. Tieni le 1024 coppie con `s` più grande; ordinale per posizione (u, v). Bit i ↔ coppia i.
5. Per un'ampiezza `a`:
   - `C'(p_i1) = C(p_i1) + a · sgn(C(p_i1)) · w_i`  → il modulo cresce di `a·w_i`
   - `C'(p_i2) = C(p_i2) − a · sgn(C(p_i2)) · w_i`  → il modulo cala di `a·w_i`
   - tutti gli altri coefficienti restano uguali;
   - `I_w = clip(round(IDCT2(C')), 0, 255)` in `uint8`.
6. Bisezione su `a` → la più grande con `WPSNR(I, I_w) ≥ TARGET_WPSNR`. Restituisci `I_w`.

### 5.3 Estrazione (servirà identica nella detection)

Dati l'originale `I` e un'immagine qualunque `J` (watermarked o attaccata):

1. `C = DCT2(I)`, stesse 1024 coppie del passo 4 (dipendono solo da `I`).
2. `X = DCT2(J)`.
3. `r_ij = X(p_ij) / C(p_ij) − 1` (variazione relativa con segno).
4. `W_i = (r_i1 − r_i2) / (1/|C(p_i1)| + 1/|C(p_i2)|)`.

Detection (prossimo passo): `W_ext = estrai(I, I_w)`, `W_att = estrai(I, J)`, watermark presente se `sim(W_ext, W_att) ≥ τ`.

### 5.4 Le proprietà (dimostrazioni brevi)

Scriviamo `s_j = sgn(C_j)`, così `C_j = s_j |C_j|`, e osserviamo che `s_j·(X_j − C_j)/|C_j| = (X_j − C_j)/C_j = r_j`.

**(a) Sull'immagine watermarkata** (ignorando l'arrotondamento): `X₁ = C₁ + s₁·a·w`, `X₂ = C₂ − s₂·a·w`.
`r₁ = a·w/|C₁|`, `r₂ = −a·w/|C₂|` ⇒ `W = a·w·(1/|C₁| + 1/|C₂|)/(1/|C₁| + 1/|C₂|) = a·w`. ✔ (in pratica, con l'arrotondamento, correlazione 0,98–0,99 col watermark vero)

**(b) Sull'immagine originale**: `X = C` ⇒ `r = 0` ⇒ `W = 0` esatto ⇒ la similarità vale 0 (la definiamo 0 quando un vettore è nullo) ⇒ "watermark assente", come richiesto a p.5.

**(c) Guadagno uguale sulla coppia** (blur, sharpening, resize, contrasto): `X_j = g·C'_j`.
`r₁ = g(1 + a·w/|C₁|) − 1`, `r₂ = g(1 − a·w/|C₂|) − 1` ⇒ `r₁ − r₂ = g·a·w·(1/|C₁| + 1/|C₂|)` ⇒ `W = g·a·w`.
Il termine `(g − 1)` è sparito; il coseno non vede il fattore `g` ⇒ similarità invariata. ✔
(Con lo schema del lab invece resterebbe `(g−1)/α`, enorme.)

**(d) Rumore additivo** `n_j` di varianza σ² per coefficiente:
`W = a·w + (s₁n₁/|C₁| − s₂n₂/|C₂|)/(1/|C₁| + 1/|C₂|)`, varianza del rumore `σ²·(1/C₁² + 1/C₂²)/(1/|C₁| + 1/|C₂|)²` compresa tra σ²/2 (moduli uguali) e σ². Energia spesa per bit `2a²`. Similarità attesa ≈ `1/√(1 + σ²_eff/a²)`.

**(e) Immagini estranee** (originale + watermark di un altro gruppo, o immagini qualsiasi): `W_att` non dipende dal nostro `w`; poiché `W_ext ≈ a·w` con `w` casuale ±1, `sim` ha media 0 e varianza 1/1024 (§4.2).

---

## 6. Risultati sperimentali

### 6.1 Setup

- **Immagini**: le **101 immagini di esempio** (0000–0100.bmp, 512×512 grigi, foto naturali), cioè quelle che nel 2022 venivano usate per la ROC, prese dal repository pubblico di un gruppo di quell'anno; le **3 immagini della gara 2022** (*buildings*, *rollercoaster*, *tree*) e le **18 versioni watermarkate da 6 gruppi del 2022** (per i falsi positivi).
- **WPSNR**: codice dei lab. Per velocizzare gli esperimenti abbiamo usato una versione via FFT matematicamente identica (stessa convoluzione circolare `boundary='wrap'`): differenza 0,000000 dB in tutti i confronti fatti. Il file consegnato usa il codice originale.
- **Batteria d'attacco (248 attacchi, solo quelli permessi a p.6, uscita salvata come `uint8` come farebbe un vero BMP)**:
  - AWGN σ = 2…30; blur σ = 0,4…3 (anche anisotropo); sharpening σ ∈ {0,5…20} × α ∈ {0,2…3}; median 3…11 (anche 1×3, 3×1, 1×7, 5×7…);
  - resize: versione del lab (skimage, con il taglio `[:512,:512]`; quando esce 510–511 px replichiamo l'ultima riga/colonna, altrimenti il WPSNR non si potrebbe calcolare) e versione "esatta" cv2 (INTER_AREA + CUBIC/LINEAR), scale 0,15…0,95;
  - JPEG QF 2…95;
  - combinazioni: AWGN+blur (rumore colorato), JPEG+blur, median+JPEG, resize+JPEG, blur+sharpen, AWGN+median;
  - **attacchi localizzati**: JPEG/median/AWGN solo sulle zone lisce, AWGN/blur/median/JPEG solo sulle zone testurizzate.
- **Attacco riuscito** (come da regolamento): `sim < τ` **e** WPSNR(watermarked, attaccata) ≥ 35 dB. Per ogni immagine registriamo l'attacco riuscito con il WPSNR più alto (quello che un attaccante caricherebbe).
- **τ**: la soglia vera la calcoleremo con la ROC (detection). Qui riportiamo τ = 0,10 (FPR ≈ 0,07% sull'H0 teorica) e, dove utile, 0,08 e 0,12.

### 6.2 Confronto tra schemi a parità di WPSNR (60 dB)

9 immagini (3 della gara 2022 + 6 di esempio), 157 attacchi (batteria senza gli attacchi localizzati), τ = 0,10. Tutti gli schemi a spread spectrum sono calibrati a WPSNR 60,0 dB; il bit-plane non è calibrabile (il WPSNR dipende solo dal bit usato).

| schema | WPSNR embedding | immagini rotte | WPSNR medio attacchi riusciti | max | attacchi che rompono |
|---|---|---|---|---|---|
| Bit-plane 2 (lab 3, es.1) | 51,2 | 9/9 | 53,5 | 60,8 | AWGN σ=4–5, JPEG QF 80 |
| Cox del lab (moltiplicativo, 1024 coeff. più grandi) | 60,0 | 9/9 | 42,0 | 51,0 | blur σ=0,8, sharpening σ=10 α=0,2 |
| Cox a coppie (moltiplicativo) | 60,0 | 7/9 | 42,0 | 48,0 | AWGN+blur, JPEG QF 10 |
| Coppie, ampiezza uniforme, banda r ≥ 1 | 60,0 | 3/9 | 37,6 | 40,1 | JPEG QF 5, AWGN+blur |
| **Coppie, ampiezza uniforme, banda 32–128 (scelto)** | 60,0 | **1/9** | 38,4 | 38,4 | JPEG QF 5 |

Tradotto in punti robustezza (solo su queste 9 immagini): bit-plane 0, Cox del lab 4, Cox a coppie 4; nella nostra versione una sola immagine viene rotta (a 38,4 dB → 5 punti) e nelle altre 8 nessun attacco valido.

Dalla seconda riga in poi ogni riga aggiunge una sola modifica alla precedente, così si vede quanto vale ciascuna (coppie → invarianza al guadagno; ampiezza uniforme → SNR per bit; banda → via dal rumore del JPEG).

### 6.3 Altri risultati

- Banda e varianti a blocchi: tabella in §4.5.
- Ridondanza 2 coppie/bit: nessun guadagno (§4.6).
- Falsi positivi su immagini di altri gruppi: §4.2.
- Invisibilità: a 62 dB su *tree* l'82% dei pixel è identico all'originale, il resto differisce di ±1 (57 pixel di ±2). A occhio originale e watermarkata sono indistinguibili; la differenza amplificata è una trama uniforme di "macchioline" senza strutture.

### 6.4 Quali attacchi ci rompono (e dove)

Il pattern (campione da 43 immagini):

- **Le rotture "care" per noi (attacco ≥ 38 dB) avvengono quasi tutte su immagini con molte zone lisce** (a 62,05 dB, 8 su 10; foschia di montagna, muri bianchi, sfondi sfocati, cielo/acqua): lì un attacco pesantissimo costa poco WPSNR perché non c'è dettaglio da rovinare. L'attacco tipico è il **JPEG con QF 4–9**, globale o applicato **solo sulle zone lisce** (il più insidioso: il blocking mette rumore proprio nella nostra banda e nelle zone lisce costa poco); oltre al JPEG compare anche resize 0,3 + JPEG (36,0–38,0 dB). Caso peggiore: 0070.bmp (81% di zone lisce), rotta a 38,4 dB anche a 58 dB.
- **Sulle immagini testurizzate** (a 62,05 dB e τ = 0,10 ne vengono rotte 9) l'attacco che funziona è soprattutto il **resize del lab con stiramento** (scala 0,55: 4 volte, a 35,1–36,1 dB e una volta a 39,4 dB); poi JPEG QF 6 globale (2 volte, 35,3–35,6 dB), JPEG applicato solo alle poche zone lisce (2 volte, 37,7–38,1 dB) e rumore colorato AWGN+blur (una volta, 35,1 dB). Il median 9–11 compare solo con la soglia più alta (τ = 0,12).
- Immagini molto testurizzate (*tree*, *0007*…): nessun attacco ≥ 35 dB ci rompe nemmeno a 62 dB. *buildings* a 62 dB cede solo a JPEG QF 6 a 35,6 dB (al limite della validità → 6 punti robustezza); a 58 dB non cede.

Prossimo miglioramento naturale (fine-tuning della detection, non dell'embedding): rendere l'estrazione più tollerante allo stiramento del resize.

---

## 7. Conformità al regolamento: checklist del file `embedding_fakebitinvestigation.py`

- [x] `def embedding(input1, input2): … return output1`
- [x] `input1`, `input2` usati come **percorsi stringa**; lettura immagine con `cv2.imread(..., cv2.IMREAD_GRAYSCALE)`
- [x] output `uint8` 512×512, valori in [0,255]
- [x] nessuna `print`, nessuna finestra, nessun file scritto (verificato in una cartella pulita: stdout e stderr vuoti, nessun file nuovo)
- [x] nessuna dipendenza da file esterni oltre ai due input (il WPSNR dei lab è incluso nel file)
- [x] determinismo: stessa immagine + stesso watermark ⇒ stessa uscita (nessun numero casuale)
- [x] calibrazione: su tutte le 104 immagini disponibili il WPSNR ottenuto è tra 62,0500 e 62,0545 dB (target 62,05)
- [x] parametri modificabili nel fine-tuning raccolti in testa al file
- [x] librerie: `cv2`, `numpy`, `scipy` (già usate nei lab)

---

## 8. Come usarlo

```python
import cv2
from embedding_fakebitinvestigation import embedding

wm = embedding('lena.bmp', 'fakebitinvestigation.npy')
cv2.imwrite('fakebitinvestigation_lena.bmp', wm)
```

Test completo su una cartella di BMP (salva le immagini in `watermarked/` e stampa WPSNR, controlli e attacchi):

```bash
python test_embedding.py cartella_immagini fakebitinvestigation.npy 0.10
```

Parametri (in testa al file):

| costante | default | cosa fa |
|---|---|---|
| `TARGET_WPSNR` | 62.05 | WPSNR a cui si calibra l'embedding. Valori sensati: 58.05 (4 punti), 62.05 (5 punti), 66.05 (6 punti); vedi §4.8 |
| `R_MIN`, `R_MAX` | 32, 128 | banda radiale dei coefficienti DCT (= "posizioni di embedding") |
| `N_BITS` | 1024 | lunghezza del watermark |
| `AMP_MIN`, `AMP_MAX`, `N_ITER` | 0.2, 40, 14 | intervallo e passi della bisezione |

---

## 9. Punti aperti e rischi

1. **Da chiedere ai TA: robustezza se nessuno ci attacca.** La tabella dà i punti in base al WPSNR medio degli attacchi riusciti. Se non ce ne sono, ci aspettiamo che si prenda il massimo (6) + il bonus 2, ma non è scritto. Se invece in quel caso si prendesse 0 + 2, la scelta cambierebbe radicalmente verso 62 o 66 dB.
2. **Da chiedere ai TA: punti qualità per immagine o media delle 3?** Se è la media, si può dare più energia all'immagine più liscia (la più vulnerabile) e meno alle altre, restando nella stessa fascia in media. Tecnicamente è solo un parametro per immagine.
3. **Immagini lisce**: sono il nostro punto debole. Se le 3 immagini della gara sono molto piatte conviene scendere di 1–2 dB.
4. **La batteria non è esaustiva**: gli attaccanti veri cercheranno parametri più fini e combinazioni di 3 attacchi. Mettiamo in conto 1–2 dB di margine.
5. **Soglia τ**: tutta la robustezza dipende da τ. Più bassa = più robusti ma più falsi positivi (penalità). Dai dati, τ ≈ 0,10 è il compromesso; la fissiamo con la ROC.
6. **Saturazione**: in immagini con grandi zone a 0 o 255 una parte del watermark viene tagliata dal `clip`. La calibrazione tiene conto dell'effetto sul WPSNR; sulla robustezza l'effetto sembra piccolo: 0025.bmp, una delle più saturate del set (circa il 10% dei pixel a 0 o 255), è nel campione da 43 e non viene rotta né a 58,05 né a 62,05 dB (similarità minima 0,211 e 0,127), e sull'immagine più saturata in assoluto (0004.bmp, 11%) il WPSNR arriva comunque a 62,05 con correlazione 0,988 col watermark vero. Per gli attacchi sulle immagini molto saturate però non abbiamo ancora un test dedicato.
7. **Kerckhoffs**: chi legge la nostra detection sa quale banda usiamo e può stimare le coppie dalla watermarkata (non può usare l'originale). Con gli attacchi permessi però non si possono modificare singoli coefficienti DCT, quindi conoscere le posizioni aiuta poco.

---

## 10. Prossimi passi (detection, ROC, gara)

**Detection (`detection_fakebitinvestigation.py`, file unico, niente funzioni esterne tranne il WPSNR):**

1. leggere le 3 immagini con cv2 dai 3 percorsi;
2. ricalcolare le coppie **dall'originale** con lo stesso `_select_pairs` e gli stessi `R_MIN`, `R_MAX` (vanno copiati, non importati: niente percorsi relativi né script esterni, p.5);
3. `W_ext = estrai(orig, watermarked)`, `W_att = estrai(orig, attacked)`;
4. `sim` coseno, definita 0 se un vettore è nullo;
5. `output2 = wpsnr(watermarked, attacked)`;
6. `output1 = 1` se `sim ≥ τ`, altrimenti 0; **in più `output1 = 0` se `output2 ≤ 25`** (regola della penalità sulle immagini distrutte, p.8: il nostro watermark è abbastanza robusto da sopravvivere anche ad attacchi distruttivi, e lì dire "presente" verrebbe penalizzato);
7. nessuna print, < 5 s (misurato circa 0,8 s).

**ROC (codice separato, da consegnare il 27/10):** come da p.4, ma con il **watermark casuale in forma ±1** (stessa mappatura dell'embedding), altrimenti H0 viene falsata. Attacchi casuali presi dalla nostra batteria. Scegliere τ con FPR molto basso (≈0,001) dentro l'intervallo ammesso [0, 0,1].

**Consegna del 27/10 (p.3 e p.6):** oltre a embedding, detection e ROC vanno mandati anche il codice degli attacchi (con un README che spiega come usarlo) e un codice demo che mostri come lanciare ciascuno dei file; `test_embedding.py` può essere la base della demo.

**Giorno della gara (defense phase 8:30–10:00):**

1. scaricare le 3 immagini;
2. per ciascuna, embedding a 58,05 / 62,05 / 66,05 e batteria d'attacco nostra;
3. scegliere il target (eventualmente per immagine, se i punti qualità sono una media) con la regola: il più alto per cui il miglior attacco riuscito resta sotto circa 37 dB;
4. controllare che detection(orig, wm, orig) = 0, detection(orig, wm, wm) = 1, tempo < 5 s;
5. caricare immagini e detection (dopo la fine della defense phase non si può più cambiare nessun file caricato: 2 punti di penalità, p.8).

---

### Fonti

- Regolamento: `1BattleRules2026.pdf` (Multimedia Data Security, Competition Rules).
- I. J. Cox, J. Kilian, F. T. Leighton, T. Shamoon, *Secure Spread Spectrum Watermarking for Multimedia*, IEEE Transactions on Image Processing 6(12), 1997 (`ss_cox_1.pdf`).
- Notebook dei lab: `LAB3-SS_Watermarking`, `LAB3-LSB_Watermarking`, `LAB3-ROC`, `EXERCISEs_SOL` (lab 3), `EXERCISE_SOL` (lab 2), `01-Image-processing`, `02-Transform-domain-solutions`, `MDS-Lab1a/b`.
- Immagini di esempio, immagini gara 2022 e risultati degli attacchi 2022: repository pubblico [vicentinileonardo/DWT-SVD-digital-watermarking](https://github.com/vicentinileonardo/DWT-SVD-digital-watermarking) (gruppo *howimetyourmark*, a.a. 2022/23).
