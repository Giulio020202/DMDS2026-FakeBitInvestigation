#!/bin/bash
# Predizione dell'attacco, versione VELOCE (14/10 sera). Parte dopo la coda4.
#  1) immagini distrutte + tempi a CPU libera (un processo)
#  2) batteria v3 su 26 immagini della meta' A (cos, rs_cos, sq, pr, prsq)
#  3) falsi positivi realistici con gli stessi detector
# Analisi: cd esperimenti && python equal_fpr_v3.py v3A26_fast
cd "$(dirname "$0")/.."
PY=../OFF_MDS_ENV/bin/python
export OMP_NUM_THREADS=1
D="cos,rs_cos,sq:16:0.5,pr,prsq:16:0.5"
while pgrep -f "^/bin/bash esperimenti/coda4.sh" >/dev/null; do sleep 30; done
echo "### 1) immagini distrutte e tempi (CPU libera)"
$PY esperimenti/distrutte.py 0.11 --dets $D
echo "### 2) batteria v3, 26 immagini di A"
$PY esperimenti/run_battery.py --tag v3A26_fast --subset A --limit 26 --procs 13 --v3 --dets $D
echo "### 3) falsi positivi realistici"
(cd esperimenti && $PY h0.py v3 "$D" 40)
echo CODA5 FINITA
