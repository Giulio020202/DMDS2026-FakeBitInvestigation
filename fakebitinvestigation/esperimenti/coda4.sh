#!/bin/bash
# Terza iterazione (14/10), dopo coda3:
#  1) attaccante adattivo contro v1.0 e v2.0 ibrida (62.05 e 58.05 dB), ciascuna alla sua tau a FPR realistica 0.1%
#  2) soglia normalizzata: batteria v2 (meta' A e B) + falsi positivi realistici, v1.0 e v2.0
#  3) ROC ufficiale della v2.0 ibrida (tau da mettere nella detection v2.0)
cd "$(dirname "$0")/.."
PY=../OFF_MDS_ENV/bin/python
export OMP_NUM_THREADS=1
while pgrep -f "^/bin/bash esperimenti/coda3.sh" >/dev/null; do sleep 30; done   # aspetta che la coda3 non sia attiva
echo "### 1) attaccante adattivo"
$PY esperimenti/adaptive.py base 0.1045 --tag v1
$PY esperimenti/adaptive.py dct_dwt34 0.0851 --tag v2
# stesso test a 58.05 dB (con il "6 politico" + bonus potrebbe convenire: vedi CLAUDE.md §7)
$PY esperimenti/adaptive.py base 0.1045 --target 58.05 --tag v1_t58
$PY esperimenti/adaptive.py dct_dwt34 0.0851 --target 58.05 --tag v2_t58
echo "### 2) soglia normalizzata"
for c in base dct_dwt34; do for s in A B; do $PY esperimenti/multi.py $c --subset $s; done; done
$PY esperimenti/h0_multi.py base 40
$PY esperimenti/h0_multi.py dct_dwt34 40
echo "### 3) ROC v2.0"
(cd v2.0_ibrida && ../$PY roc_fakebitinvestigation.py)   # PY e' relativo a fakebitinvestigation/
echo CODA4 FINITA
