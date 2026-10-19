#!/bin/bash
# Terza iterazione (14/10): detection con predizione dell'attacco, batteria v3 (v2 + anti-predizione).
cd "$(dirname "$0")/.."
PY=../OFF_MDS_ENV/bin/python
D="cos,rs_cos,sq:16:0.5,pr,prsq:16:0.5"
export OMP_NUM_THREADS=1
run() { tag=$1; shift; [ -f esperimenti/risultati/$tag.pkl ] || $PY esperimenti/run_battery.py --tag $tag --procs 13 --v3 --dets $D "$@"; }
run v3A_base --subset A
(cd esperimenti && $PY h0.py v3 "$D" 40)
echo CODA3 FINITA
