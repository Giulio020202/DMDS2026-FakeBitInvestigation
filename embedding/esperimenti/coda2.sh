#!/bin/bash
# Verifica su B delle varianti vincenti su A + falsi positivi con la maschera.
cd "$(dirname "$0")/.."
PY=../OFF_MDS_ENV/bin/python
D="cos,rs_cos,sq:16:0.5,sq:8:0.1"
export OMP_NUM_THREADS=1
while pgrep -f "tag eA_dwt" >/dev/null; do sleep 20; done
run() { tag=$1; shift; [ -f esperimenti/risultati/$tag.pkl ] || $PY esperimenti/run_battery.py --tag $tag --procs 13 --dets $D "$@"; }
run eB_base   --subset B
run eB_g05    --subset B --gamma 0.5
run eA_t58g05 --subset A --gamma 0.5 --target 58.05
(cd esperimenti && $PY h0.py g05 "cos,rs_cos,sq:16:0.5" 40 0.5)
echo CODA2 FINITA
