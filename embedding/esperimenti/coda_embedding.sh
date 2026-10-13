#!/bin/bash
# Varianti di embedding valutate sulla meta' A con il detector combinato.
cd "$(dirname "$0")/.."
PY=../OFF_MDS_ENV/bin/python
D="cos,rs_cos,sq:16:0.5,sq:8:0.1"
export OMP_NUM_THREADS=1
run() { tag=$1; shift; [ -f esperimenti/risultati/$tag.pkl ] || $PY esperimenti/run_battery.py --tag $tag --subset A --procs 13 --dets $D "$@"; }
run eA_base
run eA_b16_96   --rmin 16 --rmax 96
run eA_b24_128  --rmin 24 --rmax 128
run eA_b32_96   --rmin 32 --rmax 96
run eA_b48_160  --rmin 48 --rmax 160
run eA_g05      --gamma 0.5
run eA_g10      --gamma 1.0
run eA_t58      --target 58.05
echo CODA FINITA
