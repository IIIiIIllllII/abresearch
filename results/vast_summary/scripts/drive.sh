#!/bin/bash
# usage: drive.sh <model> <layer|auto> [<layer> ...]
set -u
cd /workspace/abresearch
source /venv/main/bin/activate
unset HF_TOKEN
export HF_HUB_ENABLE_HF_TRANSFER=1 HF_XET_HIGH_PERFORMANCE=1 PYTHONPATH=src PYTHONUNBUFFERED=1 MPLCONFIGDIR=/tmp/mpl
SP=/tmp/claude-0/-workspace-abresearch/e15e1135-1601-45d2-95d8-48eb63b0ab53/scratchpad
M=$1; shift; S=${M//\//__}
EXTRA="${BS_EXTRA:-}"
python -m abr.run_all --set model=$M tag=base $EXTRA --stages collect directions || exit 1
python $SP/sweep.py --set model=$M tag=base $EXTRA || exit 1
B=results/${S}_base
for L in "$@"; do
  [ "$L" = auto ] && L=$(cat $B/auto_layer.txt)
  R=results/${S}_L$L; mkdir -p $R; ln -sfn full data/sets/L$L
  for f in activations directions.pt directions_layers.csv layer_sweep.csv; do ln -sfn ../${S}_base/$f $R/$f; done
  python -m abr.run_all --set model=$M tag=L$L n_eval=${N_EVAL:-48} 'extra_conditions=[D_ra]' $EXTRA --stages ablate_eval analyze --layer $L || exit 1
done
