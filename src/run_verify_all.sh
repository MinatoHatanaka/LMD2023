#!/usr/bin/env bash
# Run the full H1/H2/H5 grid: one process per (dataset, experiment, featureset, model group),
# P processes in parallel with T threads each. Finished (model, fold) results are skipped.
# Usage: bash src/run_verify_all.sh [P=6] [T=4]
set -u
cd "$(dirname "$0")/.."
P=${1:-6}; T=${2:-4}
mkdir -p results/logs
{
  # ET (slowest) first, on the main feature sets only.
  for ds in authors public15; do
    case $ds in authors) exps="E1 E2 E3 E3c E4a E4b";; public15) exps="E1 E2 E3 E3h E4a E4b";; esac
    for e in $exps; do for fs in F0 F4; do echo "$ds $e $fs ET"; done; done
  done
  for ds in authors public15; do
    case $ds in authors) exps="E1 E2 E3 E3c E4a E4b";; public15) exps="E1 E2 E3 E3h E4a E4b";; esac
    for e in $exps; do
      echo "$ds $e base MAJ-host MAJ-time"
      for fs in F0 F1 F2 F3 F4; do echo "$ds $e $fs NB LSVC DT RF LGBM CB"; done
    done
  done
} | xargs -P "$P" -L 1 bash -c '
  ds=$0; e=$1; fs=$2; shift 2
  log=results/logs/verify_${ds}_${e}_${fs}_${1}.log
  uv run python src/run_verify.py --dataset $ds --experiment $e --featureset $fs --models "$@" --n-jobs '"$T"' > $log 2>&1 \
    && echo "DONE $ds $e $fs $*" || echo "FAILED $ds $e $fs $*"
'
