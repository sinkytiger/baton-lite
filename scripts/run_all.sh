#!/usr/bin/env bash
# The full test-split run. Resumable: after an interruption, run the same command again.
# Paid steps come first (Sonnet and Haiku alone, then efficient-to-capable); the long local steps follow, Sonnet
# pair before the Haiku pair. Each step waits for Ollama and is retried if the model server drops out; finished
# trajectories are kept on disk, so a retry continues where it stopped. A step that hits the API budget stops
# the run (exit 3) instead of retrying.
set -uo pipefail
PY="${PY:-python}"
BUDGET="${BUDGET:-10}"
export PYTHONIOENCODING=utf-8

wait_ollama() { until ollama ps >/dev/null 2>&1; do sleep 15; done; }
attempt() {
  for i in 1 2 3 4 5 6; do
    wait_ollama
    "$PY" -u -m baton "$@" --split test --budget "$BUDGET"
    rc=$?
    [ $rc -eq 0 ] && return 0
    [ $rc -eq 3 ] && { echo "budget reached: $*"; exit 3; }
    echo "step failed (attempt $i, rc $rc): $*; retrying in 60s"
    sleep 60
  done
  echo "giving up on: $*"
  return 1
}

S=claude-sonnet-5-5
H=claude-haiku-5-5
Q=qwen3:4b
attempt solo --model $S || exit 1
attempt solo --model $H || exit 1
attempt solo --model $Q || exit 1
attempt reverse --source $Q --receiver $S || exit 1
attempt detect --source $S --judge $Q || exit 1
attempt sweep --source $S --receiver $Q --transfer incontext || exit 1
attempt logfile --source $S --receiver $Q --judge $Q || exit 1
attempt detect --source $H --judge $Q || exit 1
attempt sweep --source $H --receiver $Q --transfer incontext || exit 1
attempt logfile --source $H --receiver $Q --judge $Q || exit 1
"$PY" -m baton report --split test --out reports/results.md --csv reports/outcomes.csv
"$PY" -m baton spent
