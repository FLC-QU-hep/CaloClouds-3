#!/bin/bash
# ---------------------------------------------------------------------------
# run_generation_batch.sh -- run generate_showers.sh for several variants,
# one after the other, on the machine it is launched on (one GPU per NGT pod,
# so the variants have to be sequential, not parallel).
#
#   ./run_generation_batch.sh withincell subcell subcell_6kcut
#
# Env: DEVICE (cuda), LOGDIR (<repo>/gen_logs)
# ---------------------------------------------------------------------------

REPO="${REPO:-/eos/user/m/mamozzan/CaloClouds-3}"
LOGDIR="${LOGDIR:-$REPO/gen_logs}"
DEVICE="${DEVICE:-cuda}"
host="$(hostname)"

mkdir -p "$LOGDIR"
status="$LOGDIR/batch_${host}.status"

echo "=== batch on $host: $* (device=$DEVICE) started $(date)" | tee -a "$status"
for v in "$@"; do
    log="$LOGDIR/${v}_${host}.log"
    echo "--- $v -> $log  ($(date))" | tee -a "$status"
    bash "$REPO/slurm_script/generate_showers.sh" "$v" "$DEVICE" > "$log" 2>&1
    echo "--- $v finished rc=$? at $(date)" | tee -a "$status"
done
echo "=== batch on $host done $(date)" | tee -a "$status"
