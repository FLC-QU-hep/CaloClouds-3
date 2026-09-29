#!/bin/bash
# ---------------------------------------------------------------------------
# generate_after_flow.sh -- wait for a variant's ShowerFlow training to finish,
# then immediately run the full generation pipeline for it.
#
# Chains train_showerflows.sh -> generate_showers.sh without a human in the
# middle, so a flow that finishes overnight is followed straight away by the
# 10k + scan generation that depends on it.
#
# Usage:
#   ./generate_after_flow.sh <variant> [<variant> ...]
#
#   variant : the same name generate_showers.sh takes AND the config basename
#             after caloclouds_3_S2P_ (they coincide for all five retrains):
#               withincell  subcell  subcell_6kcut
#               hdbscan_ms8_mcs40  hdbscan_ms12_mcs12
#
# Readiness test per variant, both conditions required:
#   1. its log1_stable_nb4 *_best.pth exists in the variant's showerFlow dir, and
#   2. no ShowerFlow.py process is running for that variant's config.
# Condition 1 alone is not enough - best.pth is rewritten every time validation
# improves, so it appears early in training. Condition 2 alone is not enough
# either: variants queued behind others have not started yet, so their process
# is legitimately absent. Together they mean "trained and finished".
#
# Launch detached so it outlives the ssh session that started it:
#   setsid nohup ./generate_after_flow.sh withincell subcell subcell_6kcut \
#       > gen_logs/chain_<host>.log 2>&1 < /dev/null &
#
# Env overrides: REPO, DEVICE (default cuda), POLL_SECONDS (default 120),
#                TIMEOUT_HOURS (default 24, per variant)
# ---------------------------------------------------------------------------
set -uo pipefail

REPO="${REPO:-/eos/user/m/mamozzan/CaloClouds-3}"
DEVICE="${DEVICE:-cuda}"
POLL_SECONDS="${POLL_SECONDS:-120}"
TIMEOUT_HOURS="${TIMEOUT_HOURS:-24}"
SF_BASE="/eos/user/m/mamozzan/point-cloud-diffusion-data/showerFlow"
FLOW_PTH="ShowerFlow_log1_stable_nb4_inputs1152921504606846975_fnorms_best.pth"

cd "$REPO" || exit 1

# variant -> showerFlow directory (they do not follow one naming rule)
flow_dir_for() {
    case "$1" in
        withincell)         echo "$SF_BASE/input_cc3_merge_within_cell" ;;
        subcell)            echo "$SF_BASE/input_cc3_merge_within_regular_subcell" ;;
        subcell_6kcut)      echo "$SF_BASE/input_cc3_merge_within_regular_subcell_6kcut" ;;
        hdbscan_ms8_mcs40)  echo "$SF_BASE/input_cc3_hdbscan_ms8_mcs40" ;;
        hdbscan_ms12_mcs12) echo "$SF_BASE/input_cc3_hdbscan_ms12_mcs12" ;;
        *)                  echo "" ;;
    esac
}

if [[ $# -eq 0 ]]; then
    echo "usage: $0 <variant> [<variant> ...]" >&2
    exit 2
fi

mkdir -p gen_logs
echo "############################################################"
echo "# generate_after_flow.sh on $(hostname)"
echo "# variants : $*"
echo "# started  : $(date)"
echo "############################################################"

for v in "$@"; do
    fdir=$(flow_dir_for "$v")
    if [[ -z "$fdir" ]]; then
        echo "!!! [$v] unknown variant, skipping" >&2
        continue
    fi
    pth="$fdir/$FLOW_PTH"

    echo
    echo "=== [$(date +%H:%M:%S)] waiting for $v flow -> $pth"
    deadline=$(( SECONDS + TIMEOUT_HOURS * 3600 ))
    ready=0
    while (( SECONDS < deadline )); do
        if [[ -f "$pth" ]] && ! pgrep -f "ShowerFlow\.py .*caloclouds_3_S2P_${v}\.py" >/dev/null; then
            ready=1; break
        fi
        sleep "$POLL_SECONDS"
    done

    if (( ! ready )); then
        echo "!!! [$v] timed out after ${TIMEOUT_HOURS}h waiting for its flow - not generating" >&2
        continue
    fi

    echo "=== [$(date +%H:%M:%S)] $v flow ready, starting generation"
    log="gen_logs/gen_${v}_$(date +%Y%m%d_%H%M%S).log"
    bash slurm_script/generate_showers.sh "$v" "$DEVICE" > "$log" 2>&1
    rc=$?
    nfail=$(grep -c FAILED "$log" 2>/dev/null || echo 0)
    echo "=== [$(date +%H:%M:%S)] $v generation exited rc=$rc, $nfail failed step(s) -> $log"
done

echo
echo "############################################################"
echo "# generate_after_flow.sh finished $(date)"
echo "# NOTE: projection/run.sh still has to be run once at the end"
echo "#       to grid-project and re-plot everything."
echo "############################################################"
