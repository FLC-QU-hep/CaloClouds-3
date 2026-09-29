#!/bin/bash
# ---------------------------------------------------------------------------
# train_showerflows.sh -- train a ShowerFlow for one or more config variants,
# sequentially, one variant after the other.
#
# scripts/training/ShowerFlow.py takes a config FILE PATH as argv[1] (see its
# __main__ block), so this script does NOT touch the pointcloud/configs.py
# symlink that showerflow.sh/cc3.sh juggle.  That is what makes it safe to run
# two of these at once on different machines (e.g. ngt session-1 and
# session-2) against this same EOS checkout: nothing is global state, and each
# variant writes into its own showerFlow/input_cc3_<variant>/ directory.
#
# Usage:
#   ./train_showerflows.sh <variant> [<variant> ...]
#
#   variant : config basename without the "caloclouds_3_S2P_" prefix and the
#             ".py" suffix, e.g.
#               withincell  subcell  subcell_6kcut
#               hdbscan_ms8_mcs40  hdbscan_ms12_mcs12
#
# Each variant's stdout/stderr goes to gen_logs/showerflow_<variant>_<ts>.log.
# The flow version/num_blocks/af_dim come from the config itself, so make sure
# the config says what you want before starting (log1_stable / nb 4 / af 14 for
# the ep1500 search winner).
#
# Env overrides: REPO (default /eos/user/m/mamozzan/CaloClouds-3)
# ---------------------------------------------------------------------------
set -uo pipefail

REPO="${REPO:-/eos/user/m/mamozzan/CaloClouds-3}"
cd "$REPO" || exit 1

# venv must live off EOS for compiled extensions (see cc3.sh), but the activate
# script itself is read from EOS the same way every other script here does it.
source caloclouds3/bin/activate || exit 1

# EOS FUSE doesn't implement the POSIX locks libhdf5 wants on open/create.
export HDF5_USE_FILE_LOCKING=FALSE

if [[ $# -eq 0 ]]; then
    echo "usage: $0 <variant> [<variant> ...]" >&2
    echo "  e.g. $0 withincell subcell subcell_6kcut" >&2
    exit 2
fi

mkdir -p gen_logs

echo "############################################################"
echo "# train_showerflows.sh"
echo "# host     : $(hostname)"
echo "# variants : $*"
echo "# started  : $(date)"
echo "############################################################"

failed=0
for v in "$@"; do
    cfg="pointcloud/config_varients/caloclouds_3_S2P_${v}.py"
    if [[ ! -f "$cfg" ]]; then
        echo "!!! no such config: $cfg -- skipping $v" >&2
        failed=$((failed + 1))
        continue
    fi

    ver=$(grep -oP 'shower_flow_version\s*=\s*"\K[^"]+' "$cfg" | head -1)
    nb=$(grep -oP 'shower_flow_num_blocks\s*=\s*\K\d+' "$cfg" | head -1)
    af=$(grep -oP 'af_dim\s*=\s*\K\d+' "$cfg" | head -1)

    log="gen_logs/showerflow_${v}_$(date +%Y%m%d_%H%M%S).log"
    echo
    echo "=== [$(date +%H:%M:%S)] $v  (version=$ver nb=$nb af_dim=$af)"
    echo "=== log: $log"

    python scripts/training/ShowerFlow.py "$cfg" > "$log" 2>&1
    rc=$?

    if [[ $rc -ne 0 ]]; then
        echo "!!! [$v] FAILED rc=$rc -- see $log" >&2
        failed=$((failed + 1))
    else
        echo "=== [$(date +%H:%M:%S)] $v done"
    fi
done

echo
echo "############################################################"
echo "# finished $(date) -- $failed failed"
echo "############################################################"
exit $((failed > 0))
