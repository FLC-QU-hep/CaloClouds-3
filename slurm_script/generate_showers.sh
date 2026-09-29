#!/bin/bash
# ---------------------------------------------------------------------------
# generate_showers.sh -- the shower-generation part of cc3.sh, on its own.
#
# Extracted verbatim from slurm_script/cc3.sh: everything from "# to run
# generation" down to (but not including) the "NOT NEEDED / step2point"
# section.  No diffusion/CD training, no step2point conversion or projection.
#
# The configs.py symlink dance from cc3.sh is deliberately NOT here:
# generate_showers.py imports the config variant directly (_CONFIG_CLS,
# selected by _detect_variant() from the log_dir name), so nothing in this
# script touches pointcloud/configs.py.  That is what makes it safe to run
# several variants at once on different machines sharing this EOS home.
#
# Usage:
#   ./generate_showers.sh <variant> [device]
#
#   variant : withincell | subcell | subcell_6kcut
#             hdbscan_ms8_mcs40 | hdbscan_ms3_mcs10 | hdbscan_ms3_mcs3
#             hdbscan_ms7_mcs7
#             hdbscan_ms12_mcs12
#   device  : torch device, default "cuda".  (cc3.sh passed no --device at all
#             and so silently used generate_showers.py's "cpu" default.)
#
# Env overrides: N_MAIN (10000), N_SCAN (5000), REPO, SCAN_E, SCAN_THETA
# ---------------------------------------------------------------------------

set -uo pipefail

REPO="${REPO:-/eos/user/m/mamozzan/CaloClouds-3}"
cd "$REPO" || exit 1
# venv must live off EOS (EOS's FUSE mount doesn't reliably support mmap,
# which dlopen() needs for compiled extensions -> intermittent Bus error)
source caloclouds3/bin/activate || exit 1

# EOS's FUSE mount doesn't implement POSIX locks the way libhdf5 expects; without
# this an h5 write can die mid-file and leave a 0-byte output behind.
export HDF5_USE_FILE_LOCKING=FALSE

export data="${1:-}"
device="${2:-cuda}"
n_main="${N_MAIN:-10000}"
n_scan="${N_SCAN:-5000}"
scan_e=(${SCAN_E:-10 50 100})
scan_theta=(${SCAN_THETA:-0 20 40})

# option [withincell, subcell, subcell_6kcut, hdbscan_ms8_mcs40,
#         hdbscan_ms3_mcs10, hdbscan_ms3_mcs3, hdbscan_ms7_mcs7, hdbscan_ms12_mcs12]
if [[ "$data" == "withincell" ]]; then
    # export log_dir=merge_within_cell_2026_07_10__15_01_37 # only 10 files
    export log_dir=merge_within_cell_2026_08_13__12_44_33
    export file="/eos/project/f/fast/input_cc3/cc3input_merge_within_cell/input_cc3_file_0.h5"
elif [[ "$data" == "subcell" ]]; then
    export log_dir=merge_within_regular_subcell_2026_08_17__12_22_08
    export file="/eos/project/f/fast/input_cc3/cc3input_merge_within_regular_subcell/input_cc3_file_0.h5"
elif [[ "$data" == "subcell_6kcut" ]]; then
    export log_dir=merge_within_regular_subcell_6kcut_2026_08_17__13_11_48
    export file="/eos/project/f/fast/input_cc3/cc3input_merge_within_regular_subcell_6kcut/input_cc3_file_0.h5"
elif [[ "$data" == "hdbscan_ms8_mcs40" ]]; then
    export log_dir=hdbscan_ms8_mcs40_2026_08_19__14_32_00 # ms 8 mcs 40
    export file="/eos/project/f/fast/input_cc3/cc3input_hdbscan_ms8_mcs40/input_cc3_file_0.h5"
elif [[ "$data" == "hdbscan_ms3_mcs10" ]]; then
    # export log_dir=hdbscan_ms3_mcs10_2026_07_10__15_46_43 # 10 files
    export log_dir=hdbscan_ms3_mcs10_2026_07_29__15_31_21
    export file="/eos/project/f/fast/input_cc3/cc3input_hdbscan_ms3_mcs10/input_cc3_file_0.h5"
elif [[ "$data" == "hdbscan_ms3_mcs3" ]]; then
    export log_dir=hdbscan_ms3_mcs3_2026_09_10__18_39_40 # ms 3 mcs 3, diffusion done 2026-09-12 (10M iters)
    export file="/eos/project/f/fast/input_cc3/cc3input_hdbscan_ms3_mcs3/input_cc3_file_0.h5"
elif [[ "$data" == "hdbscan_ms7_mcs7" ]]; then
    export log_dir=hdbscan_ms7_mcs7_2026_09_09__11_19_50 # ms 7 mcs 7, diffusion done 2026-09-10 (10M iters)
    export file="/eos/project/f/fast/input_cc3/cc3input_hdbscan_ms7_mcs7/input_cc3_file_0.h5"
elif [[ "$data" == "hdbscan_ms12_mcs12" ]]; then
    # export log_dir=hdbscan_ms12_mcs12_2026_07_21__10_44_53 # the one cc3.sh names
    export log_dir=hdbscan_ms12_mcs12_2026_08_26__18_00_00 # ms 12 mcs 12, newer retrain
    export file="/eos/project/f/fast/input_cc3/cc3input_hdbscan_ms12_mcs12/input_cc3_file_0.h5"
else
    echo "usage: $0 <withincell|subcell|subcell_6kcut|hdbscan_ms8_mcs40|hdbscan_ms3_mcs10|hdbscan_ms3_mcs3|hdbscan_ms7_mcs7|hdbscan_ms12_mcs12> [device]" >&2
    exit 2
fi

# Optional: generate from one specific checkpoint instead of the newest one
# found by _latest_checkpoint() in $log_dir.  Needed when EOS's FUSE mount goes
# stale on a log_dir -- readdir() returns an empty directory (so
# _latest_checkpoint raises "No checkpoint found") while the MGM still lists
# every file, e.g.
#   eos root://eosuser.cern.ch ls log_dir/<variant>/   # 1001 files
#   ls log_dir/<variant>/                              # 0 files
# Workaround: copy the checkpoint off EOS and point MODEL_PATH at it,
#   eos root://eosuser.cern.ch cp <...>/ckpt_0.000000_10000000.pt /tmp/
#   MODEL_PATH=/tmp/ckpt_0.000000_10000000.pt ./generate_showers.sh <variant> <device>
# Both generate_showers.py and calculate_coef.py take --model_path.
model_args=()
if [[ -n "${MODEL_PATH:-}" ]]; then
    model_args=(--model_path "$MODEL_PATH")
    echo "# MODEL_PATH override: $MODEL_PATH"
fi

failed=0
run() {   # echo + run, keep going on failure so one bad plot can't kill the batch
    echo
    echo "=== [$data] $* "
    "$@"
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "!!! [$data] FAILED (rc=$rc): $*" >&2
        failed=$((failed + 1))
    fi
    return $rc
}

echo "############################################################"
echo "# variant : $data"
echo "# log_dir : $log_dir"
echo "# cond    : $file"
echo "# device  : $device   n_main=$n_main  n_scan=$n_scan"
echo "# host    : $(hostname)   started $(date)"
echo "############################################################"

# ---------------------------------------------------------------- base run --
export n=$n_main
run python scripts/generate_showers.py \
    --log_dir log_dir/$log_dir \
    --n_showers $n \
    --chunk_size 2000 \
    --gen_batch_size 64 \
    --cond_file $file \
    --energy_units MeV \
    --device $device

run python scripts/plotting/compare_generated_vs_input_cc3.py \
    generated_showers/$log_dir/generated_showers_$n/generated_showers.h5 --data-used $data

# ---- occupancy poly-fit calibration (alternative to the flat _OCCUPANCY_SCALE_N) ----
# only a marginal improvement over the flat scale for subcell (mid-range slightly
# better, high-energy slightly worse, plus a handful of zero-hit showers at low
# energy from the cubic fit's extrapolation) -- kept here for comparison, not
# because it's clearly better.
run python scripts/training/calculate_coef.py \
    --log_dir log_dir/$log_dir \
    --n_showers $n \
    --degree 3 \
    --device $device

# ---------------------------------------------------------------- poly run --
run python scripts/generate_showers.py \
        --log_dir log_dir/$log_dir \
        --n_showers $n \
        --chunk_size 2000 \
        --gen_batch_size 128 \
        --cond_file $file \
        --energy_units MeV \
        --occupancy_fit poly \
        --output_file generated_showers_poly.h5 \
        --device $device

for f in generated_showers/$log_dir/generated_showers_$n/generated_showers_poly.h5; do
    run python scripts/plotting/compare_generated_vs_input_cc3.py "$f" \
        --data-used $data \
        --out-dir generated_showers/$log_dir/generated_showers_$n/poly_fit_plots
done

# ------------------------------------- fixed-energy / fixed-theta scans -----
export n=$n_scan
for E in "${scan_e[@]}"; do
    run python scripts/generate_showers.py \
        --log_dir log_dir/$log_dir \
        --n_showers $n \
        --chunk_size 2000 \
        --gen_batch_size 64 \
        --cond_file $file \
        --energy_units MeV \
        --occupancy_fit poly \
        --output_file generated_showers_poly.h5 \
        --fixed_energy_gev $E \
        --device $device
done
for T in "${scan_theta[@]}"; do
    run python scripts/generate_showers.py \
        --log_dir log_dir/$log_dir \
        --n_showers $n \
        --chunk_size 2000 \
        --gen_batch_size 64 \
        --cond_file $file \
        --energy_units MeV \
        --occupancy_fit poly \
        --output_file generated_showers_poly.h5 \
        --fixed_theta_deg $T \
        --device $device
done

for f in generated_showers/$log_dir/generated_showers_$n/generated_showers_poly_*.h5; do
    [[ -e "$f" ]] || continue
    # cc3.sh's glob also matches leftover DDML *_postprocessed.h5 outputs sitting in the
    # same folder. Those have no 'events' dataset, so compare_generated_vs_input_cc3.py
    # dies with a KeyError on every one of them. Skip them instead.
    [[ "$f" == *_postprocessed.h5 ]] && continue
    run python scripts/plotting/compare_generated_vs_input_cc3.py "$f" \
        --data-used $data \
        --out-dir generated_showers/$log_dir/generated_showers_$n/poly_fit_plots
done

echo
echo "############################################################"
echo "# [$data] done $(date) -- $failed failed step(s)"
echo "############################################################"
exit $(( failed > 0 ))
