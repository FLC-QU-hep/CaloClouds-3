#!/bin/bash
# Runs the whole shower-flow hyperparameter search end to end:
#   1. generate the trial manifest (af_dim x num_blocks x shower_flow_version)
#   2. train + score every trial, sequentially
#   3. summarize results into summary.csv / summary_scatter.png
#
# Edit the variables below, then run from a GPU-accessible machine:
#   bash scripts/training/showerflow_search/pipeline.sh
#
# For parallel/queued execution on your actual cluster (Slurm/HTCondor),
# don't use this loop - submit one job per trial_id calling train_trial.py
# directly instead (ask for a proper array/submit file once you know which
# scheduler you're using).
set -e

REPO_ROOT="/eos/user/m/mamozzan/CaloClouds-3"
BASE_CONFIG="pointcloud/config_varients/caloclouds_3_S2P_hdbscan_ms3_mcs10.py"
CACHE_DIR="/eos/user/m/mamozzan/point-cloud-diffusion-data/showerFlow/input_cc3_hdbscan_ms3_mcs10"
SEARCH_ROOT="/eos/user/m/mamozzan/point-cloud-diffusion-data/showerFlow_search/hdbscan_ms3_mcs10"

N_TRIALS=16
AF_DIM_MIN=4
AF_DIM_MAX=16
NUM_BLOCKS_MIN=2
NUM_BLOCKS_MAX=6
SHOWER_FLOW_VERSIONS="alt1,log1_stable"   # log-vs-linear-scale comparison; "log1" (unfixed) is numerically broken, don't use it
EPOCHS=800                          # reduced budget for ranking; re-train the winner at ~3000 for a final model
SEED=0

cd "$REPO_ROOT"
source caloclouds3/bin/activate

echo "=== 1. generating trial manifest ==="
python scripts/training/showerflow_search/run_search.py \
    --base_config "$BASE_CONFIG" \
    --cache_dir "$CACHE_DIR" \
    --search_root "$SEARCH_ROOT" \
    --n_trials "$N_TRIALS" \
    --af_dim_min "$AF_DIM_MIN" --af_dim_max "$AF_DIM_MAX" \
    --num_blocks_min "$NUM_BLOCKS_MIN" --num_blocks_max "$NUM_BLOCKS_MAX" \
    --shower_flow_versions "$SHOWER_FLOW_VERSIONS" \
    --seed "$SEED"

echo "=== 2. running trials ==="
N_TRIALS_ACTUAL=$(python3 -c "import json; print(len(json.load(open('$SEARCH_ROOT/trials.json'))['trials']))")
for ((i=0; i<N_TRIALS_ACTUAL; i++)); do
    echo "--- trial $i / $((N_TRIALS_ACTUAL-1)) ---"
    python scripts/training/showerflow_search/train_trial.py \
        --search_root "$SEARCH_ROOT" --trial_id "$i" --epochs "$EPOCHS"
done

echo "=== 3. summarizing results ==="
python scripts/training/showerflow_search/summarize_results.py --search_root "$SEARCH_ROOT"
