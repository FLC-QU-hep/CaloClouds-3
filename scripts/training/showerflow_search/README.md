# Shower-flow hyperparameter search (temporary)

Exploratory scaffolding to search over `af_dim`, `shower_flow_num_blocks`,
and `shower_flow_version` for the shower-flow model, in response to
`af_dim=10` giving a visibly worse `num_points`/`visible_energy` fit than
`af_dim=6` despite near-identical validation NLL loss (see
`ShowerFlow_eval_1.png` / `_eval_2.png` in `input_cc3_hdbscan_ms3_mcs10` vs
`..._lessparams`). Meant to be deleted once you've picked a winning config -
it doesn't touch the main training pipeline (`scripts/training/ShowerFlow.py`
and `pointcloud/` are only imported, never edited).

`shower_flow_version` is a log-vs-linear-scale comparison: `"alt1"` (your
current configs) models `clusters_per_layer`/`energy_per_layer` directly on
their linearly-rescaled scale - checked directly against the data prep code
(`showerflow_training.py`'s `get_clusters_per_layer`/`get_energy_per_layer`
and `_train_ds_function_factory`): no `log()` anywhere in that path, just
division by a single global dataset-wide constant. `"log1"` uses the same
coupling/spline architecture but log-transforms those same 60 inputs
internally first (`compile_HybridTanH_log1` + `SafeExpTransform` in
`pointcloud/models/shower_flow.py`) - already implemented, just never wired
into any config or compared until now.

Each trial gets its own private `shower_flow_data_dir`
(`<search_root>/trials/trial_NNNN/`) so different `af_dim` values never
collide - the checkpoint filename encodes `num_blocks` and `shower_flow_version`,
but not `af_dim`, so two trials sharing a directory could otherwise silently
load or overwrite each other's weights. The cached per-event arrays
(clusters/energy per layer, cog, ...) don't depend on `af_dim`/`num_blocks`/
`version` at all, so they're symlinked in from an existing run rather than
recomputed, to skip the ~5-10 minute recompute per trial.

## This node has no GPU / scheduler

This dev node can't run training. `train_trial.py` is scheduler-agnostic
(just a CLI script indexed by `--trial_id`), so it works however you launch
jobs on your GPU cluster. `run_search.py` writes a plain sequential bash
runner (`run_all_sequential.sh`) for an interactive GPU session; if you use
Slurm or HTCondor at CERN, tell me which and I'll add a proper array job /
submit file instead.

## Usage

Everything below happens in `pipeline.sh` - edit the variables at the top
(dataset, search ranges, epochs, `SHOWER_FLOW_VERSIONS`) and run:

```bash
bash /eos/user/m/mamozzan/CaloClouds-3/scripts/training/showerflow_search/pipeline.sh
```

That's equivalent to running the three steps by hand:

```bash
cd /eos/user/m/mamozzan/CaloClouds-3
source caloclouds3/bin/activate

# 1. Generate the trial manifest (af_dim x num_blocks x shower_flow_version
#    random search). Always includes af_dim=6/nb=2/alt1, af_dim=10/nb=2/alt1,
#    and af_dim=6/nb=2/log1 as trials 0/1/2, so the search is directly
#    comparable to the two runs already on disk and isolates the
#    log-vs-linear question with af_dim/num_blocks held fixed.
python scripts/training/showerflow_search/run_search.py \
    --base_config pointcloud/config_varients/caloclouds_3_S2P_hdbscan_ms3_mcs10.py \
    --cache_dir /eos/user/m/mamozzan/point-cloud-diffusion-data/showerFlow/input_cc3_hdbscan_ms3_mcs10 \
    --search_root /eos/user/m/mamozzan/point-cloud-diffusion-data/showerFlow_search/hdbscan_ms3_mcs10 \
    --n_trials 16 --af_dim_min 4 --af_dim_max 16 \
    --num_blocks_min 2 --num_blocks_max 6 \
    --shower_flow_versions alt1,log1 --seed 0

# 2. Run every trial (sequential; edit run_all_sequential.sh, or pipeline.sh,
#    for parallel/queued execution on your actual cluster).
bash /eos/user/m/mamozzan/point-cloud-diffusion-data/showerFlow_search/hdbscan_ms3_mcs10/run_all_sequential.sh

# 3. (run_all_sequential.sh / pipeline.sh do this automatically, but standalone:)
python scripts/training/showerflow_search/summarize_results.py \
    --search_root /eos/user/m/mamozzan/point-cloud-diffusion-data/showerFlow_search/hdbscan_ms3_mcs10
```

`summarize_results.py` ranks trials by combined Wasserstein distance
(`w_num_points + w_visible_energy`, sampled vs. truth), prints a mean
score broken out by `shower_flow_version` if more than one was searched,
and writes `summary.csv` + `summary_scatter.png` (marker shape = version,
color = score).

## Notes / things to check before trusting a result

- `--epochs 800` in `run_all_sequential.sh` is a reduced budget (real runs
  use 3000) purely to make the search cheaper - it's a ranking proxy, not a
  final model. Re-train the winning `(af_dim, num_blocks)` at the full
  epoch budget before using it for real.
- The baseline trials (0 and 1, and 2 if `log1` is included) are re-trained
  from scratch by this harness, not reused from the existing
  `input_cc3_hdbscan_ms3_mcs10` / `_lessparams` checkpoints - with 800
  epochs instead of ~3000 their absolute numbers won't exactly match what we
  looked at earlier, but they should still be directly comparable to every
  other trial in the same search since all trials share the epoch budget.
- `wasserstein_metrics()` in `common.py` only implements the
  fixed-norm-derived-sum path (`shower_flow_inputs = ["clusters_per_layer",
  "energy_per_layer"]` with `shower_flow_fixed_input_norms = True`, true for
  all the hdbscan/withincell/subcell configs, and what `caloclouds_3_S2P_
  hdbscan_ms3_mcs10.py` uses). It does **not** handle configs like
  `caloclouds_2.py`/`default.py` that put `total_clusters`/`total_energy`
  directly in `shower_flow_inputs` - extend it (mirroring the `have_num_points`
  / `have_visible_energy` branches in `ShowerFlow.py`'s eval block) before
  reusing this on those.

## Why log1_stable

Moved here from the docstring of `pointcloud/models/stable_log1.py`.

Root cause of the NaN/inf seen in the original "log1". Measured on the cached
hdbscan_ms3_mcs10 arrays, exact zeros are only 7.2% of the 60 values,
concentrated in the shower tail (layer 29 is 41% zero, layer 0 is 11%, the
middle layers are ~0%). The problem is not how many zeros there are, it is
that they form a point mass, and a continuous flow cannot fit an atom: to
place finite probability on a delta it has to drive its density to infinity
there, which sends the log-det term to +-inf.

log(x + eps) with a tiny eps is what turns a mild zero-inflation into a fatal
one. With eps=1e-6 the atom lands at log(1e-6) = -13.8 and, after z-scoring,
sits 6.2 sigma (median) to 14.6 sigma (max) away from the non-zero body - z
range [-14.57, +2.13]. In linear space (alt1) that same atom sits at 0, at the
edge of the data, which is why alt1 trains cleanly and every log run hit NaN
training batches.

Fix, all confined to `stable_log1.py`:
1. `LOG_OFFSET` instead of a 1e-6 epsilon, which pulls the atom in to a z
   range of [-5.46, +8.72].
2. z-score standardize the 60 log-space values (per-dimension mean/std,
   computed once from the cached clusters_per_layer.npz / energy_per_layer.npz
   / input_norms.npz - the same "fixed norm" inputs the real training pipeline
   uses) via an extra affine bijector right before the log/exp step. This
   cannot remove the atom on its own: an affine map of a point mass is still a
   point mass, which is why (1) was needed.
3. clamp the pre-exp() value (`ClampedSafeExpTransform`, a local copy of
   `shower_flow.SafeExpTransform` with a clamp added) as a hard safety net
   against overflow.

Choice of `LOG_OFFSET`, measured on the z-scored log inputs of the cached
hdbscan_ms3_mcs10 arrays. Larger pulls the zero-atom towards the data body,
but also lets a sampled layer undershoot to -LOG_OFFSET:

| offset | z range         | frac \|z\|>5 | worst undershoot |
|--------|-----------------|--------------|------------------|
| 1e-6   | [-14.57, +2.13] | 0.00398      | -0.00 hits       |
| 1e-3   | [-7.85, +4.71]  | 0.00132      | -0.13 hits       |
| 1e-2   | [-5.46, +8.72]  | 0.00020      | -1.25 hits (chosen) |
| 1e-1   | [-3.49, +22.1]  | 0.00045      | -12.5 hits       |

1e-2 minimizes the tail fraction; -1.25 hits of possible undershoot is
negligible against Wasserstein distances of 40-250 hits.

Still outstanding: the real fix for the atom is dequantization (the cluster
counts are integers, so log(n + u) with u ~ U(0,1) removes it outright), or
factorizing out an explicit per-layer emptiness mask. The clusters and energy
zero-masks are identical - a layer with no hits has no energy - so that mask is
one 30-dim Bernoulli, not two. See also the log_abs_det_jacobian caveat on
`ClampedSafeExpTransform`.
