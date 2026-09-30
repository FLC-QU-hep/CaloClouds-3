#!/bin/bash
# Run step2point's HDBSCAN sweep for the ILD photon sample (the CaloClouds-3
# input) and make the two sweep figures of the paper:
#   hdbscan_sweep_mean_compression_ratio.pdf  (mean compression ratio vs m_cs)
#   hdbscan_sweep_profiles.pdf                (points per layer, point energy; m_cs = 20)
#
# Uses hdbscan_sweep/sweep_hdbscan.py from step2point (--preset ild). Grid
# points already in $SWEEP_DIR are not rerun, so with the existing outputs this
# only remakes the plots. Locations (override if needed):
STEP2POINT_DIR="${STEP2POINT_DIR:-/eos/user/m/mamozzan/step2point}"
export CC3_DIR="${CC3_DIR:-/eos/user/m/mamozzan/CaloClouds-3}"     # needed by --preset ild
export SWEEP_DIR="${SWEEP_DIR:-$CC3_DIR/outputs/hdbscan_sweep}"   # sweep runs, read by the plots
FIG_DIR="${FIG_DIR:-$CC3_DIR/outputs/paper_figures}"

# EDM4hep input: Key4hep + the step2point venv
# shellcheck source=/dev/null
source /cvmfs/sw.hsf.org/key4hep/setup.sh
# shellcheck source=/dev/null
source "$STEP2POINT_DIR/.venv-key4hep/bin/activate"

SWEEP="$STEP2POINT_DIR/hdbscan_sweep/sweep_hdbscan.py"

# The paper grid: 37 (m_cs, m_s) points, m_s <= m_cs. m_s = 2 only for
# m_cs = 3, 5, 8, 12, 20, hence two sweeps.
python "$SWEEP" --preset ild --min-cluster-sizes 3 5 8 12 20 --min-samples 2 3 5 8 12 20 || exit
python "$SWEEP" --preset ild --min-cluster-sizes 10 15 25 40 --min-samples 3 5 8 12 20 || exit

# Paper figures, read from $SWEEP_DIR. They were made with step2point's plain
# .venv (matplotlib 3.10.9), outside Key4hep; Key4hep's matplotlib (3.10.7)
# renders them slightly differently. So plot in a clean environment:
PLOT_PY="${PLOT_PY:-$STEP2POINT_DIR/.venv/bin/python}"
plot() { env -i HOME="$HOME" PATH=/usr/bin:/bin SWEEP_DIR="$SWEEP_DIR" "$PLOT_PY" "$@"; }
mkdir -p "$FIG_DIR"
plot "$CC3_DIR/paper_figures/plot_mean_compression.py" "$FIG_DIR/hdbscan_sweep_mean_compression_ratio.pdf" || exit
plot "$CC3_DIR/paper_figures/plot_sweep_profiles.py" "$FIG_DIR" || exit
mv "$FIG_DIR/sweep_profiles.pdf" "$FIG_DIR/hdbscan_sweep_profiles.pdf"
echo "Figures in $FIG_DIR"
