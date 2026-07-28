cd /eos/user/m/mamozzan/CaloClouds-3/ || exit
# unset PYTHONPATH
# venv must live off EOS (EOS's FUSE mount doesn't reliably support mmap,
# which dlopen() needs for compiled extensions -> intermittent Bus error)
source caloclouds3/bin/activate
cd ../projection || exit

# EOS FUSE doesn't reliably support the POSIX locks HDF5 takes on file
# create/open -> intermittent BlockingIOError ("Resource temporarily
# unavailable") on writes.
export HDF5_USE_FILE_LOCKING=FALSE

INPUT=/eos/user/m/mamozzan/CaloClouds-3/generated_showers/merge_within_regular_subcell_6kcut_2026_07_24__13_10_06/generated_showers_5000/generated_showers_poly.h5
POSTPROCESSED="${INPUT%.h5}_postprocessed.h5"
# Reference ("Geant4"/gray) line for the shower-profile comparison plot - the
# actual conditioning file generation was paired against, run through the
# grid-projection pipeline (not the raw --no-project per-hit output).
# REFERENCE=/eos/user/m/mamozzan/step2point/outputs/cc3input_merge_within_cell/input_cc3_file_0_postprocessed.h5
REFERENCE=/eos/user/m/mamozzan/step2point/outputs/cc3input_merge_within_regular_subcell/input_cc3_file_0_postprocessed.h5
REFERENCE1=/eos/user/m/mamozzan/CaloClouds-3/generated_showers/merge_within_cell_2026_07_10__15_01_37/generated_showers_5000/generated_showers_poly_postprocessed.h5
# REFERENCE2=/eos/user/m/mamozzan/CaloClouds-3/generated_showers/merge_within_regular_subcell_2026_07_20__15_35_38/generated_showers_5000/generated_showers_poly_postprocessed.h5
REFERENCE2=/eos/user/m/mamozzan/CaloClouds-3/generated_showers/merge_within_regular_subcell_6kcut_2026_07_24__13_10_06/generated_showers_5000/generated_showers_poly_postprocessed.h5
REFERENCE3=/eos/user/m/mamozzan/CaloClouds-3/generated_showers/hdbscan_ms3_mcs10_2026_07_10__15_46_43/generated_showers_5000/generated_showers_poly_postprocessed.h5
REFERENCE4=/eos/user/m/mamozzan/CaloClouds-3/generated_showers/hdbscan_ms8_mcs40_2026_07_20__18_29_36/generated_showers_5000/generated_showers_poly_postprocessed.h5
REFERENCE5=/eos/user/m/mamozzan/CaloClouds-3/generated_showers/hdbscan_ms12_mcs12_2026_07_21__10_44_53/generated_showers_5000/generated_showers_poly_postprocessed.h5
# postprocessing.py takes a DDML-format h5 (events/energy/layer_counts/n_points/
# phi_global/theta_global/...), NOT a raw .edm4hep.root - point it at a
# CaloClouds-3 generated_showers.h5 (or the output of convert_to_DDML_format.py,
# which has the same layout). Positional args: <input> [output], output
# defaults to <input>_postprocessed.h5 next to the input. By default (no
# --no-project) that output IS the grid-projected file - no separate _grid.h5.
python postprocessing.py "$INPUT"

# Plotting scripts take one or more grid-projected h5 files (postprocessing.py's
# default output); --output-dir defaults to plots/ next to this script. The
# profile script's first arg is the reference (solid/gray "Geant4" line).
python plot_distributions_from_grid.py "$REFERENCE" "$REFERENCE1" "$REFERENCE2" "$REFERENCE3" #"$REFERENCE4" "$REFERENCE5"
python plot_shower_profiles_from_grid.py "$REFERENCE" "$REFERENCE1" "$REFERENCE2" "$REFERENCE3" #"$REFERENCE4" "$REFERENCE5"

# plot_cog_from_grid.py computes CoG in the LOCAL, shift-aligned frame (to
# match the paper's own CoG convention) - it does its own box cut + half-MIP
# merge internally, so it needs the RAW pre-postprocessing files, not the
# grid-projected (global frame) ones the other two scripts use above.
python plot_cog_from_grid.py \
    "${REFERENCE/_postprocessed/}" "${REFERENCE1/_postprocessed/}" \
    "${REFERENCE2/_postprocessed/}" "${REFERENCE3/_postprocessed/}" \
    --n-showers 5000
