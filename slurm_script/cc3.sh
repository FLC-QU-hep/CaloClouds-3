#!/bin/bash

cd /eos/user/m/mamozzan/CaloClouds-3/ || exit
# unset PYTHONPATH
# venv must live off EOS (EOS's FUSE mount doesn't reliably support mmap,
# which dlopen() needs for compiled extensions -> intermittent Bus error)
source caloclouds3/bin/activate

# Link your config file to configs.py
cd pointcloud/
rm configs.py
# ln -s config_varients/caloclouds_3_S2P_withincell.py configs.py
# ln -s config_varients/caloclouds_3_S2P_subcell.py configs.py
# ln -s config_varients/caloclouds_3_S2P_subcell_6kcut.py configs.py
# ln -s config_varients/caloclouds_3_S2P_hdbscan_ms8_mcs40.py configs.py
ln -s config_varients/caloclouds_3_S2P_hdbscan_ms3_mcs10.py configs.py
# ln -s config_varients/caloclouds_3_S2P_hdbscan_ms12_mcs12.py configs.py
# ln -s config_varients/caloclouds_3_S2P_hdbscan_ms40_mcs40.py configs.py
# ln -s config_varients/caloclouds_3_S2P_steps.py configs.py
cd ..

python scripts/training/diffusion.py

# student
# python scripts/training/cd.py

# to check the genrated showers go inside --> CaloClouds-3/scripts/plotting/inferepy.ipynb

# to run generation
export data=subcell_6kcut #option [withincell, subcell, hdbscan_ms8_mcs40, hdbscan_ms3_mcs10, hdbscan_ms12_mcs12, hdbscan_ms40_mcs40]

if [[ "$data" == "withincell" ]]; then
    export log_dir=merge_within_cell_2026_07_10__15_01_37
    export file="../step2point/outputs/cc3input_merge_within_cell/input_cc3_file_0.h5"
elif [[ "$data" == "subcell" ]]; then
    export log_dir=merge_within_regular_subcell_2026_07_20__15_35_38
    export file="../step2point/outputs/cc3input_merge_within_regular_subcell/input_cc3_file_0.h5"
elif [[ "$data" == "subcell_6kcut" ]]; then
    export log_dir=merge_within_regular_subcell_6kcut_2026_07_24__13_10_06
    export file="../step2point/outputs/cc3input_merge_within_regular_subcell_6kcut/input_cc3_file_0.h5"
elif [[ "$data" == "hdbscan_ms8_mcs40" ]]; then
    export log_dir=hdbscan_ms8_mcs40_2026_07_20__18_29_36 # ms 8 mcs 40
    export file="../step2point/outputs/cc3input_hdbscan_ms8_mcs40/input_cc3_file_0.h5"
elif [[ "$data" == "hdbscan_ms3_mcs10" ]]; then
    export log_dir=hdbscan_ms3_mcs10_2026_07_10__15_46_43 # ms 3 mcs 10
    export file="../step2point/outputs/cc3input_hdbscan_ms3_mcs10/input_cc3_file_0.h5"
elif [[ "$data" == "hdbscan_ms12_mcs12" ]]; then
    export log_dir=hdbscan_ms12_mcs12_2026_07_21__10_44_53 # ms 12 mcs 12
    export file="../step2point/outputs/cc3input_hdbscan_ms12_mcs12/input_cc3_file_0.h5" 
elif [[ "$data" == "hdbscan_ms40_mcs40_2026_06_23__11_24_02" ]]; then
    export log_dir=hdbscan_ms40_mcs40_2026_06_23__11_24_02 # ms 40 mcs 40
    export file="../step2point/outputs/cc3input_hdbscan_ms40_mcs40/input_cc3_file_0.h5"
fi 

export n=15000
python scripts/generate_showers.py \
    --log_dir log_dir/$log_dir \
    --n_showers $n \
    --chunk_size 2000 \
    --gen_batch_size 64 \
    --cond_file $file \
    --energy_units MeV

python scripts/plotting/compare_generated_vs_input_cc3.py generated_showers/$log_dir/generated_showers_$n/generated_showers.h5 --data-used $data

# ---- occupancy poly-fit calibration (alternative to the flat _OCCUPANCY_SCALE_N) ----
# only a marginal improvement over the flat scale for subcell (mid-range slightly
# better, high-energy slightly worse, plus a handful of zero-hit showers at low
# energy from the cubic fit's extrapolation) -- kept here for comparison, not
# because it's clearly better.
python scripts/training/calculate_coef.py \
    --log_dir log_dir/$log_dir \
    --n_showers $n \
    --degree 3


export n=5000
python scripts/generate_showers.py \
    --log_dir log_dir/$log_dir \
    --n_showers $n \
    --chunk_size 2000 \
    --gen_batch_size 64 \
    --cond_file $file \
    --energy_units MeV \
    --occupancy_fit poly \
    --output_file generated_showers_poly.h5

python scripts/plotting/compare_generated_vs_input_cc3.py generated_showers/$log_dir/generated_showers_$n/generated_showers_poly.h5 \
    --data-used $data \
    --out-dir generated_showers/$log_dir/generated_showers_$n/poly_fit_plots

# --------------- NOT NEEDED: NOW WE PROJECT BACK WITH DDML !  ---------------
# now it is also wrong cause i modified generated showers to give me directly the input for DDML

# for plotting, you can use the same script as the one for generating the inputs 
python ../step2point/martina_test/plot_check_cc3_format.py generated_showers/$log_dir/generated_showers_$n/generated_showers.h5

# for converting to input for step2point
python ../step2point/martina_test/convert_from_cc3_format.py ../CaloClouds-3/generated_showers/$log_dir/generated_showers_$n/generated_showers.h5

# apply projection inside step2point
# better to open a new terminal
cd /eos/user/m/mamozzan/step2point/
source /cvmfs/sw.hsf.org/key4hep/setup.sh
source .venv-key4hep/bin/activate
export algo=merge_within_cell # always since i want the projection back into the real cells
python examples/run_step2point_pipeline.py \
      --input ../CaloClouds-3/generated_showers/$log_dir/generated_showers_$n/generated_showers.step2point.h5 \
      --algorithm "$algo" \
      --output ../CaloClouds-3/generated_showers/$log_dir/generated_showers_$n \
      --collections EcalBarrelCollection


python ../CaloClouds-3/scripts/plotting/compare_pipeline_outputs.py --output-dir ../CaloClouds-3/generated_showers/comparison_plots
exit