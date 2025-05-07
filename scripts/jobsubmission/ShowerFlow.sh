#!/bin/bash
#SBATCH --time 5-00:00:00
#SBATCH --nodes 1
#SBATCH --partition maxgpu
#SBATCH --job-name Showerflow
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=henry.day-hall@desy.de
#SBATCH --output /data/dust/user/dayhallh/point-cloud-diffusion-logs/joblogs/%j.out      # terminal output
#SBATCH --error /data/dust/user/dayhallh/point-cloud-diffusion-logs/joblogs/%j.err
#SBATCH --constraint="GPUx1&A100"

module load maxwell mamba
. mamba-init
cd ~/training/point-cloud-diffusion
mamba activate /data/dust/user/dayhallh/envs/calogpu
#mamba activate /data/dust/user/dayhallh/envs/calogpu_cp313
gun_henry_path=\
"/data/dust/user/dayhallh/data/ILCsoftEvents/p22_th90_ph90_en10-100_joined"\
"/p22_th90_ph90_en10-100_seed{}_all_steps.hdf5"
anatolli_data=\
"/data/dust/user/akorol/data/AngularShowers_RegularDetector/"\
"hdf5_for_CC/sim-E1261AT600AP180-180_file_{}slcio.hdf5"
calchal_10k=\
"/data/dust/user/dayhallh/point-cloud-diffusion-data/10k_samples/"\
"10-90GeV_x36_grid_regular_524k_float32_10k.hdf5"
calchal=\
"/data/dust/user/akorol/data/CaloClouds/hdf5/high_granular_grid/train"\
"/10-90GeV_x36_grid_regular_524k_float32.hdf5"

config_name=caloclouds_2

dataset_path=$gun_henry_path
n_dataset_files=10
shower_flow_version=original
shower_flow_num_blocks=10
shower_flow_detailed_history=True
shower_flow_weight_decay=0.0

#config_name=caloclouds_3_simple_shower


#shower_flow_version=alt1
#shower_flow_num_blocks=2

echo python3 scripts/ShowerFlow.py $config_name dataset_path=$dataset_path \
    n_dataset_files=$n_dataset_files shower_flow_version=$shower_flow_version \
    shower_flow_num_blocks=$shower_flow_num_blocks \
    shower_flow_detailed_history=$shower_flow_detailed_history \
    shower_flow_weight_decay=$shower_flow_weight_decay
python3 scripts/ShowerFlow.py $config_name dataset_path=$dataset_path \
    n_dataset_files=$n_dataset_files shower_flow_version=$shower_flow_version \
    shower_flow_num_blocks=$shower_flow_num_blocks \
    shower_flow_detailed_history=$shower_flow_detailed_history \
    shower_flow_weight_decay=$shower_flow_weight_decay

