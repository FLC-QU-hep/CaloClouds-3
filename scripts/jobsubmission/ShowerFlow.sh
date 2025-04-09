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
#SBATCH --array=1-9

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

dataset_path=$anatolli_data
n_dataset_files=88
shower_flow_version=original
shower_flow_num_blocks=10
diffusion_precision=float32
shower_flow_detailed_history=True
shower_flow_weight_decay=0.0


varient=$SLURM_ARRAY_TASK_ID
if [ $varient -eq 1 ]; then
    echo "No change"
elif [ $varient -eq 2 ]; then
    shower_flow_version=alt1
elif [ $varient -eq 3 ]; then
    shower_flow_version=alt1
    shower_flow_num_blocks=2
elif [ $varient -eq 4 ]; then
    shower_flow_weight_decay=0.0001
elif [ $varient -eq 5 ]; then
    shower_flow_weight_decay=0.0001
    shower_flow_version=alt1
elif [ $varient -eq 6 ]; then
    shower_flow_weight_decay=0.0001
    shower_flow_version=alt1
    shower_flow_num_blocks=2
elif [ $varient -eq 7 ]; then
    shower_flow_weight_decay=0.1
elif [ $varient -eq 8 ]; then
    shower_flow_weight_decay=0.1
    shower_flow_version=alt1
elif [ $varient -eq 9 ]; then
    shower_flow_weight_decay=0.1
    shower_flow_version=alt1
    shower_flow_num_blocks=2
fi

echo python3 scripts/ShowerFlow.py caloclouds_3 dataset_path=$dataset_path \
    n_dataset_files=$n_dataset_files shower_flow_version=$shower_flow_version \
    shower_flow_num_blocks=$shower_flow_num_blocks diffusion_precision=$diffusion_precision \
    shower_flow_detailed_history=$shower_flow_detailed_history \
    shower_flow_weight_decay=$shower_flow_weight_decay
python3 scripts/ShowerFlow.py caloclouds_3 dataset_path=$dataset_path \
    n_dataset_files=$n_dataset_files shower_flow_version=$shower_flow_version \
    shower_flow_num_blocks=$shower_flow_num_blocks diffusion_precision=$diffusion_precision \
    shower_flow_detailed_history=$shower_flow_detailed_history \
    shower_flow_weight_decay=$shower_flow_weight_decay
