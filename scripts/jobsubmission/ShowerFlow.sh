#!/bin/bash
#SBATCH --time 5-00:00:00
#SBATCH --nodes 1
#SBATCH --partition maxgpu
#SBATCH --job-name SF_ori_6
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=henry.day-hall@desy.de
#SBATCH --output /gpfs/dust/maxwell/user/dayhallh/point-cloud-diffusion-logs/joblogs/%j.out      # terminal output
#SBATCH --error /gpfs/dust/maxwell/user/dayhallh/point-cloud-diffusion-logs/joblogs/%j.err
#SBATCH --constraint="GPUx1&A100"

module load maxwell mamba
. mamba-init
cd ~/training/point-cloud-diffusion
mamba activate /gpfs/dust/maxwell/user/dayhallh/envs/calogpu
gun_henry_path=\
"/gpfs/dust/maxwell/user/dayhallh/data/ILCsoftEvents/p22_th90_ph90_en10-100_joined"\
"/p22_th90_ph90_en10-100_seed{}_all_steps.hdf5"
anatolli_data=\
"/beegfs/desy/user/akorol/data/AngularShowers_RegularDetector/"\
"hdf5_for_CC/sim-E1261AT600AP180-180_file_{}slcio.hdf5"

python3 scripts/ShowerFlow.py caloclouds_3 dataset_path=$anatolli_data \
    n_dataset_files=88 shower_flow_version=original shower_flow_num_blocks=6
