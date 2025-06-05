#!/bin/bash
#SBATCH --time 1:00:00
#SBATCH --nodes 1
#SBATCH --partition maxgpu
#SBATCH --job-name speed
#SBATCH --output /data/dust/user/dayhallh/point-cloud-diffusion-logs/joblogs/speed%j.out      # terminal output
#SBATCH --error /data/dust/user/dayhallh/point-cloud-diffusion-logs/joblogs/speed%j.err
#SBATCH --constraint="GPUx1&A100"
#SBATCH --array=2-125

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


python3 scripts/evaulation/speed-showerflow.py $SLURM_ARRAY_TASK_ID
