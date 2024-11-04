#!/bin/bash
#SBATCH --time 2-00:00:00
#SBATCH --nodes 1
#SBATCH --partition maxgpu
#SBATCH --job-name discri
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=henry.day-hall@desy.de
#SBATCH --output /gpfs/dust/maxwell/user/dayhallh/point-cloud-diffusion-logs/joblogs/%j.out      # terminal output
#SBATCH --error /gpfs/dust/maxwell/user/dayhallh/point-cloud-diffusion-logs/joblogs/%j.err
#SBATCH --array=0-21

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

echo $SLURM_ARRAY_TASK_ID

python3 scripts/evaulation/discrimintor-showerflow.py $SLURM_ARRAY_TASK_ID
