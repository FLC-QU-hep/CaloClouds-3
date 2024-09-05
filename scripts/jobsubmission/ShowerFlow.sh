#!/bin/bash
#SBATCH --time 72:00:00
#SBATCH --nodes 1
#SBATCH --partition maxgpu
#SBATCH --job-name SF_orig_10
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=henry.day-hall@desy.de
#SBATCH --output /gpfs/dust/maxwell/user/dayhallh/point-cloud-diffusion-logs/joblogs/%j.out      # terminal output
#SBATCH --error /gpfs/dust/maxwell/user/dayhallh/point-cloud-diffusion-logs/joblogs/%j.err

module load maxwell mamba
. mamba-init
cd ~/training/point-cloud-diffusion
mamba activate /gpfs/dust/maxwell/user/dayhallh/envs/calogpu
python3 scripts/ShowerFlow.py original 10
