#!/bin/bash
#SBATCH --time 2-00:00:00
#SBATCH --nodes 1
#SBATCH --partition maxgpu
#SBATCH --job-name discri
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=henry.day-hall@desy.de
#SBATCH --output /data/dust/user/dayhallh/point-cloud-diffusion-logs/joblogs/%j.out      # terminal output
#SBATCH --error /data/dust/user/dayhallh/point-cloud-diffusion-logs/joblogs/%j.err
#SBATCH --array=0-21

module load maxwell mamba
. mamba-init
cd ~/training/point-cloud-diffusion
mamba activate /data/dust/user/dayhallh/envs/calogpu

echo $SLURM_ARRAY_TASK_ID

python3 scripts/evaulation/discrimintor-showerflow.py $SLURM_ARRAY_TASK_ID
