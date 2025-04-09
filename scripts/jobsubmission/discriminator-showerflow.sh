#!/bin/bash
#SBATCH --time 3-00:00:00
#SBATCH --nodes 1
#SBATCH --partition maxcpu
#SBATCH --job-name discri
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=henry.day-hall@desy.de
#SBATCH --output /data/dust/user/dayhallh/point-cloud-diffusion-logs/joblogs/discr%j.out      # terminal output
#SBATCH --error /data/dust/user/dayhallh/point-cloud-diffusion-logs/joblogs/discri%j.err
#SBATCH --array=1-17

module load maxwell mamba
. mamba-init
cd ~/training/point-cloud-diffusion
mamba activate /data/dust/user/dayhallh/envs/calogpu

echo $SLURM_ARRAY_TASK_ID

python3 scripts/evaulation/discrimintor-showerflow.py $SLURM_ARRAY_TASK_ID
