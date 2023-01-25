#!/bin/bash
#SBATCH --time 0-10:00:00
#SBATCH --nodes 1
#SBATCH --partition maxgpu
#SBATCH --constraint="GPUx1&A100"
#SBATCH --job-name point-cloud-training
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=akorol98.17@gmail.com

bash
source ~/.bashrc
conda activate A100-torch

# conda activate py36

cd /beegfs/desy/user/akorol/projects/point-cloud

python main.py

exit
