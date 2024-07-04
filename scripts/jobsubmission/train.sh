#!/bin/bash

#SBATCH --time 7-00:00:00
#SBATCH --nodes 1
#SBATCH --output ./joblog/%j.out      # terminal output
#SBATCH --error ./joblog/%j.err

#SBATCH --partition maxgpu
#SBATCH --constraint="GPUx1&A100"
#SBATCH --mail-type=END
#SBATCH --mail-user lorenzo.valente@desy.de
#SBATCH --job-name train-nDA15%

bash
source ~/.bashrc

conda activate pointcloud_env

# cd /beegfs/desy/user/akorol/projects/point-cloud
cd /home/valentel/projects/point-cloud-diffusion/scripts

python main.py
# python cd.py
# python timing.py

exit
