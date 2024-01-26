#!/bin/bash
#SBATCH --time 7-00:00:00
#SBATCH --nodes 1
#SBATCH --partition maxcpu
#SBATCH --job-name CCtraining
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=erik.buhmann@desy.de
#SBATCH --output ./joblog/%j.out      # terminal output
#SBATCH --error ./joblog/%j.err
##SBATCH --constraint="GPUx1&A100"
#SBATCH --nodelist="max-wn063"         # you can select specific nodes, if necessary

bash
source ~/.bashrc

# conda activate A100-torch
# conda activate py36
conda activate torch_113

# cd /beegfs/desy/user/akorol/projects/point-cloud
cd /home/buhmae/6_PointCloudDiffusion

# python main.py
# python cd.py
python timing.py

exit
