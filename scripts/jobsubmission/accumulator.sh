#!/bin/bash -l
# Job will import a variable called NUM_SECTIONS and SECTION_N
# to use
#
# export NUM_SECTIONS=10
# export SECTION_N=0
# sbatch accumulator.sh --export=NUM_SECTIONS --export=SECTION_N
#
#SBATCH --job-name=accumulator_${NUM_SECTIONS}_${SECTION_N}
#SBATCH --time=0-01:00:00
#SBATCH --nodes=1
#SBATCH --partition=maxcpu

echo "Running accumulator with $NUM_SECTIONS sections and section number $SECTION_N"
# go to the right directory
cd /home/dayhallh/training/point-cloud-diffusion/scripts/jobsubmission
# activate the conda environment
source /beegfs/desy/user/dayhallh/miniconda3/etc/profile.d/conda.sh
conda activate calogpu
# run the python script
python3 run_accumulator.py $NUM_SECTIONS $SECTION_N
# exit
exit 0
