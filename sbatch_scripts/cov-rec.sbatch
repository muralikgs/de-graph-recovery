#!/bin/bash
#SBATCH -JRunSythBench                          # Job name
#SBATCH --account=gts-ff14                      # charge account
#SBATCH -N1 --gres=gpu:V100:1                   # Number of nodes and request GPU
#SBATCH --mem-per-cpu=16G                       # Memory per core
#SBATCH -t01:00:00                              # Duration of the job (Ex: 15 mins)
#SBATCH -qinferno                               # QOS Name
#SBATCH -ocov-rec-%j.out                        # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail preferences
#SBATCH --mail-user=msethuraman7@gatech.edu     # E-mail address for notifications

cd $SLURM_SUBMIT_DIR

source ~/.bashrc
conda activate cov-rec 

cd ../

srun python recover_covariance_y_cov.py $@