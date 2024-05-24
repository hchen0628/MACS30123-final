#!/bin/bash

#SBATCH --job-name=WGAN_Training         # Job name: WGAN_Training
#SBATCH --output=WGAN_Training.out       # Standard output redirected to WGAN_Training.out
#SBATCH --error=WGAN_Training.err        # Standard error redirected to WGAN_Training.err
#SBATCH --nodes=1                        # Request 1 node
#SBATCH --ntasks-per-node=1              # Request 1 task per node
#SBATCH --partition=gpu                  # Specify the GPU partition
#SBATCH --gres=gpu:1                     # Request 1 GPU
#SBATCH --account=macs30123              # Specify the account for job charging
#SBATCH --mem=30G                        # Allocate 30GB memory per CPU

module load cuda python

python3 ./WGAN_trial_v4_tune_May.py >> WGAN_Training.out