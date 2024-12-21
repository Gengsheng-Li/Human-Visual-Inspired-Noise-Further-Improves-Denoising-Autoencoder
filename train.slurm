#!/bin/bash
#SBATCH -p gpupar
#SBATCH --cpus-per-task 4
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --nodelist gpu02
#SBATCH -o ./slurm_log/slurm%A_out.log
#SBATCH -e ./slurm_log/slurm%A_err.log

cd /data0/user/gsli/CNS/Lab1/

python denoise_autoencoder.py