#!/bin/bash
#SBATCH --job-name=ResNet18_bn
#SBATCH --error=err_ResNet18.log
#SBATCH --output=out_ResNet18.log
#SBATCH --nodes=1                                       
#SBATCH --partition=sleuths                     
#SBATCH --nodelist=geralt                  
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=48     
#SBATCH --time=10000:00:00                         


srun python3 main.py

