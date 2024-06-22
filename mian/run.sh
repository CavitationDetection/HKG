#!/bin/bash
#SBATCH --job-name=HL
#SBATCH --error=err_HL_resnet34_test.log
#SBATCH --output=out_HL_resnet34_test.log
#SBATCH --nodes=1                                         
#SBATCH --partition=sleuths                      
#SBATCH --nodelist=geralt                 
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=48     
#SBATCH --time=10000:00:00                         

srun python3 train.py

