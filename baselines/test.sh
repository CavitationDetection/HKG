#!/bin/bash
#SBATCH --job-name=Test
#SBATCH --error=err_Test.log
#SBATCH --output=out_Test.log
#SBATCH --nodes=1                                        
#SBATCH --partition=sleuths                      
#SBATCH --nodelist=geralt                
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=48   
#SBATCH --time=10000:00:00                         


srun python3 test.py

