#!/bin/bash
###
 # @Description: 
 # @Author: Yu Sha
 # @Date: 2021-08-17 09:57:39
 # @LastEditors: Yu Sha
 # @LastEditTime: 2021-08-17 14:52:48
### 

#SBATCH --job-name=Test
#SBATCH --error=err_Test.log
#SBATCH --output=out_Test.log
#SBATCH --reservation deepthinkers
#SBATCH --nodes=1                                         # set the number of nodes
#SBATCH --partition=sleuths                      # set partition
#SBATCH --nodelist=tussock                  # set node (turbine,vane,speedboat,jetski,scuderi,tussock#SBATCH --gres=gpu:1 #SBATCH --reservation deepthinkers)  
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12     
#SBATCH --time=10000:00:00                          # run time of task


srun python3 test.py

