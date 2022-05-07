#!/bin/bash 
#SBATCH --nodes=1                        # requests 3 compute servers
#SBATCH --ntasks-per-node=48             # runs 2 tasks on each server
#SBATCH --cpus-per-task=1               # uses 1 compute core per task
#SBATCH --time=60:00:00
#SBATCH --mem=16GB
#SBATCH --job-name=HPCscript
#SBATCH --output=HPCscript.out

module purge
module load python/intel/3.8.6

python ./frac.py
