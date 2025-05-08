#!/bin/bash


#SBATCH --job-name=2025_simu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --array=0-1000%300
#SBATCH --output=/scratch/jabecass/log/2025_simu/20250221_simu_test_%A_%a.out
#SBATCH --error=/scratch/jabecass/log/2025_simu/20250221_simu_test_%A_%a.err

# Specify the path to the config file
INPUT_FILE=/scratch/jabecass/20250221_simu2025.csv

VALUES=({10001..11000})
THISJOBVALUE=${VALUES[$SLURM_ARRAY_TASK_ID]}
i=1
for PARAM in folderpath
do i=$((i+1)) ; eval $PARAM=$(grep "^$THISJOBVALUE," $INPUT_FILE | cut -d',' -f$i) ; done



/scratch/jabecass/judith_abecassis/src/2025_test_simulation_settings.py $folderpath

