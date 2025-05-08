#!/bin/bash


#SBATCH --job-name=2024_ukbb_estim
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --array=0-78
#SBATCH --output=/scratch/jabecass/judith_abecassis/log/2024_ukbb_estimation/20240105_ukbb_estim_%A_%a.out
#SBATCH --error=/scratch/jabecass/judith_abecassis/log/2024_ukbb_estimation/20240105_ukbb_estim_%A_%a.err

# Specify the path to the config file
INPUT_FILE=/scratch/jabecass/20250105_ukbb_params.csv

VALUES=({1..79})
THISJOBVALUE=${VALUES[$SLURM_ARRAY_TASK_ID]}
i=1
for PARAM in pca_mental_health_lifestyle_ncp mri_data_type treatment_name
do i=$((i+1)) ; eval $PARAM=$(grep "^$THISJOBVALUE," $INPUT_FILE | cut -d',' -f$i) ; done



/scratch/jabecass/judith_abecassis/src/ukbb_application/get_ukbb_estimation.py $pca_mental_health_lifestyle_ncp $mri_data_type $treatment_name

