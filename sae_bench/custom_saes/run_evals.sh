#!/bin/bash

#SBATCH -J run_evals                       # Job name
#SBATCH -A gts-pli77                     # Charge account
#SBATCH -N1 --gres=gpu:A100:1                # Number of nodes and GPUs requiredi
#SBATCH --mem-per-gpu=60G                    # Memory per gpu
#SBATCH -t1:00:00                            # Duration of the job (Ex: 15 mins)
#SBATCH -qinferno                            # QOS name
#SBATCH -o results/run_evals.out          # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL           # Mail preferences
#SBATCH --mail-user=sgovil9@gatech.edu           # e-mail address for notifications

module load anaconda3
conda activate wta
python run_all_evals_dictionary_learning_saes.py