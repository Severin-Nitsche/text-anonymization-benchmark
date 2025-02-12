#!/usr/bin/env zsh

#SBATCH --job-name=train-rfc
#SBATCH --output=tab_%J.txt
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1

#SBATCH --time=00:20:00
#SBATCH --mail-user=severin.nitsche@rwth-aachen.de
#SBATCH --mail-type=END

ml load GCCcore/.12.2.0
ml load Python/3.10.8
cd /home/ll464721/text-anonymization-benchmark
echo "JOB ID: $SLURM_JOB_ID"
source tab-venv/bin/activate
python rfc_experiments/train_model.py
