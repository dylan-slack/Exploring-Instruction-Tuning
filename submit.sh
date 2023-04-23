#!/bin/bash
#SBATCH --job-name=instruction-tuning
#SBATCH --output=results.txt
#SBATCH --time=365-00:00
#SBATCH --partition=ava_s.p
#SBATCH --nodelist=ava-s4
#SBATCH --cpus-per-task=32
#SBATCH --gpus=6
#SBATCH --mem=100GB

python train.py -m "google/flan-t5-large" --train -b 2 --accum 8 --overwrite-cache
