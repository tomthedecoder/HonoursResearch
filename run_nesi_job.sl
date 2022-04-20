#!/bin/bash -e

#SBATCH --job-name=honours_research
#SBATCH --time=16:00:00
#SBATCH --mem=4096MB
#SBATCH --cpus-per-task=50

ml Python/3.9.5-gimkl-2020a
ml FFmpeg/4.2.2-GCCcore-9.2.0
pipenv run python main.py
