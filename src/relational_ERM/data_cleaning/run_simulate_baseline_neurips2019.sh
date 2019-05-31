#!/bin/bash
#SBATCH --account=sml
#SBATCH --job-name=causal-spe-baseline
#SBATCH -c 1
#SBATCH --time=11:00:00
#SBATCH --mem-per-cpu=16gb

source ~/py3/bin/activate
python simulate_baseline_sbm_neurips2019.py
