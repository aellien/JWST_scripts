#!/bin/bash
#SBATCH --job-name=synth_jwst
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --time=48:00:00
#SBATCH --output /n03data/ellien/JWST/logs/%x.%j.out 
#SBATCH --error  /n03data/ellien/JWST/logs/%x.%j.err

source /home/ellien/.bashrc
conda activate dawis
ray start --head --port=6379 --block --verbose  &
echo 'BONJOUR BONJOUR BONJOUR BONJOUR'
python -u -W"ignore" /home/ellien/JWST/JWST_scripts/make_results_jwst.py
exit 0