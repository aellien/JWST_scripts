#!/usr/bin/env bash
#!/bin/bash
#SBATCH --job-name=jwst
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --output /n03data/ellien/JWST/logs/synth.%j.out
#SBATCH --error  /n03data/ellien/JWST/logs/synth.%j.err

source /home/ellien/.bashrc
conda activate dawis
python -W"ignore" /home/ellien/JWST/JWST_scripts/synthesis/synthesis.py $@

exit 0
