#!/usr/bin/env bash
#SBATCH --job-name=rabit 
#SBATCH --partition=seq 
#SBATCH --time=0:1:0 
#SBATCH --output /n03data/ellien/JWST/logs/%x.%j.out 
#SBATCH --error  /n03data/ellien/JWST/logs/%x.%j.err

source /home/ellien/.bashrc
conda activate dawis

python -W"ignore" /home/ellien/JWST/JWST_scripts/dawis_JWST_long.py jw02736001001_f444w_bkg_rot_crop_input.fits