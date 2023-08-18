#!/bin/bash
#PBS -o /home/ellien/JWST/logs/make_results.out
#PBS -j oe
#PBS -N maker
#PBS -l nodes=1:ppn=48,walltime=47:00:00
#PSB -S /bin/bash

module load intelpython/3-2020.4
python -W"ignore" /home/ellien/JWST/JWST_scripts/make_results_jwst.py


exit 0
