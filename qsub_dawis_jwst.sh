#!/bin/bash
#PBS -o /home/ellien/JWST/logs/icl_JWST_${ncl}.out
#PBS -j oe
#PBS -N icl_JWST
#PBS -l nodes=1:ppn=4,walltime=47:59:00
#PSB -S /bin/bash

#module load intelpython/3-2020.4
conda init bash
source /home/ellien/.bashrc
conda activate dawis

echo ${n}
python -W"ignore" /home/ellien/JWST/JWST_scripts/dawis_JWST_${chan}.py ${n}

exit 0
