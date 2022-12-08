#!/bin/bash
#PBS -o /home/ellien/JWST/logs/icl_JWST_${ncl}.out
#PBS -j oe
#PBS -N icl_JWST
#PBS -l nodes=1:ppn=4,walltime=47:59:00
#PSB -S /bin/bash

module load intelpython/3-2020.4
echo ${n}
python /home/ellien/JWST/JWST_scripts/dawis_JWST.py ${n}

exit 0
