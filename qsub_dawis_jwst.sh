#!/bin/bash
#PBS -o /home/ellien/JWST/logs/icl_JWST_${ncl}.out
#PBS -j oe
#PBS -N icl_JWST
#PBS -l nodes=1:ppn=1,walltime=47:59:00
#PSB -S /bin/bash

module load intelpython/3-2020.4
echo ${ncl}
python /home/ellien/JWST/JWST_scripts/dawis_JWST.py ${ncl}

exit 0
