#!/bin/bash
#
#

for file in f277_rot.fits f356_rot.fits f444_rot.fits
do
      echo "Launch Dawis on file ${file}"
      qsub qsub_dawis_jwst.sh -v n=${file},ncl=${file:0:-5}
      sleep 2
done
