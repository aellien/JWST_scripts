#!/bin/bash
#
#

for file in f277_rot.fits f356_rot.fits f444_rot.fits
do
      echo "Launch Dawis on file ${file}"
      qsub qsub_dawis_jwst.sh -v ncl=${n:0:-5}
      sleep 2
done
