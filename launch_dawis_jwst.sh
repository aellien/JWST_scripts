#!/bin/bash
#
#

for f in f444w #f277w f356w 
do
      file=jw02736001001_${f}_bkg_rot_crop_det_nosky.fits
      echo "Launch Dawis on file $file"
      qsub qsub_dawis_jwst.sh -v n=${file},ncl=${file:0:-5}
      sleep 2
done
