#!/bin/bash
#
#

for f in f444w f277w f356w
do
      #file=jw02736001001_${f}_bkg_rot_crop_det_nosky.fits
      file=jw02736001001_${f}_bkg_rot_crop.fits
      echo "Launch Dawis on file $file"
      qsub qsub_dawis_jwst.sh -v n=${file},ncl=${file:0:-5},chan=long
      sleep 2
done

for f in f090w f150w f200w
do
      #file=jw02736001001_${f}_bkg_rot_crop_det_nosky.fits
      file=jw02736001001_${f}_bkg_rot_crop.fits
      echo "Launch Dawis on file $file"
      qsub qsub_dawis_jwst.sh -v n=${file},ncl=${file:0:-5},chan=short
      sleep 2
done
