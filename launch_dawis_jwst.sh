#!/bin/bash
#
#

#for f in f444w f277w f356w
#do
#      #file=jw02736001001_${f}_bkg_rot_crop_det_nosky.fits
#      file=jw02736001001_${f}_bkg_rot_crop_input.fits
#      echo "Launch Dawis on file $file"
#      qsub qsub_dawis_jwst.sh -v n=${file},ncl=${file:0:-5},chan=long
#      sleep 2
#done

for f in f090w f150w f200w
do
      #file=jw02736001001_${f}_bkg_rot_crop_det_nosky.fits # out12
      #file=jw02736001001_${f}_bkg_rot_crop_warp_det_nosky_input.fits # out13 - rebinned
      file=jw02736001001_${f}_bkg_rot_crop_warp_nobkg1_det_nosky_input.fits # out14 rm scattered light
      echo "Launch Dawis on file $file"
      echo "Launch Dawis on file $file"
      qsub qsub_dawis_jwst.sh -v n=${file},ncl=${file:0:-5},chan=long # Long here is just for dawis input file (rebinned short images)
      sleep 2
done
