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

for file in jw02736001001_f277w_bkg_rot_crop_input.fits \
            jw02736001001_f356w_bkg_rot_crop_input.fits \
            jw02736001001_f444w_bkg_rot_crop_input.fits \
            jw02736001001_f090w_bkg_rot_crop_warp_nobkg2.fits \
            jw02736001001_f150w_bkg_rot_crop_warp_nobkg2.fits \
            jw02736001001_f200w_bkg_rot_crop_warp_nobkg2.fits \
            
do
      #file=jw02736001001_${f}_bkg_rot_crop_det_nosky.fits # out12
      #file=jw02736001001_${f}_bkg_rot_crop_warp_det_nosky_input.fits # out13 - rebinned
      #file=jw02736001001_${f}_bkg_rot_crop_warp_nobkg1_det_nosky_input.fits # out14 rm scattered light
      #file=jw02736001001_${f}_bkg_rot_crop_warp_nobkg2.fits # out15 - slightly deeper run, new scattered light bkg, DAWIS ran on every filter again
      echo "Launch Dawis on file $file"
      #qsub qsub_dawis_jwst.sh -v n=${file},ncl=${file:0:-5},chan=long # Long here is just for dawis input file (rebinned short images)
      bash slurm_dawis.sh $file

done
