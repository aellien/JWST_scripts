#!/bin/bash
# conda activate gnuastro

for f in f090w f150w f200w
do
  #input=/home/ellien/JWST/data/jw02736001001_${f}_bkg_rot_crop_warp_nobkg1.fits
  input=/home/ellien/JWST/data/jw02736001001_${f}_bkg_rot_crop_warp_res2.fits
  bkg1=/home/ellien/JWST/data/jw02736001001_${f}_bkg_rot_crop_warp_bkg1.fits
  det=${input:0:-5}_det.fits
  nosky=${det:0:-5}_nosky.fits
  bkg2=${det:0:-5}_bkg2.fits
  combbkg=${det:0:-5}_combbkg.fits

  astnoisechisel $input --output=$det --qthresh=0.4 --dthresh=0.1  --kernel=/home/ellien/JWST/data/kernel.fits -h0 #--tilesize=30,30

  # Create detection mask
  astarithmetic $det -hINPUT-NO-SKY $det -hDETECTIONS nan where --output=${det:0:-5}_detmask.fits

  # Create input for DAWIS
  astfits $det --copy=INPUT-NO-SKY --output=$nosky --primaryimghdu
  astfits $nosky --copy=0 --output=${nosky:0:-5}_input.fits --primaryimghdu

  # Create combinned background map
  astfits $det --copy=SKY --output=$bkg2 --primaryimghdu
  astarithmetic $bkg2 $bkg1 + --output=$combbkg -h0 -h0

done
