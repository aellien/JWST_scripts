#!/bin/bash
# conda activate gnuastro

size_short=4151
size_long=2045

for f in f090w f150w f200w
do

input=/home/ellien/JWST/data/jw02736001001_${f}_bkg.fits
rot=${input:0:-5}_rot.fits
crop=${rot:0:-5}_crop.fits
warp=${crop:0:-5}_warp.fits
det=${warp:0:-5}_det.fits
quantcheck=${crop:0:-5}_quantcheck.fits
nosky=${det:0:-5}_nosky.fits

# Rotation of image (hand made)
#astwarp --rotate=35 -h0 $input\
#        --output=$rot
#
## Crop image (hand made with box region)
#astcrop --mode=wcs --center=110.82744,-73.453903 \
#        --width=0.03577284,0.03577284 \
#        -h1 \
#        --output=$crop \
#          $rot
#

# /!\ For short channels only!!!!
astwarp $crop --scale=1./2.02983 --output=${warp}

## Noise Chisel to model sky bkg + segmentation map (compare to DAWIS output)
#astnoisechisel $crop --checkqthresh --qthresh=0.05 --dthresh=0.1 --output=$quantcheck # test f090w
#astnoisechisel $crop --output=$det --dthresh=0.1  --kernel=/home/ellien/JWST/data/kernel.fits
astnoisechisel ${warp} --output=$det --qthresh=0.3 --tilesize=25,25 --dthresh=0.1  --kernel=/home/ellien/JWST/data/kernel.fits # test f090w

## /!\ Remove the previous files before running the commands or else it appends the HDUs one after the other...
rm /home/ellien/JWST/data/*${f}*det_*
astarithmetic $det -hINPUT-NO-SKY $det -hDETECTIONS nan where --output=${det:0:-5}_detmask.fits
astarithmetic $det -hINPUT-NO-SKY $det -hDETECTIONS not nan where --output=${det:0:-5}_skymask.fits
astfits $det --copy=INPUT-NO-SKY --output=$nosky --primaryimghdu
astfits $nosky --copy=0 --output=${nosky:0:-5}_input.fits --primaryimghdu
astfits $crop --copy=1 --output=${crop:0:-5}_input.fits --primaryimghdu
done
