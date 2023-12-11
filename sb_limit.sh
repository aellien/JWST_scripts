#!/bin/bash
# conda activate gnuastro

nsigma=3
zeropoint=22.5
areaarcsec2=100

for f in f090w f150w f200w
do
echo $f
input=/home/ellien/JWST/data/jw02736001001_${f}_bkg_rot_crop_warp_nobkg2.fits
oim=/home/ellien/JWST/data/jw02736001001_${f}_bkg.fits
detmask=/home/ellien/JWST/data/jw02736001001_${f}_bkg_rot_crop_warp_det_detmask.fits
std=$(aststatistics $input --sigclip-std -h0)
pixarcsec2=$(astfits $detmask --pixelscale --quiet \
                         | awk '{print $3*3600*3600}')
pixar_sr=$(astfits $oim --keyvalue=PIXAR_SR --hdu=0 --quiet )
zeropoint=$(astarithmetic -h1 $pixar_sr float64 log10 -2.5 x -6.1 + --quiet)
astarithmetic --quiet $nsigma $std x $areaarcsec2 $pixarcsec2 x sqrt / $zeropoint counts-to-mag
done
