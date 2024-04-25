#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 15:14:03 2024

@author: Amaël Ellien

Peut être essayer de centraliser ici les différents codes où j'extrait les SEDs
des différents composants; une partie (ICL, total gal, etc) est effectuée au
sein du code 'make_results.py' durant la phase de synthèse des images et intégré
au catalogue de mesures, et une autre est effectuée dans 'plot_pub.py' avec 
quelques fonctions qui produisent les graphes de SED. Ici j'ai besoin de faire
plus de tests sur les SEDs, et d'extraire notemment les SEDs des galaxies indi-
viduelles, ce qui demande une méthodologie différente des quantités intégrées
sur régions spatiales arbitraires utilisée jusque là. On utilise pas les props
des atomes ici, et on travaille directement sur les pixels pour séparer les gals.
"""

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Modules
import numpy as np
import os
import matplotlib.pyplot as plt
import pyregion as pyr
import pandas as pd
import photutils as phut
import astropy.units as u
from astropy.coordinates import SkyCoord
from skimage.measure import label, regionprops
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import *

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def deblend_synth_images(idc):
    '''
    Deblending step
    '''
    
    sdc = np.copy(idc)
    sdc[sdc > 0] = 1
    master_sup = np.prod(sdc, axis = 0)
    lab, labc = label(master_sup, return_num = True)
    props_all_bands = []
    new_props = []
    
    for z, band in enumerate(idc):
        props_band = regionprops(lab, band)
    
        print('before segmentation', len(props_band))
        for i, reg in enumerate(props_band):
            x_min, y_min, x_max, y_max = reg.bbox
            
            new_reg, lab, labc = deblend_region(reg, image = band, 
                                             lab = lab, labc = labc)
            
            # Update region
            if new_reg:
                props_band.pop(i)
                for r in new_reg:
                    new_props.append(r)
        
        #import pdb;pdb.set_trace()
        for r in new_props:
            props_band.append(r)
        labc = len(props_band)
        print('after', labc)
        props_all_bands.append(props_band)
        
        
    return master_sup, lab, labc, props_all_bands

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def plot_RGB_fullfield(dc, artist_list):
    plt.figure()
    b = dc[0] + dc [1] # f090w + f150w
    b *= master_sup*255.0/b.max()
    g = dc[2] + dc [3] # f200w + f277w
    g *= master_sup*255.0/g.max()
    r = dc[4] + dc [5] # f356w + f444w
    r *= master_sup*255.0/r.max()
    image = make_lupton_rgb(r, g, b, Q = 10, stretch = 0.5)
    for a in artist_list:
        #a.set_marker('x')
        a.set_markeredgecolor('blue')
        a.set_markersize('2')
        plt.gca().add_artist(a)
        
    plt.imshow(image, origin= 'lower')
    plt.savefig('/home/aellien/JWST/plots/RGB_gals_test.png', format = 'png', dpi = 1000)

def plot_master_sup(master_sup, artist_list):
    plt.figure()
    plt.imshow(master_sup, origin= 'lower')
    for a in artist_list:
        a.set_marker('x')
        a.set_markeredgecolor('white')
        a.set_markersize('4')
        plt.gca().add_artist(a)
    plt.show()
    
def plot_coords(cat_gal, props_skimage):
    plt.figure()
    for ygal, xgal in cat_gal:
        plt.plot(xgal, ygal, 'b+')
    for band in props_skimage:
        for reg in band:
            xco, yco = reg.centroid
            plt.plot(xco, yco, 'r+')
    plt.show()
    plt.savefig('/home/aellien/JWST/plots/coords_test.png', format = 'png', dpi = 1000)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if __name__=='__main__':
    '''
    Multiband deblending
    '''
    # Path list & variables
    working_dir = '/home/aellien/JWST/wavelets/out15/'
    scripting_dir = '/home/aellien/JWST/JWST_scripts'
    path_data = '/home/aellien/JWST/data'
    nfdl = [ {'nf':'jw02736001001_f090w_bkg_rot_crop_warp_nobkg2.synth.gal.bcgwavsizesepmask_005_080.fits', 'chan':'short', 'pix_scale':0.063, 'pixar_sr':9.31E-14}, \
            {'nf':'jw02736001001_f150w_bkg_rot_crop_warp_nobkg2.synth.gal.bcgwavsizesepmask_005_080.fits', 'chan':'short', 'pix_scale':0.063, 'pixar_sr':9.31E-14}, \
            {'nf':'jw02736001001_f200w_bkg_rot_crop_warp_nobkg2.synth.gal.bcgwavsizesepmask_005_080.fits', 'chan':'short', 'pix_scale':0.063, 'pixar_sr':9.31E-14}, \
            {'nf':'jw02736001001_f277w_bkg_rot_crop_input.synth.gal.bcgwavsizesepmask_005_080.fits', 'chan':'long', 'pix_scale':0.063, 'pixar_sr':9.31E-14}, \
            {'nf':'jw02736001001_f356w_bkg_rot_crop_input.synth.gal.bcgwavsizesepmask_005_080.fits', 'chan':'long', 'pix_scale':0.063, 'pixar_sr':9.31E-14}, \
            {'nf':'jw02736001001_f444w_bkg_rot_crop_input.synth.gal.bcgwavsizesepmask_005_080.fits', 'chan':'long', 'pix_scale':0.063, 'pixar_sr':9.31E-14},]
    iml = []
    rc_pix = 5

    # Catalogue galaxies
    nf = 'mahler_noirot_merged_member_gal_ra_dec_pix_long.reg'
    r = pyr.open(os.path.join(path_data, nf))
    patch_list, artist_list = r.get_mpl_patches_texts()
    print(len(artist_list))
    
    rgal = pyr.open(os.path.join(path_data, 'mahler_noirot_merged_member_gal_ra_dec_pix_long.reg'))
    cat_gal = []
    for gal in rgal:
        cat_gal.append(gal.coord_list)
    cat_gal = np.array(cat_gal)

    # Read 6 filters and create datacube
    for nfd in nfdl:
        print(nfd['nf'])
        fnp = os.path.join(working_dir, nfd['nf'])
        hdu = fits.open(fnp)
        im = hdu[0].data
        head = hdu[0].header
        w = WCS(head)
        iml.append(im[:2045,:2045]) # short channel larger by 1 pixel.

    dc = np.array(iml)
    finder = phut.segmentation.SourceFinder(npixels = 10, progress_bar = True )
    segment_map = finder(im, 0)
    
    norm = ImageNormalize(im, interval = AsymmetricPercentileInterval(40, 99.5), stretch = LogStretch())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 12.5))
    ax1.imshow(im, origin='lower', cmap='Greys_r', norm=norm)
    ax1.set_title('Background-subtracted Data')
    ax2.imshow(segment_map, origin='lower', cmap=segment_map.cmap,
               interpolation='nearest')
    ax2.set_title('Segmentation Image')
    
    cat = phut.segmentation.SourceCatalog(im, segment_map)
    

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 12.5))
    ax1.imshow(im, origin='lower', cmap='Greys_r', norm=norm)
    ax1.set_title('Data')
    ax2.imshow(segment_map, origin='lower', cmap=segment_map.cmap, interpolation='nearest')
    ax2.set_title('Segmentation Image')
    cat.plot_kron_apertures(ax=ax1, color='yellow', lw=1.5)
    cat.plot_kron_apertures(ax=ax2, color='white', lw=1.5)
    
    
        
    
    # Multiband deblending
    #master_sup, lab, labc, props_skimage = deblend_synth_images(dc)       
    
    
    
    
 