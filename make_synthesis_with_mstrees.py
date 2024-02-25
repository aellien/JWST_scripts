#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 10:33:36 2023

@author: Amaël Ellien
"""

import sys
import dawis as d
import glob as glob
import os
import numpy as np
import pyregion as pyr
from power_ratio import *
import random
import pandas as pd
import ray
from astropy.io import fits
from astropy.visualization import *
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.morphology import binary_dilation
from scipy.stats import kurtosis

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def synthesis_with_mstrees( nfp, gamma, lvl_sep_big, xs, ys, n_levels, mscoim, mscell, mscbcg, mscsedl, R, cat_gal, rc_pix, N_err, per_err, Jy_lim, rm_gamma_for_big = True, kurt_filt = True, plot_vignet = False, write_fits = True, measure_PR = False ):
    
    # path, list & variables
    icl = np.zeros( (xs, ys) )
    gal = np.zeros( (xs, ys) )
    im_art = np.zeros( (xs, ys) )
    im_unclass = np.zeros( (xs, ys) )

    icl_al = []
    gal_al = []
    noticl_al = []
    unclass_al = []

    #%
    at_test = []
    #%

    xc = xs / 2.
    yc = ys / 2.

    # Read atoms
    ol, itl = read_image_atoms( nfp, verbose = False )

    # Kurtosis + ICL
    for j, o in enumerate(ol):

        x_min, y_min, x_max, y_max = o.bbox
        sx = x_max - x_min
        sy = y_max - y_min
        itm = itl[j].interscale_maximum
        xco = itm.x_max
        yco = itm.y_max
        lvlo = o.level
        
    return None    

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if __name__ == '__main__':

    # Paths, lists & variables
    path_data = '/home/aellien/JWST/data/'
    path_scripts = '/home/aellien/JWST/JWST_scripts'
    path_wavelets = '/home/aellien/JWST/wavelets/out15/'
    path_plots = '/home/aellien/JWST/plots'
    path_analysis = '/home/aellien/JWST/analysis/'
    
    nfl = [ {'nf':'jw02736001001_f444w_bkg_rot_crop_input.fits', 'chan':'long', 'pix_scale':0.063, 'pixar_sr':9.31E-14, 'n_levels':11, 'lvl_sep_max':999, 'mu_lim':999 } ]

    physcale = 5.3 # kpc/"
    gamma = 0.5
    lvl_sep_big = 5
    lvl_sep_bcg = 6
    rm_gamma_for_big = True

    # Read galaxy catalog
    rgal = pyr.open(os.path.join(path_data, 'mahler_noirot_merged_member_gal_ra_dec_pix_long.reg'))
    cat_gal = []
    for gal in rgal:
        cat_gal.append(gal.coord_list)
    cat_gal = np.array(cat_gal)

    # Read star region files
    rco = pyr.open(os.path.join(path_data, 'star_flags_polygon_pix_long.reg'))
    rell = pyr.open(os.path.join(path_data, 'icl_flags_ellipse_pix_long.reg'))
    rbcg = pyr.open(os.path.join(path_data, 'bcg_flags_ellipse_pix_long.reg'))