##!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 23:44:18 2024

@author: aellien
"""
import sys
import dawis as d
import glob as glob
import os
import numpy as np
import pyregion as pyr
import random
import gc
import h5py
import pandas as pd
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.io import fits
from astropy.convolution import convolve
from astropy.coordinates import SkyCoord
from astropy.visualization import *
from astropy.table import Table
from astropy.wcs import WCS
from scipy.stats import kurtosis
from datetime import datetime
from photutils.segmentation import SourceFinder, SourceCatalog, detect_sources, make_2dgaussian_kernel
from photutils.background import Background2D, MedianBackground
#from cosmo_calc import cosmo_calc
#from power_ratio import *

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def flux_selection_error(atom_in_list, atom_out_list, M, percent, xs, ys, flux_lim, ml = [], write_plots = False, path_plots = None, nfp = None, name = None):
    '''Update - 08/10/2024

    Computes the flux selection error by performing a bootstrap analysis on a sample of 
    image segments (atoms), producing statistics on flux and optionally generating plots 
    for visual analysis.

    Parameters:
    -----------
    atom_in_list : list of tuples
        List of input atoms, each represented as a tuple containing image data and coordinates.
        Each atom tuple structure: (image, det_err_image, x_min, y_min, x_max, y_max).
    
    atom_out_list : list of tuples
        List of additional atoms used for selection replacement, with the same structure as 
        `atom_in_list`.

    M : int
        Number of bootstrap iterations (samples) to draw in the selection process.

    percent : float
        Percentage parameter controlling the size of each bootstrap sample relative to the 
        original atom list.

    xs : int
        Width of the output image for each bootstrap sample.

    ys : int
        Height of the output image for each bootstrap sample.

    flux_lim : float
        Minimum flux value threshold. Only pixels with flux values above this limit are included 
        in the flux calculations.

    ml : list of numpy.ndarray, optional
        List of masks (each as a boolean 2D array) used for masking regions in the images 
        to calculate flux within specific areas.

    write_plots : bool, optional, default=False
        If True, saves visualizations of each bootstrap sample and the final flux histogram.

    path_plots : str, optional
        Directory path where the plots will be saved if `write_plots` is True.

    nfp : str, optional
        Unused in the current function. Kept as a placeholder for possible future functionality.

    name : str, optional
        Name prefix for the saved plots if `write_plots` is True.

    Returns:
    --------
    tuple : (mean_flux, low_flux, up_flux, out_msc)
        - mean_flux : float
            The median flux across all bootstrap samples for the entire image.
        - low_flux : float
            The 5th percentile flux value across all bootstrap samples.
        - up_flux : float
            The 95th percentile flux value across all bootstrap samples.
        - out_msc : numpy.ndarray
            A 1D array of size len(ml) x 3 containing the median, 5th, and 95th percentile flux values 
            for each masked region (if `ml` is provided).

    Notes:
    ------
    - The function generates bootstrap samples by drawing random subsets of `atom_in_list` 
      and replacing part of the sample with atoms from `atom_out_list`.
    - For each bootstrap sample, it calculates the total flux (subject to `flux_lim`) 
      for the entire image and for each masked region specified in `ml`.
    - If `write_plots` is True, individual sample images and a histogram of flux values 
      across bootstrap samples are saved to `path_plots`.
    '''
        
    # Output array
    msc_sample = []

    # Draw atom sample sizes -------------
    size_sample = np.random.uniform(low = int( len(atom_in_list) * (1. - percent)), \
                               high = int( len(atom_in_list) + len(atom_in_list) * percent ), \
                               size = M).astype(int)
    
    # Draw replaced atom sample sizes ---
    replace_sample = []
    for s in size_sample:
        r = int(np.random.uniform(low = 0, high = min(int( s * percent ), len(atom_out_list))))

        # new security in cases where len(atom_out_list) is not so large
        #if (s - len(atom_in_list) + r) >= len(atom_out_list) :
        #    r = abs(len(atom_out_list) - s + len(atom_in_list))
        #    if r < 0:r = 0

        replace_sample.append(r)  
    replace_sample = np.array(replace_sample)

    # Draw actual samples ---------------
    flux_sample = []
    for i, (s, r) in enumerate(zip(size_sample, replace_sample)):
        
        print(i, s, r, len(atom_in_list), len(atom_out_list), end ='\r')
        
        im_s = np.zeros((xs, ys))
        
        if s < len(atom_in_list):
            flux = 0
            draw = random.sample(atom_in_list, s)

        if s >= len(atom_in_list):

            flux = 0
            draw1 = random.sample(atom_in_list, len(atom_in_list) - r)

            # new security in cases where len(atom_out_list) is not so large
            if (s - len(atom_in_list) + r) >= len(atom_out_list):
                draw2 = atom_out_list
            else:
                draw2 = random.sample(atom_out_list, s - len(atom_in_list) + r)
            draw = draw1 + draw2
            
        for atom in draw:
            image = atom[0]
            x_min, y_min, x_max, y_max = atom[2:6]
            im_s[ x_min : x_max, y_min : y_max ] += image

        # clear some memory --------
        del draw

        # Compute fluxes ----------
        
        # Whole image
        flux = np.sum(im_s[im_s >= flux_lim])
        flux_sample.append(flux)
        
        # Masks
        line = []
        for msc in ml:
            im_msc = im_s[msc.astype(bool)] # Apply mask
            im_msc = im_msc[im_msc >= flux_lim] # Apply lower flux lim
            flux_msc = np.sum(im_msc)
            line.append(flux_msc, flux_msc_err)
        msc_sample.append(line)

        # plot individual bootstrap draw results -----
        if write_plots == True:
            
            im_m = np.zeros((xs, ys)) # deterministic image for plot normalization
            for atom in atom_in_list:
                image = atom[0]
                x_min, y_min, x_max, y_max = atom[2:6]
                im_m[ x_min : x_max, y_min : y_max ] += image
            
            interval = AsymmetricPercentileInterval(60, 99.99) # meilleur rendu pour images reconstruites
            norm = ImageNormalize( im_m, interval = interval, stretch = LogStretch())
            plt.figure(num = 1, clear = True)
            plt.imshow(im_s, norm = norm, cmap = 'binary', origin = 'lower')
            plt.savefig( os.path.join( path_plots, name + '_%03d.png'%i), format = 'png')
            plt.close('all')

    # clear some memory -----------
    del im_s
    gc.collect()

    # Bootstrap stats ------------
    
    # Whole image
    flux_sample = np.array(flux_sample)
    mean_flux = np.median(flux_sample) # mean flux value
    up_flux = np.percentile(flux_sample, 95) # upper flux value
    low_flux = np.percentile(flux_sample, 5) # lower flux value

    # Masks
    msc_sample = np.array(msc_sample)
    mean_flux_msc = np.median(msc_sample, axis = 0) # mean flux value
    up_flux_msc = np.percentile(msc_sample, 95, axis = 0) # upper flux value
    low_flux_msc = np.percentile(msc_sample, 5, axis = 0) # lower flux value
    out_msc = np.array([ mean_flux_msc, low_flux_msc, up_flux_msc ] ).swapaxes(0, 1).flatten() # 1d size n_sed_region x 3

    # Plot bootstrap histogram ---
    if write_plots == True:
        plt.figure()
        plt.hist(flux_sample, bins = 10)
        plt.savefig(os.path.join(path_plots, name + '_bootstrap_hist.png'), format = 'png')
        plt.close('all')

    return mean_flux, low_flux, up_flux, out_msc

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def PR_selection_error(atom_in_list, atom_out_list, M, percent, R_pix, xs, ys, flux_lim, ml = [], write_plots = False, path_plots = None, nfp = None, name = None):
    '''Computation of selection error on power ratio.
    Update - 08/10/2024
    '''
    # Output array
    msc_sample = []

    # Draw atom sample sizes -------------
    size_sample = np.random.uniform(low = int( len(atom_in_list) * (1. - percent)), \
                               high = int( len(atom_in_list) + len(atom_in_list) * percent ), \
                               size = M).astype(int)
    
    # Draw replaced atom sample sizes ---
    replace_sample = []
    for s in size_sample:
        r = int(np.random.uniform(low = 0, high = min(int( s * percent ), len(atom_out_list))))

        # new security in cases where len(atom_out_list) is not so large
        #if (s - len(atom_in_list) + r) >= len(atom_out_list) :
        #    r = abs(len(atom_out_list) - s + len(atom_in_list))
        #    if r < 0:r = 0

        replace_sample.append(r)  
    replace_sample = np.array(replace_sample)

    # Draw actual samples ---------------
    PR_sample = []
    for i, (s, r) in enumerate(zip(size_sample, replace_sample)):
        
        print(i, s, r, len(atom_in_list), len(atom_out_list), end ='\r')
        
        im_s = np.zeros((xs, ys))
        
        if s < len(atom_in_list):
            flux = 0
            draw = random.sample(atom_in_list, s)

        if s >= len(atom_in_list):

            flux = 0
            draw1 = random.sample(atom_in_list, len(atom_in_list) - r)

            # new security in cases where len(atom_out_list) is not so large
            if (s - len(atom_in_list) + r) >= len(atom_out_list):
                draw2 = atom_out_list
            else:
                draw2 = random.sample(atom_out_list, s - len(atom_in_list) + r)
            draw = draw1 + draw2
            
        for atom in draw:
            image = atom[0]
            x_min, y_min, x_max, y_max = atom[2:6]
            im_s[ x_min : x_max, y_min : y_max ] += image

        # clear some memory --------
        del draw

        # Compute fluxes ----------
        
        # Whole image
        orderl = []
        for order in range(1, 5):
            PR = power_ratio( image = im_s, order = order, radius = R_pix )
            orderl.append(PR)
        PR_sample.append(orderl)

        # plot individual bootstrap draw results -----
        if write_plots == True:
            
            im_m = np.zeros((xs, ys)) # deterministic image for plot normalization
            for atom in atom_in_list:
                image = atom[0]
                x_min, y_min, x_max, y_max = atom[2:6]
                im_m[ x_min : x_max, y_min : y_max ] += image
            
            interval = AsymmetricPercentileInterval(60, 99.99) # meilleur rendu pour images reconstruites
            norm = ImageNormalize( im_m, interval = interval, stretch = LogStretch())
            plt.figure(num = 1, clear = True)
            plt.imshow(im_s, norm = norm, cmap = 'binary', origin = 'lower')
            plt.savefig( os.path.join( path_plots, name + '_%03d.png'%i), format = 'png')
            plt.close('all')

    # clear some memory -----------
    del im_s
    gc.collect()

    # Bootstrap stats ------------
    
    # Whole image
    PR_sample = np.array(PR_sample)
    PR_results = []
    for i in range(1, 5):
        mean_PR = np.median(PR_sample[:, i - 1])
        up_PR = np.percentile(PR_sample[:, i - 1], 95)
        low_PR = np.percentile(PR_sample[:, i - 1], 5)
        PR_results.append([mean_PR, up_PR, low_PR])

    # Plot bootstrap histogram ---
    if write_plots == True:
        plt.figure()
        plt.hist(flux_sample, bins = 10)
        plt.savefig(os.path.join(path_plots, name + '_bootstrap_hist.png'), format = 'png')
        plt.close('all')

    return PR_results

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def synthesis_bcgwavsizesep_with_masks( cln, oim, header, nfwp, lvl_sep, lvl_sep_max, lvl_sep_bcg, size_sep, size_sep_pix, xs, ys, micl, mbcg, mstar, msat, ml = [], R = np.inf, N_err = 50, per_err = 0.1, kurt_filt = True, plot_vignet = False, write_fits = True, measure_flux = False, measure_PR = False, plot_boot = False ):
    '''
    Synthesizes an image model using wavelet decompositions to separate BCG (Brightest Cluster Galaxy) and ICL (Intracluster Light) components, applying user-defined priors.

    Parameters:
    -----------
    cln : str
        Name identifier for the dataset or observation being processed.
    oim : array-like
        The original input image array.
    header : FITS header object
        Header information for the output FITS file.
    nfwp : str
        Directory path to store output files.
    lvl_sep : int
        Level separation threshold for identifying BCG and ICL components.
    lvl_sep_max : int
        Maximum wavelet level threshold for ICL selection.
    lvl_sep_bcg : int
        Wavelet level threshold specific for BCG components.
    size_sep : float
        Size threshold for identifying extended sources in the separation process.
    size_sep_pix : int
        Minimum pixel size threshold for an object to be classified as an ICL.
    xs, ys : int
        Dimensions of the input image in pixels.
    micl, mbcg, mstar, msat : array-like
        Boolean masks for ICL, BCG, star, and satellite components.
    ml : list, optional
        Optional mask layer; used in flux or error analysis.
    R : float, optional
        Radius within which to analyze objects; defaults to infinite.
    N_err : int, optional
        Number of bootstrap samples for error estimation; defaults to 50.
    per_err : float, optional
        Percentage error threshold for bootstrap; defaults to 0.1.
    kurt_filt : bool, optional
        Whether to apply a kurtosis filter to remove artifacts; defaults to True.
    plot_vignet : bool, optional
        Flag to plot vignettes of models, if desired; defaults to False.
    write_fits : bool, optional
        Flag to write the output synthesized images to a FITS file; defaults to True.
    measure_flux : bool, optional
        Flag to measure flux of the synthesized components using bootstrap; defaults to False.
    measure_PR : bool, optional
        Flag to measure power ratio of the components using bootstrap; defaults to False.
    plot_boot : bool, optional
        Flag to plot bootstrap distribution histograms, if desired; defaults to False.

    Returns:
    --------
    df : pandas.DataFrame
        DataFrame summarizing synthesized quantities, including flux and power ratios, with associated uncertainties.

    Notes:
    ------
    This function processes a series of atoms files to separate and synthesize BCG, ICL, and galaxy components. The separation is based on wavelet levels, size thresholds and optionnal spatial filtering. Additional functionalities include error estimation via bootstrap sampling, power ratio measurement, and output FITS file generation for visualization and analysis.
    '''

    # Empty arrays for models
    icl = np.zeros( (xs, ys) )
    icl_dei = np.zeros( (xs, ys) )
    gal = np.zeros( (xs, ys) )
    gal_dei = np.zeros( (xs, ys) )
    im_art = np.zeros( (xs, ys) )
    im_unclass = np.zeros( (xs, ys) )
    im_unclass_dei = np.zeros( (xs, ys) )
    recim = np.zeros( (xs, ys) )

    # Empty list for selected atoms
    tot_icl_al = []
    tot_gal_al = []
    icl_boot_al = []

    xc = xs / 2.
    yc = ys / 2.

    # List of files to read
    opath = os.path.join(nfwp, '*ol.it*.hdf5')
    opathl = glob.glob(opath)
    opathl.sort()
    print(opath)

    # Iterate over iteration files
    for i, op in enumerate(opathl):
        
        print('Reading iteration %d' %(i), end ='\r')
        with h5py.File(op, "r") as f1:

            # Empty lists for selected atoms within this iteration
            gc.collect()
            #icl_al = []
            #gal_al = []
            noticl_al = []
            unclass_al = []

            # Iterate over objects
            for o in f1.keys():

                # Read data
                x_min, y_min, x_max, y_max = np.copy(f1[o]['bbox'][()])
                image = np.copy(f1[o]['image'][()])
                det_err_image = np.copy(f1[o]['det_err_image'][()])
                lvlo = np.copy(f1[o]['level'][()])
                wr = np.copy(f1[o]['norm_wr'][()])

                # Compute a few quantities
                sx = x_max - x_min
                sy = y_max - y_min
                m = detect_sources(image, threshold = 0., npixels=1)
                c = SourceCatalog(image, m)
                xco = int(c.centroid_quad[0][1] + x_min)
                yco = int(c.centroid_quad[0][0] + y_min)

                # Filter bad atoms/artifacts
                if kurt_filt == True:
                    k = kurtosis(image.flatten(), fisher=True)
                    if k < 0:
                        im_art[ x_min : x_max, y_min : y_max ] += image
                        gc.collect()
                        continue

                # Add to full reconstructed image
                recim[ x_min : x_max, y_min : y_max ] += image

                # Remove background, and add it to bootstrap if center is within ICL mask
                if lvlo >= lvl_sep_max:
                    if micl[xco, yco] == 1:
                        icl_boot_al.append([image, det_err_image, x_min, y_min, x_max, y_max, xco, yco, lvlo, wr])
                    continue
        
                # Only atoms within analysis radius
                dR = np.sqrt( (xc - xco)**2 + (yc - yco)**2 )
                if dR > R:
                    continue
    
                # ICL + BCG
                if (mstar[xco, yco] != 1) & (micl[xco, yco] > 0):
                            
                    # BCG
                    if mbcg[xco, yco] > 0:
                        icl[ x_min : x_max, y_min : y_max ] += image
                        icl_dei[ x_min : x_max, y_min : y_max ] += det_err_image
                        tot_icl_al.append([image, det_err_image, x_min, y_min, x_max, y_max, xco, yco, lvlo, wr])
                        continue
    
                    # ICL
                    if (lvlo >= lvl_sep) & (sx >= size_sep_pix) & (sy >= size_sep_pix):
                        icl[ x_min : x_max, y_min : y_max ] += image
                        icl_dei[ x_min : x_max, y_min : y_max ] += det_err_image
                        tot_icl_al.append([image, det_err_image, x_min, y_min, x_max, y_max, xco, yco, lvlo, wr])
                        
                    else:
                        noticl_al.append([image, det_err_image, x_min, y_min, x_max, y_max, xco, yco, lvlo, wr])
                else:
                    noticl_al.append([image, det_err_image, x_min, y_min, x_max, y_max, xco, yco, lvlo, wr])
                    
            # Galaxies
            for j, (image, det_err_image, x_min, y_min, x_max, y_max, xco, yco, lvlo, wr) in enumerate(noticl_al):
                
                # Satellites
                if (mstar[xco, yco] != 1) & (lvlo < lvl_sep) & (msat[xco, yco] > 0):

                    gal[ x_min : x_max, y_min : y_max ] += image
                    gal_dei[ x_min : x_max, y_min : y_max ] += det_err_image
                    tot_gal_al.append([image, det_err_image, x_min, y_min, x_max, y_max, xco, yco, lvlo, wr])

                    # Add 'tendencious' galaxy atoms to list for bootstrap
                    if ( lvlo == (lvl_sep - 1)) & (micl[xco, yco] > 0):
                        icl_boot_al.append([image, det_err_image, x_min, y_min, x_max, y_max, xco, yco, lvlo, wr]) 
                        
                # If not identified as galaxies --> test if BCG again
                else:
                    unclass_al.append([image, det_err_image, x_min, y_min, x_max, y_max, xco, yco, lvlo, wr])

            # Test for unclassified atoms --> sometimes extended ICL/BCG signal can be missed because
            # it is at wavelet scales lower than lvl_sep and not in mbcg
            for j, (image, det_err_image, x_min, y_min, x_max, y_max, xco, yco, lvlo, wr) in enumerate(unclass_al):
                
                # Case in which it is possible that it is BCG halo?
                if (lvl_sep > lvl_sep_bcg) & (lvlo >= lvl_sep_bcg) & (np.sqrt( (xc - xco)**2 + (yc - yco)**2 ) < size_sep_pix) :
                    icl[ x_min : x_max, y_min : y_max ] += image
                    icl_dei[ x_min : x_max, y_min : y_max ] += det_err_image
                    tot_icl_al.append([image, det_err_image, x_min, y_min, x_max, y_max, xco, yco, lvlo, wr])
                    
                # If not --> unclassified 
                else:
                    im_unclass[ x_min : x_max, y_min : y_max ] += image
                    im_unclass_dei[ x_min : x_max, y_min : y_max ] += det_err_image
                    
                    # Add tendencious atoms to list for bootstrap (arbitrary conditions)
                    if ( lvlo == (lvl_sep - 1)) & (micl[xco, yco] > 0):
                        icl_boot_al.append([image, det_err_image, x_min, y_min, x_max, y_max, xco, yco, lvlo, wr])

    # clear some memory
    gc.collect()
    noticl_al.clear()
    unclass_al.clear()

    # Write synthesized models to disk
    if write_fits == True:

        kernel = make_2dgaussian_kernel(5.0, size = 5)  # FWHM = 3.0
        icl = convolve(icl, kernel)
        gal = convolve(gal, kernel)
        recim = convolve(recim, kernel)
        
        # BCG SIZE SEP
        print('\nWrite fits in %s'%(nfwp))
        hdu = fits.PrimaryHDU()
        hdu_icl = fits.ImageHDU(icl, name = 'ICL+BCG', header = header)
        hdu_gal = fits.ImageHDU(gal, name = 'SATELLITES', header = header)
        tot = gal + icl
        hdu_tot = fits.ImageHDU(tot, name = 'ICL+BCG+SATELLITES', header = header)
        hdu_icl_dei = fits.ImageHDU(icl_dei, name = 'ICL+BCG DET. ERR.', header = header)
        hdu_gal_dei = fits.ImageHDU(gal_dei, name = 'SAT DET. ERR.', header = header)
        tot_dei = icl_dei + gal_dei
        hdu_tot_dei = fits.ImageHDU(tot_dei, name = 'ICL+BCG+SAT DET. ERR.', header = header)
        hdu_unclass = fits.ImageHDU(im_unclass, name = 'UNCLASSIFIED', header = header)
        hdu_unclass_dei = fits.ImageHDU(im_unclass, name = 'UNCLASSIFIED DET. ERR.', header = header)
        hdul = fits.HDUList([ hdu, hdu_icl, hdu_gal, hdu_tot, hdu_unclass, hdu_icl_dei, hdu_gal_dei, hdu_tot_dei, hdu_unclass_dei ])
        hdul.writeto( os.path.join(nfwp, 'synth.bcgwavsizesepmask_%03d_%03d.fits'%(lvl_sep, size_sep)), overwrite = True )
        
        # FULL FIELD
        hdu = fits.PrimaryHDU()
        hdu_oim = fits.ImageHDU(oim, name = 'ORIGINAL', header = header)
        hdu_recim = fits.ImageHDU(recim, name = 'REC.', header = header)
        hdu_res = fits.ImageHDU(oim - recim, name = 'RESIDUALS', header = header)
        hdul = fits.HDUList([ hdu, hdu_oim, hdu_recim, hdu_res ])
        hdul.writeto( os.path.join(nfwp, 'synth.full_field.fits'), overwrite = True )
        
    # Measure integrated quantities
    if measure_flux == True:
        
        print('Start bootstrap.')
        start = datetime.now()
        
        # Measure Fluxes from bootstrap distribution ------------

        # ICL+BCG fluxes + bootstrap
        F_ICLBCG_m, F_ICLBCG_low, F_ICLBCG_up, msc_out_icl =  flux_selection_error(tot_icl_al, icl_boot_al, N_err, per_err, xs, ys, flux_lim, ml = ml, write_plots = False, path_plots = None, nfp = None, name = None)

        # Satellite fluxes + bootstrap
        F_gal_m, F_gal_low, F_gal_up, msc_out_gal =  flux_selection_error(tot_gal_al, icl_boot_al, N_err, per_err, xs, ys, flux_lim, ml = ml, write_plots = False, path_plots = None, nfp = None, name = None)

        # Compute uncertainties ------------------------

        # ICL+BCG
        wrl = [ a[-1] for a in tot_icl_al ]
        iclbcg_wr_err = np.sum( np.array(wrl)**2)
        iclbcg_det_err = np.sum(icl_dei**2)
        iclbcg_sel_err_up = F_ICLBCG_up - F_ICLBCG_m
        iclbcg_sel_err_low = F_ICLBCG_m - F_ICLBCG_low
        iclbcg_tot_err_up = np.sqrt( iclbcg_wr_err + iclbcg_det_err + iclbcg_sel_err_up**2 )
        iclbcg_tot_err_low = np.sqrt( iclbcg_wr_err + iclbcg_det_err + iclbcg_sel_err_low**2 )

        # Satellites
        wrl = [ a[-1] for a in tot_gal_al ]
        gal_wr_err = np.sum( np.array(wrl)**2)
        gal_det_err = np.sum(gal_dei**2)
        gal_sel_err_up = F_gal_up - F_gal_m
        gal_sel_err_low = F_gal_m - F_gal_low
        gal_tot_err_up = np.sqrt( gal_wr_err + gal_det_err + gal_sel_err_up**2 )
        gal_tot_err_low = np.sqrt( gal_wr_err + gal_det_err + gal_sel_err_low**2 )
        
        # ICL+BCG fractions --------------------------------
        
        f_ICLBCG_m = F_ICLBCG_m / (F_ICLBCG_m + F_gal_m)
        f_ICLBCG_low = F_ICLBCG_low / (F_ICLBCG_low + F_gal_up)
        f_ICLBCG_up = F_ICLBCG_up / (F_ICLBCG_up + F_gal_low)

        # ICL fractions
        #f_ICL_m = F_ICL_m / (F_ICL_m + F_gal_m)
        #f_ICL_low = F_ICL_low / (F_ICL_low + F_gal_up)
        #f_ICL_up = F_ICL_up / (F_ICL_up + F_gal_low)
        
        print(datetime.now() - start)
        
    else:
         F_ICLBCG_m = np.nan
         F_ICLBCG_low = np.nan
         F_ICLBCG_up = np.nan
         F_gal_m = np.nan
         F_gal_low = np.nan
         F_gal_up = np.nan
         f_ICLBCG_m = np.nan
         f_ICLBCG_low = np.nan
         f_ICLBCG_up = np.nan

    # Measure Power ratio
    if measure_PR == True:
        
        results_PR = PR_selection_error(tot_icl_al, icl_boot_al, N_err, per_err, R_pix, xs, ys, flux_lim, ml = ml, write_plots = False, path_plots = None, nfp = None, name = None)
        PR_1_m, PR_1_up, PR_1_low = results_PR[0]
        PR_2_m, PR_2_up, PR_2_low = results_PR[1]
        PR_3_m, PR_3_up, PR_3_low = results_PR[2]
        PR_4_m, PR_4_up, PR_4_low = results_PR[3]
    else:
        PR_1_m, PR_1_up, PR_1_low = np.nan, np.nan, np.nan
        PR_2_m, PR_2_up, PR_2_low = np.nan, np.nan, np.nan
        PR_3_m, PR_3_up, PR_3_low = np.nan, np.nan, np.nan
        PR_4_m, PR_4_up, PR_4_low = np.nan, np.nan, np.nan
        
    # Display
    print('\nWS + SF -- ICL+BCG --  z = %d'%lvl_sep)
    print('N = %4d   F_ICL = %1.2e ADU  err_low = %1.2e ADU  err_up = %1.2e ADU'\
                                  %(len(tot_icl_al), F_ICLBCG_m, F_ICLBCG_low, F_ICLBCG_up))
    print('N = %4d   F_gal = %1.2e ADU  err_low = %1.2e ADU  err_up = %1.2e ADU'\
                                  %(len(tot_gal_al), F_gal_m, F_gal_low, F_gal_up))
    print('Det. error: de_ICL = %1.2e ADU  Wav. res. error: wr_ICL = %1.2e'\
                                  %(iclbcg_det_err, iclbcg_wr_err))
    print('Det. error: de_gal = %1.2e ADU  Wav. res. error: wr_gal = %1.2e'\
                                  %(gal_det_err, gal_wr_err))
    print('f_ICL = %1.3f    f_ICL_low = %1.3f   f_ICL_up = %1.3f'%(f_ICLBCG_m, f_ICLBCG_low, f_ICLBCG_up))
    print('PR_1_m = %1.2e    PR_1_low = %1.2e    PR_1_up = %1.2e'%(PR_1_m, PR_1_low, PR_1_up))
    print('PR_2_m = %1.2e    PR_2_low = %1.2e    PR_2_up = %1.2e'%(PR_2_m, PR_2_low, PR_2_up))
    print('PR_3_m = %1.2e    PR_3_low = %1.2e    PR_3_up = %1.2e'%(PR_3_m, PR_3_low, PR_3_up))
    print('PR_4_m = %1.2e    PR_4_low = %1.2e    PR_4_up = %1.2e'%(PR_4_m, PR_4_low, PR_4_up))

    # Clear memory
    tot_icl_al.clear()
    tot_gal_al.clear()
    icl_boot_al.clear()

    # Compact dataframe
    df = pd.DataFrame( [[ cln, R_pix, lvl_sep, size_sep, F_ICLBCG_m, F_ICLBCG_low, F_ICLBCG_up, F_gal_m, F_gal_low, F_gal_up, f_ICLBCG_m, f_ICLBCG_low, f_ICLBCG_up, PR_1_m, PR_1_up, PR_1_low, PR_2_m, PR_2_up, PR_2_low, PR_3_m, PR_3_up, PR_3_low, PR_4_m, PR_4_up, PR_4_low ]], \
                        columns = [ 'name', 'R_pix', 'lvl_sep', 'size_sep', 'F_ICLBCG_m', 'F_ICLBCG_low', 'F_ICLBCG_up', 'F_gal_m', 'F_gal_low', 'F_gal_up', 'f_ICLBCG_m', 'f_ICLBCG_low', 'f_ICLBCG_up', 'PR_1_m', 'PR_1_up', 'PR_1_low', 'PR_2_m', 'PR_2_up', 'PR_2_low', 'PR_3_m', 'PR_3_up', 'PR_3_low', 'PR_4_m', 'PR_4_up', 'PR_4_low'  ])
    return df
    

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def make_annuli_mask(r1, r2, xs, ys, xc, yc):

    mask = np.ones((xs, ys))

    Y, X = np.ogrid[:xs, :ys]
    dist_from_center = np.sqrt((X - xc)**2 + (Y-yc)**2)

    mask[dist_from_center < r1] = 0.
    mask[dist_from_center > r2] = 0.
    
    return mask   

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def lines_that_contain(string, fp):
    return [line for line in fp if string in line]

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def create_sat_mask(oim, header, cat):

    # make deblended segmentation map
    bkg_estimator = MedianBackground()
    data = np.copy(oim)
    bkg = Background2D(data, (50, 50), filter_size=(3, 3), bkg_estimator=bkg_estimator)
    data -= bkg.background
    threshold = 1.5 * bkg.background_rms
    kernel = make_2dgaussian_kernel(3.0, size = 5)  # FWHM = 3.0
    convolved_data = convolve(data, kernel)
    finder = SourceFinder(npixels = 10, progress_bar = False)
    segment_map = finder(convolved_data, threshold)

    # WCS
    w = WCS(header)
    satlabl = []
    for gal in cat:
        sky = SkyCoord(gal['RAdeg'], gal['DEdeg'], unit = 'deg')
        y, x = w.world_to_pixel(sky)
        try:
            satlabl.append( segment_map.data[int(x), int(y)])
        except:
            continue

    # Remove non-satellite labels
    for lab in segment_map.labels:
        if lab not in satlabl:
            segment_map.remove_labels(labels = lab)

    mask = np.copy(segment_map.data)
    mask[mask > 0] = 1
    return mask
    
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if __name__ == '__main__':

    # Paths, lists & variables
    path_data = '/n03data/ellien/JWST/data/'
    path_scripts = '/n03data/ellien/JWST/JWST_scripts'
    path_wavelets = '/n03data/ellien/JWST/wavelets/out21/'
    path_plots = '/n03data/ellien/JWST/plots'
    path_analysis = '/home/ellien/JWST/analysis/'
    
    # Input files
    clnl = sys.argv[1:]

    # Cosmology
    H0 = 70.0 # Hubble constant
    Om_M = 0.3 # Matter density
    Om_Lam = 0.7 # Energy density
    physcale = 5.3 # [kpc/"]
    
    # spatial scales related
    pix_scale = 0.187 # pixel scale [arcsec/pix]
    size_sep = 80 # size separation [kpc]
    size_sep_pix = size_sep / physcale / pix_scale # [pix]
    rc = 10 # distance to center to be classified as gal [kpc]
                
    # wavelet scales related
    lvl_sep = 5 # wavelet scale separation
    lvl_sep_bcg = 4
    lvl_sep_max = 1000
    n_levels = 10
    
    # bootstrap
    N_err = 100
    per_err = 0.1

    # Read galaxy catalog
    rgal = pyr.open(os.path.join(path_data, 'mahler_noirot_merged_member_gal_ra_dec_pix_long.reg'))
    cat_gal = []
    for gal in rgal:
        cat_gal.append(gal.coord_list)
    cat_gal = np.array(cat_gal)

    # Read star region files
    rstar = pyr.open(os.path.join(path_data, 'star_flags_polygon_pix_long.reg'))
    rell = pyr.open(os.path.join(path_data, 'icl_flags_ellipse_pix_long.reg'))
    rbcg = pyr.open(os.path.join(path_data, 'bcg_flags_ellipse_pix_long.reg'))
    r128 = pyr.open(os.path.join(path_data, 'circle_128kpc_pix.reg'))
    r200 = pyr.open(os.path.join(path_data, 'circle_200kpc_pix.reg'))
    r400 = pyr.open(os.path.join(path_data, 'circle_400kpc_pix.reg'))

    # misc
    kurt_filt = True
    plot_vignet = False
    write_fits = True
    measure_flux = True
    measure_PR = False
    plot_boot = False
    write_dataframe = True
    
    for cln in clnl:

        # find cluster properties in catalog
        with open('bands.txt') as bands:
            split = lines_that_contain(cln, bands)[0].split()
        ID = split[0]
        if ID != cln:
            print('%s not matching %s --> ignored!'%(ID, cln))
            continue
        chan = split[1]
        pixscale = float(split[2])
        pixar = float(split[3])
        n_levels = float(split[4])
        lvl_sep_max = float(split[5])
        mu_lim = float(split[6])

        # Photometry for limiting depth
        ZP_AB = -6.10 - 2.5 * np.log10(pixar)
        flux_lim = 10**( (ZP_AB - mu_lim) / 2.5 )
            
        # Read original file
        nfp = os.path.join(path_data, cln) 
        hdu = fits.open(nfp)
        head = hdu[0].header
        oim = hdu[0].data
        
        xs, ys = oim.shape
        xc, yc = xs / 2., ys / 2.

        # create masks
        micl = rell.get_mask(hdu = hdu[0]) # not python convention
        mstar = rstar.get_mask(hdu = hdu[0]) # not python convention
        mbcg = rbcg.get_mask(hdu = hdu[0]) # not python convention
        m128 = r128.get_mask(hdu = hdu[0])
        m200 = r200.get_mask(hdu = hdu[0])
        m400 = r400.get_mask(hdu = hdu[0])
        msat = create_sat_mask(oim, head, cat_gal)
        
        # synthesis
        df = synthesis_bcgwavsizesep_with_masks( cln = cln, 
                                             oim = oim, 
                                             header = head, 
                                             nfwp = path_wavelets, 
                                             lvl_sep = lvl_sep, 
                                             lvl_sep_max = lvl_sep_max, 
                                             lvl_sep_bcg = lvl_sep_bcg, 
                                             size_sep = size_sep, 
                                             size_sep_pix = size_sep_pix, 
                                             xs = xs, 
                                             ys = ys, 
                                             micl = micl, 
                                             mbcg = mbcg, 
                                             mstar = mstar,
                                             msat = msat,
                                             ml = [r128, r200, r400], 
                                             R = None, 
                                             N_err = N_err, 
                                             per_err = per_err, 
                                             kurt_filt = kurt_filt, 
                                             plot_vignet = plot_vignet, 
                                             write_fits = write_fits,
                                             measure_flux = measure_flux,
                                             measure_PR = measure_PR,
                                             plot_boot = plot_boot)
        
        if write_dataframe == True:
            print('Write dataframe to %s' %os.path.join(path_analysis, cln + '_fICL_PR.txt'))
            df.to_csv(os.path.join(path_analysis, cln + '_fICL_PR.txt'), sep=' ')
