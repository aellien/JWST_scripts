import sys
import dawis as d
import glob as glob
import os
import numpy as np
import pyregion as pyr
import random
import pandas as pd
import ray
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import *
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.morphology import binary_dilation
from scipy.stats import kurtosis
import gc
from power_ratio import *
from datetime import datetime
import h5py

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def synthesis_fullfield( oim, nfp, xs, ys, write_fits = True ):
    '''Synthesis of the full astronomical field (e.g. sum of all atoms)
    --- Args:
    oim         # Original astronomical field
    nfp         # root path of *.pkl
    gamma       # attenuation factor
    lvl_sep_big # wavelet scale at which gamma set to 1
    lvl_sep     # wavelet scale threshold for the separation
    xs, ys      # image size
    n_levels    # number of wavelet scales
    plot_vignet # plot pdf vignet of output
    --- Output:
    rec         # synthesis image with all atoms
    res         # residuals (original - rec)
    '''
    # path, list & variables
    res = np.zeros((xs, ys))
    rec = np.zeros((xs, ys))
    wei = np.zeros((xs, ys))
    dei = np.zeros((xs, ys))
    
    ######################################## MEMORY v
    opath = nfp + 'ol.it*.hdf5'
    opathl = glob.glob(opath)
    opathl.sort()

    for i, op in enumerate(opathl):
        print('Iteration %d' %(i), end ='\r')
        gc.collect()
        #ol = d.store_objects.read_ol_from_hdf5(op)
        with h5py.File(op, "r") as f:
            
            ############################################################## MEMORY ^
            for o in f.keys():

                x_min, y_min, x_max, y_max = f[o]['bbox'][()]
                rec[ x_min : x_max, y_min : y_max ] += f[o]['image'][()]

                # atom weight map
                w = np.copy(f[o]['image'][()])
                w[w > 0.] = 1.
                wei[ x_min : x_max, y_min : y_max ] += w
            
                # detection error image
                dei[ x_min : x_max, y_min : y_max ] += f[o]['det_err_image'][()]
            
    res = oim - rec
    if write_fits == True:

        print('\nFULLFIELD -- write results to %s'%(nfp + 'synth.full_field.fits'))
        
        hdu = fits.PrimaryHDU()
        hdu_oim = fits.ImageHDU(oim, name = 'ORIGINAL')
        hdu_rec = fits.ImageHDU(rec, name = 'RESTORED')
        hdu_res = fits.ImageHDU(res, name = 'RESIDUALS')
        hdu_wei = fits.ImageHDU(wei, name = 'WEIGHTS')
        hdu_dei = fits.ImageHDU(dei, name = 'DETECT. ERROR')
        
        hdul = fits.HDUList([hdu, hdu_oim, hdu_rec, hdu_res, hdu_wei, hdu_dei])
        hdul.writeto(nfp + 'synth.full_field.fits', overwrite = True)

    return rec, res, wei, dei

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def selection_error(atom_in_list, atom_out_list, M, percent, xs, ys, flux_lim, mscsedl):
    '''Computation of classification error on flux.
    '''
    # Output array
    sed_sample = []

    # Sample
    size_sample = np.random.uniform(low = int( len(atom_in_list) * (1. - percent)), \
                               high = int( len(atom_in_list) + len(atom_in_list) * percent ), \
                               size = M).astype(int)
    replace_sample = []
    for s in size_sample:
        replace_sample.append(int(np.random.uniform(low = 0, high = int( s * percent ))))
    replace_sample = np.array(replace_sample)

    flux_sample = []
    flux_err_sample = []
    for i, (s, r) in enumerate(zip(size_sample, replace_sample)):
        
        print(i, s, r, len(atom_in_list), len(atom_out_list))
        im_s = np.zeros((xs, ys))
        im_s_err = np.zeros((xs, ys))
        if s < len(atom_in_list):
            print('cc1')
            flux = 0
            draw = random.sample(atom_in_list, s)

        if s >= len(atom_in_list):
            print('cc2')

            flux = 0
            draw1 = random.sample(atom_in_list, len(atom_in_list) - r)
            draw2 = random.sample(atom_out_list, s - len(atom_in_list) + r)
            draw = draw1 + draw2
            
        #print(i, len(atom_in_list), len(atom_out_list), len(draw), len(draw[0]), draw[0])
        for (image, det_err_image, x_min, y_min, x_max, y_max, xco, yco, lvlo) in draw:
            im_s[ x_min : x_max, y_min : y_max ] += image
            im_s_err[ x_min : x_max, y_min : y_max ] += det_err_image
            #flux += np.sum(o.image)

        flux = np.sum(im_s[im_s >= flux_lim])
        flux_err = np.sqrt(np.sum(im_s_err[im_s >= flux_lim]**2))
        flux_sample.append(flux)
        flux_err_sample.append(flux_err)

        line = []
        for mscsed in mscsedl:
            flux_sed = np.sum(im_s[mscsed.astype(bool)])
            line.append(flux_sed)

        sed_sample.append(line)

    sed_sample = np.array(sed_sample)
    mean_sed = np.median(sed_sample, axis = 0)
    up_err_sed = np.percentile(sed_sample, 95, axis = 0)
    low_err_sed = np.percentile(sed_sample, 5, axis = 0)
    out_sed = np.array([ mean_sed, low_err_sed, up_err_sed ] ).swapaxes(0, 1).flatten() # 1d size n_sed_region x 3

    flux_sample = np.array(flux_sample)
    flux_err_sample = np.array(flux_err_sample)
    mean_flux = np.median(flux_sample)
    up_flux = np.percentile(flux_sample, 95)
    low_flux = np.percentile(flux_sample, 5)
    mean_flux_err = np.median(flux_sample)
    up_flux_err = np.percentile(flux_sample, 95)
    low_flux_err = np.percentile(flux_sample, 5)

    #plt.figure()
    #plt.hist(flux_sample, bins = 10)
    #plt.show()

    return mean_flux, low_flux, up_flux, mean_flux_err, up_flux_err, low_flux_err, out_sed

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def PR_with_selection_error(atom_in_list, atom_out_list, M, percent, R, xs, ys):
    '''Computation of classification error on PR.
    '''
    size_sample = np.random.uniform(low = int( len(atom_in_list) * (1. - percent)), \
                               high = int( len(atom_in_list) + len(atom_in_list) * percent ), \
                               size = M).astype(int)
    replace_sample = []
    for s in size_sample:
        replace_sample.append(int(np.random.uniform(low = 0, high = int( s * percent ))))
    replace_sample = np.array(replace_sample)

    PR_sample = []
    for s, r in zip(size_sample, replace_sample):
        im = np.zeros((xs, ys))

        if s < len(atom_in_list):

            draw = random.sample(atom_in_list, s)
            for (image, det_err_image, x_min, y_min, x_max, y_max, xco, yco, lvlo) in draw:
                im[ x_min : x_max, y_min : y_max ] += image

            orderl = []
            for order in range(1, 5):
                PR = power_ratio( image = im, order = order, radius = R )
                orderl.append(PR)
            PR_sample.append(orderl)

        if s >= len(atom_in_list):

            flux = 0
            draw1 = random.sample(atom_in_list, len(atom_in_list) - r)
            draw2 = random.sample(atom_out_list, s - len(atom_in_list) + r)
            draw = draw1 + draw2
            for (image, det_err_image, x_min, y_min, x_max, y_max, xco, yco, lvlo) in draw:
                im[ x_min : x_max, y_min : y_max ] += image

            orderl = []

            for order in range(1, 5):
                PR = power_ratio( image = im, order = order, radius = R )
                orderl.append(PR)
            PR_sample.append(orderl)

        #%---
        #interval = AsymmetricPercentileInterval(5, 99.5) # meilleur rendu que MinMax or ZScale pour images reconstruites
        #fig, ax = plt.subplots(1)
        #poim = ax.imshow(im, norm = ImageNormalize( im, interval = interval, stretch = LogStretch()), cmap = 'binary', origin = 'lower')
        #%---

    PR_sample = np.array(PR_sample)
    PR_results = []
    for i in range(1, 5):
        mean_PR = np.median(PR_sample[:, i - 1])
        up_err = np.percentile(PR_sample[:, i - 1], 95)
        low_err = np.percentile(PR_sample[:, i - 1], 5)
        PR_results.append([mean_PR, up_err, low_err])
    #%---
    #plt.figure()
    #for i in range(1,5):
    #    plt.hist(PR_sample[:, i-1], bins = 10, alpha = 0.5)
    #plt.show()
    #%---

    return PR_results

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def synthesis_wavsep( nfp, lvl_sep, xs, ys, n_levels, kurt_filt = False, plot_vignet = False, write_fits = True ):
    '''Simple separation based on wavelet scale, given by parameter 'lvl_sep'.
    --- Args:
    nfp         # root path of *.pkl
    lvl_sep     # wavelet scale threshold for the separation
    xs, ys      # image size
    n_levels    # number of wavelet scales
    plot_vignet # plot pdf vignet of output
    --- Output:
    icl         # synthesis image with atoms at wavelet scale >= lvl_sep
    gal         # synthesis image with atoms at wavelet scale < lvl_sep
    '''
    # path, list & variables
    icl = np.zeros( (xs, ys) )
    gal = np.zeros( (xs, ys) )
    im_art = np.zeros((xs, ys))
    icl_dei = np.zeros((xs, ys))
    gal_dei = np.zeros((xs, ys))

    ######################################## MEMORY v
    opath = nfp + '*ol.it*.hdf5'
    itpath = nfp + '*itl.it*.hdf5'
    opathl = glob.glob(opath)
    opathl.sort()

    # Interscale tree lists
    itpathl = glob.glob(itpath)
    itpathl.sort()

    for i, ( op, itlp ) in enumerate( zip( opathl, itpathl )):
        
        gc.collect()
        ol = d.store_objects.read_ol_from_hdf5(op)
        itl = d.store_objects.read_itl_from_hdf5(itlp)

        for o, it in zip(ol, itl):
            ######################################## MEMORY ^
            
            x_min, y_min, x_max, y_max = o.bbox

            if kurt_filt == True:
                k = kurtosis(o.image.flatten(), fisher=True)
                if k < 0:
                    im_art[ x_min : x_max, y_min : y_max ] += o.image
                    continue

            if o.level >= lvl_sep:
                icl[ x_min : x_max, y_min : y_max ] += o.image
                icl_dei[ x_min : x_max, y_min : y_max ] += o.det_err_image
            else:
                gal[ x_min : x_max, y_min : y_max ] += o.image
                gal_dei[ x_min : x_max, y_min : y_max ] += o.det_err_image


    if write_fits == True:
        print('WS -- write fits as %s*'%(nfp))
        
        hdu = fits.PrimaryHDU()
        hdu_icl = fits.ImageHDU(icl, name = 'LARGE SCALE')
        hdu_gal = fits.ImageHDU(gal, name = 'SMALL SCALE')
        hdu_icl_dei = fits.ImageHDU(icl_dei, name = 'LARGE SCALE DET. ERR.')
        hdu_gal_dei = fits.ImageHDU(gal_dei, name = 'SMALL SCALE DET. ERR.')

        hdul = fits.HDUList([ hdu, hdu_icl, hdu_gal, hdu_icl_dei, hdu_gal_dei ])

        # write to fits
        hdul.writeto( nfp + 'synth.wavsep_%03d.fits'%lvl_sep, overwrite = True )

    if plot_vignet == True:
        interval = AsymmetricPercentileInterval(5, 99.5) # meilleur rendu que MinMax or ZScale pour images reconstruites

        if kurt_filt == True:
            fig, ax = plt.subplots(1, 3)
        else:
            fig, ax = plt.subplots(1, 2)

        poim = ax[0].imshow(gal, norm = ImageNormalize( gal, interval = interval, stretch = LogStretch()), cmap = 'binary')
        divider = make_axes_locatable(ax[0])
        cax = divider.append_axes("top", size="5%", pad=0.05)
        caxre = fig.colorbar( poim, cax = cax, \
                                    orientation = 'horizontal', \
                                    format = '%2.1f',\
                                    pad = 0,\
                                    shrink = 1.0,\
                                    ticklocation = 'top' )
        poim = ax[1].imshow(icl, norm = ImageNormalize( icl, interval = interval, stretch = LogStretch()), cmap = 'binary')
        divider = make_axes_locatable(ax[1])
        cax = divider.append_axes("top", size="5%", pad=0.05)
        caxre = fig.colorbar( poim, cax = cax, \
                                    orientation = 'horizontal', \
                                    format = '%2.1f',\
                                    pad = 0,\
                                    shrink = 1.0,\
                                    ticklocation = 'top' )

        if kurt_filt == True:
            ax[2].imshow(im_art, norm = ImageNormalize(im_art, interval = interval, stretch = LogStretch() ), cmap = 'binary')

        plt.tight_layout()
        plt.savefig( nfp + 'results.wavsep_%03d.png'%lvl_sep, format = 'png' )
        print('Write vignet to' + nfp + 'results.wavsep_%03d.png'%(lvl_sep))
        plt.close('all')

    return icl, gal

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def synthesis_bcgwavsep_with_masks( nfp, lvl_sep, lvl_sep_max, lvl_sep_bcg, xs, ys, n_levels, mscstar, mscell, mscbcg, mscsedl, R, cat_gal, rc_pix, N_err, per_err, flux_lim, rm_gamma_for_big = True, kurt_filt = True, plot_vignet = False, write_fits = True, measure_PR = False ):
    '''Simple separation based on wavelet scale, given by parameter 'lvl_sep'.
    '''

    # path, list & variables
    icl = np.zeros( (xs, ys) )
    icl_dei = np.zeros( (xs, ys) )
    gal = np.zeros( (xs, ys) )
    gal_dei = np.zeros( (xs, ys) )
    im_art = np.zeros( (xs, ys) )
    im_unclass = np.zeros( (xs, ys) )
    im_unclass_dei = np.zeros( (xs, ys) )

    tot_icl_al = []
    tot_gal_al = []
    #tot_noticl_al = []
    tot_unclass_al = []

    #%
    at_test = []
    #%

    xc = xs / 2.
    yc = ys / 2.

    ######################################## MEMORY v
    opath = nfp + '*ol.it*.hdf5'
    itpath = nfp + '*itl.it*.hdf5'
    opathl = glob.glob(opath)
    opathl.sort()

    # Interscale tree lists
    itpathl = glob.glob(itpath)
    itpathl.sort()

    for i, ( op, itlp ) in enumerate( zip( opathl, itpathl )):
        print('Iteration %d' %(i), end ='\r')
        
        #ol = d.store_objects.read_ol_from_hdf5(op)
        #itl = d.store_objects.read_itl_from_hdf5(itlp)
        with h5py.File(op, "r") as f1, h5py.File(itlp, "r") as f2:
            gc.collect()
            icl_al = []
            gal_al = []
            noticl_al = []
            unclass_al = []
            for o, it in zip(f1.keys(), f2.keys()):
                ######################################## MEMORY ^

                x_min, y_min, x_max, y_max = f1[o]['bbox'][()]
                image = f1[o]['image'][()]
                det_err_image = f1[o]['det_err_image'][()]
                itm = f2[it]['interscale_maximum']
                xco = itm['x_max'][()]
                yco = itm['y_max'][()]
                lvlo = f1[o]['level'][()]
            
                if kurt_filt == True:
                    k = kurtosis(image.flatten(), fisher=True)
                    if k < 0:
                        im_art[ x_min : x_max, y_min : y_max ] += image
                        continue
        
                # Remove background
                if lvlo >= lvl_sep_max:
                    tot_unclass_al.append([image, det_err_image, x_min, y_min, x_max, y_max, xco, yco, lvlo])
                    continue
        
                # Only atoms within analysis radius
                dR = np.sqrt( (xc - xco)**2 + (yc - yco)**2 )
                if dR > R:
                    continue
        
                # ICL+BCG
                if (mscstar[xco, yco] != 1) & (mscell[xco, yco] == 1):
        
                    # BCG
                    if mscbcg[xco, yco] == 1:
                        icl[ x_min : x_max, y_min : y_max ] += image
                        icl_dei[ x_min : x_max, y_min : y_max ] += det_err_image
                        icl_al.append([image, det_err_image, x_min, y_min, x_max, y_max, xco, yco, lvlo])
                        tot_icl_al.append([image, det_err_image, x_min, y_min, x_max, y_max, xco, yco, lvlo])
        
                    # ICL
                    else:
                        if lvlo >= lvl_sep:
                            icl[ x_min : x_max, y_min : y_max ] += image
                            icl_dei[ x_min : x_max, y_min : y_max ] += det_err_image
                            icl_al.append([image, det_err_image, x_min, y_min, x_max, y_max, xco, yco, lvlo])
                            tot_icl_al.append([image, det_err_image, x_min, y_min, x_max, y_max, xco, yco, lvlo])
                            at_test.append([xco, yco])
                        else:
                            #gal[ x_min : x_max, y_min : y_max ] += image
                            noticl_al.append([image, det_err_image, x_min, y_min, x_max, y_max, xco, yco, lvlo])
                            #tot_noticl_al.append([image, det_err_image, x_min, y_min, x_max, y_max, xco, yco, lvlo])
        
                else:
                    noticl_al.append([image, det_err_image, x_min, y_min, x_max, y_max, xco, yco, lvlo])
                    #tot_noticl_al.append([image, det_err_image, x_min, y_min, x_max, y_max, xco, yco, lvlo])
            # Galaxies
            for j, (image, det_err_image, x_min, y_min, x_max, y_max, xco, yco, lvlo) in enumerate(noticl_al):
        
                if mscstar[xco, yco] != 1:
        
                    # Satellites
                    if lvlo < lvl_sep:
        
                        flag = False
                        for ygal, xgal in cat_gal:
                            dr = np.sqrt( (xgal - xco)**2 + (ygal - yco)**2 )
                            if dr <= rc_pix:
                                flag = True
                                gal[ x_min : x_max, y_min : y_max ] += image
                                gal_dei[ x_min : x_max, y_min : y_max ] += det_err_image
                                gal_al.append([image, det_err_image, x_min, y_min, x_max, y_max, xco, yco, lvlo])
                                tot_gal_al.append([image, det_err_image, x_min, y_min, x_max, y_max, xco, yco, lvlo])
                                break
        
                        # If not identified as galaxies --> test if BCG again
                        if flag == False:
                            unclass_al.append([image, det_err_image, x_min, y_min, x_max, y_max, xco, yco, lvlo])
                    else:
                        unclass_al.append([image, det_err_image, x_min, y_min, x_max, y_max, xco, yco, lvlo])
                else:
                    unclass_al.append([image, det_err_image, x_min, y_min, x_max, y_max, xco, yco, lvlo])

            # Test for unclassified atoms --> sometimes extended BCG halo is missed because
            # of the nature of wavsep scheme.
            for j, (image, det_err_image, x_min, y_min, x_max, y_max, xco, yco, lvlo) in enumerate(unclass_al):
        
                # Case in which it is possible that it is BCG halo?
                if lvl_sep > lvl_sep_bcg:
        
                    #BCG extended halo?
                    if (lvlo >= lvl_sep_bcg) & (mscell[xco, yco] == 1) :
                        icl[ x_min : x_max, y_min : y_max ] += image
                        icl_dei[ x_min : x_max, y_min : y_max ] += det_err_image
                        icl_al.append([image, det_err_image, x_min, y_min, x_max, y_max, xco, yco, lvlo])
                        tot_icl_al.append([image, det_err_image, x_min, y_min, x_max, y_max, xco, yco, lvlo])
        
                    #If not --> unclassified
                    else:
                        tot_unclass_al.append([image, det_err_image, x_min, y_min, x_max, y_max, xco, yco, lvlo])
                        im_unclass[ x_min : x_max, y_min : y_max ] += image
                        im_unclass_dei[ x_min : x_max, y_min : y_max ] += det_err_image
        
                #If not --> unclassified
                else:
                    tot_unclass_al.append([image, det_err_image, x_min, y_min, x_max, y_max, xco, yco, lvlo])
                    im_unclass[ x_min : x_max, y_min : y_max ] += image
                    im_unclass_dei[ x_min : x_max, y_min : y_max ] += det_err_image
                                                                                                                                   
    # Remove potential foreground star artifacts
    #gal[mscstar == 1.] = 0.
    #%
    at_test = np.array(at_test)
    #%
        
    if write_fits == True:
        print('\nWS + SF -- ICL+BCG -- write fits as %s*'%(nfp))
        
        hdu = fits.PrimaryHDU()
        hdu_icl = fits.ImageHDU(icl, name = 'ICL+BCG')
        hdu_gal = fits.ImageHDU(gal, name = 'SATELLITES')
        tot = gal + icl
        hdu_tot = fits.ImageHDU(tot, name = 'ICL+BCG+SATELLITES')
        hdu_icl_dei = fits.ImageHDU(icl_dei, name = 'ICL+BCG DET. ERR.')
        hdu_gal_dei = fits.ImageHDU(gal_dei, name = 'SAT DET. ERR.')
        tot_dei = icl_dei + gal_dei
        hdu_tot_dei = fits.ImageHDU(tot_dei, name = 'ICL+BCG+SAT DET. ERR.')
        hdu_unclass = fits.ImageHDU(im_unclass, name = 'UNCLASSIFIED')
        hdu_unclass_dei = fits.ImageHDU(im_unclass, name = 'UNCLASSIFIED DET. ERR.')

        
        hdul = fits.HDUList([ hdu, hdu_icl, hdu_gal, hdu_tot, hdu_unclass, hdu_icl_dei, hdu_gal_dei, hdu_tot_dei, hdu_unclass_dei ])

        # write to fits
        hdul.writeto( nfp + 'synth.bcgwavsepmask_%03d.fits'%lvl_sep, overwrite = True )
        
    # Plot vignets
    if plot_vignet == True:
        
        interval = AsymmetricPercentileInterval(5, 99.5) # meilleur rendu que MinMax or ZScale pour images reconstruites
        fig, ax = plt.subplots(2, 2)
        poim = ax[0][0].imshow(gal, norm = ImageNormalize( gal, interval = interval, stretch = LogStretch()), cmap = 'binary', origin = 'lower')
        poim = ax[1][0].imshow(icl, norm = ImageNormalize( icl, interval = interval, stretch = LogStretch()), cmap = 'binary', origin = 'lower')
        '''#%
        rco = pyr.open(os.path.join(path_data, 'star_flags_polygon_pix_long.reg'))
        patch_list, artist_list = rco.get_mpl_patches_texts()
        for p in patch_list:
            ax[1][0].add_patch(p)
        for a in artist_list:
            ax[1][0].add_artist(a)
        for at in at_test:
            ax[1][0].plot(at[1], at[0], 'b+')
        #%'''
        poim = ax[0][1].imshow(im_unclass, norm = ImageNormalize( gal, interval = interval, stretch = LogStretch()), cmap = 'binary', origin = 'lower')
        poim = ax[1][1].imshow(im_art, norm = ImageNormalize( gal, interval = interval, stretch = LogStretch()), cmap = 'binary', origin = 'lower')
        #plt.show()
        plt.tight_layout()
        plt.savefig( nfp + 'results.bcgwavsepmask_%03d.png'%lvl_sep, format = 'png' )
        print('Write vignet to' + nfp + 'synth.bcgwavsepmask_%03d.png'%(lvl_sep))
        plt.close('all')

    if measure_PR == True:
        print('start bootstrap')
        start = datetime.now()
        # Measure Fractions and uncertainties
        F_ICL_m, F_ICL_low, F_ICL_up, FICL_det_err, low_FICL_det_err, up_FICL_det_err, out_sed_icl =  selection_error(tot_icl_al, tot_unclass_al+tot_gal_al, M = N_err, percent = per_err, xs = xs, ys = ys, flux_lim = flux_lim, mscsedl = mscsedl)
        F_gal_m, F_gal_low, F_gal_up, Fgal_det_err, low_Fgal_det_err, up_Fgal_det_err, out_sed_gal =  selection_error(tot_gal_al, tot_unclass_al+tot_icl_al, M = N_err, percent = per_err, xs = xs, ys = ys, flux_lim = flux_lim, mscsedl = mscsedl)
        f_ICL_m = F_ICL_m / (F_ICL_m + F_gal_m)
        f_ICL_low = F_ICL_low / (F_ICL_low + F_gal_up)
        f_ICL_up = F_ICL_up / (F_ICL_up + F_gal_low)
        print(datetime.now() - start)
        
        print('\nWS + SF -- ICL+BCG --  z = %d'%lvl_sep)
        print('N = %4d   F_ICL = %f ADU  err_low = %f ADU  err_up = %f ADU'%(len(tot_icl_al), F_ICL_m, F_ICL_low, F_ICL_up))
        print('N = %4d   F_gal = %f ADU  err_low = %f ADU  err_up = %f ADU'%(len(tot_gal_al), F_gal_m, F_gal_low, F_gal_up))
        print('Det. error: deICL = %f ADU  deICL_low = %f  deICLup = %f ADU'%(FICL_det_err, low_FICL_det_err, up_FICL_det_err))
        print('f_ICL = %1.3f    f_ICL_low = %1.3f   f_ICL_up = %1.3f'%(f_ICL_m, f_ICL_low, f_ICL_up))
        
        # Measure Power ratio
        results_PR = PR_with_selection_error(atom_in_list = tot_icl_al, atom_out_list = tot_unclass_al+tot_gal_al, M = N_err, percent = per_err, R = R, xs = xs, ys = ys)
        PR_1_m, PR_1_up, PR_1_low = results_PR[0]
        PR_2_m, PR_2_up, PR_2_low = results_PR[1]
        PR_3_m, PR_3_up, PR_3_low = results_PR[2]
        PR_4_m, PR_4_up, PR_4_low = results_PR[3]
        
        print('PR_1_m = %1.2e    PR_1_low = %1.2e    PR_1_up = %1.2e'%(PR_1_m, PR_1_low, PR_1_up))
        print('PR_2_m = %1.2e    PR_2_low = %1.2e    PR_2_up = %1.2e'%(PR_2_m, PR_2_low, PR_2_up))
        print('PR_3_m = %1.2e    PR_3_low = %1.2e    PR_3_up = %1.2e'%(PR_3_m, PR_3_low, PR_3_up))
        print('PR_4_m = %1.2e    PR_4_low = %1.2e    PR_4_up = %1.2e'%(PR_4_m, PR_4_low, PR_4_up))
        
        return icl, gal, F_ICL_m, F_ICL_low, F_ICL_up, F_gal_m, F_gal_low, F_gal_up, f_ICL_m, f_ICL_low, f_ICL_up, PR_1_m, PR_1_up, PR_1_low, PR_2_m, PR_2_up, PR_2_low, PR_3_m, PR_3_up, PR_3_low, PR_4_m, PR_4_up, PR_4_low, out_sed_icl, out_sed_gal
    else:
        return icl, gal,  np.nan,    np.nan,   np.nan,  np.nan,    np.nan,   np.nan,  np.nan,    np.nan,   np.nan, np.nan,  np.nan,   np.nan, np.nan,  np.nan,   np.nan, np.nan,  np.nan,   np.nan, np.nan,  np.nan,   np.nan, [ np.nan ], [ np.nan ]
    
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def synthesis_bcgwavsizesep_with_masks( nfp, lvl_sep, lvl_sep_max, lvl_sep_bcg, size_sep, size_sep_pix, xs, ys, n_levels, mscstar, mscell, mscbcg, mscsedl, R, cat_gal, rc_pix, N_err, per_err, flux_lim, kurt_filt = True, plot_vignet = False, write_fits = True, measure_PR = False ):
    '''Wavelet Separation + Spatial filtering.
    ICL --> Atoms with z > lvl_sep, with maximum coordinates within ellipse mask 'mscell' and with size > size_sep_pix.
    Galaxies --> Satellites + BCG, so a bit complicated:
        - Atoms not classified as ICL but with maximum coordinates within ellipse mask 'mscbcg'
        - Atoms within radius 'rc' of a member galaxy position
        - In case lvl_sep > 5 (arbitrary...), atoms with z > 5 and within 'mscell' are BCG
    Unclassified --> rest of atoms
    '''
    # path, list & variables
    icl = np.zeros( (xs, ys) )
    icl_dei = np.zeros( (xs, ys) )
    gal = np.zeros( (xs, ys) )
    gal_dei = np.zeros( (xs, ys) )
    im_art = np.zeros( (xs, ys) )
    im_unclass = np.zeros( (xs, ys) )
    im_unclass_dei = np.zeros( (xs, ys) )

    tot_icl_al = []
    tot_gal_al = []
    #tot_noticl_al = []
    tot_unclass_al = []

    #%
    at_test = []
    #%

    xc = xs / 2.
    yc = ys / 2.

    ######################################## MEMORY v
    opath = nfp + '*ol.it*.hdf5'
    itpath = nfp + '*itl.it*.hdf5'
    opathl = glob.glob(opath)
    opathl.sort()

    # Interscale tree lists
    itpathl = glob.glob(itpath)
    itpathl.sort()
    
    for i, ( op, itlp ) in enumerate( zip( opathl, itpathl )):
        print('Iteration %d' %(i), end ='\r')
        
        #ol = d.store_objects.read_ol_from_hdf5(op)
        #itl = d.store_objects.read_itl_from_hdf5(itlp)
        with h5py.File(op, "r") as f1, h5py.File(itlp, "r") as f2:
            gc.collect()
            icl_al = []
            gal_al = []
            noticl_al = []
            unclass_al = []
            for o, it in zip(f1.keys(), f2.keys()):
                ######################################## MEMORY ^

                x_min, y_min, x_max, y_max = f1[o]['bbox'][()]
                image = f1[o]['image'][()]
                det_err_image = f1[o]['det_err_image'][()]
                itm = f2[it]['interscale_maximum']
                xco = itm['x_max'][()]
                yco = itm['y_max'][()]
                lvlo = f1[o]['level'][()]
                sx = x_max - x_min
                sy = y_max - y_min
            
                if kurt_filt == True:
                    k = kurtosis(image.flatten(), fisher=True)
                    if k < 0:
                        im_art[ x_min : x_max, y_min : y_max ] += image
                        continue
        
                # Remove background
                if lvlo >= lvl_sep_max:
                    tot_unclass_al.append([image, det_err_image, x_min, y_min, x_max, y_max, xco, yco, lvlo])
                    continue
        
                # Only atoms within analysis radius
                dR = np.sqrt( (xc - xco)**2 + (yc - yco)**2 )
                if dR > R:
                    continue
    
                # ICL + BCG
                if (mscstar[xco, yco] != 1) & (mscell[xco, yco] == 1):
    
                    '''# BCG
                    xbcg, ybcg = [ xs, ys ] # pix long, ds9 convention
                    if mscbcg[xco, yco] == 1:
    
                        dr = np.sqrt( (xbcg - xco)**2 + (ybcg - yco)**2 )
                        if (o.level <= 3) & (dr < rc_pix):
    
                            icl[ x_min : x_max, y_min : y_max ] += o.image
                            icl_al.append([o, xco, yco])
    
                        elif o.level >= 4:
                            icl[ x_min : x_max, y_min : y_max ] += o.image
                            icl_al.append([o, xco, yco])'''
                            
                    # BCG
                    if mscbcg[xco, yco] == 1:
                        icl[ x_min : x_max, y_min : y_max ] += image
                        icl_dei[ x_min : x_max, y_min : y_max ] += det_err_image
                        icl_al.append([image, det_err_image, x_min, y_min, x_max, y_max, xco, yco, lvlo])
                        tot_icl_al.append([image, det_err_image, x_min, y_min, x_max, y_max, xco, yco, lvlo])
                        continue
    
                    # ICL
                    if (lvlo >= lvl_sep) & (sx >= size_sep_pix) & (sy >= size_sep_pix):

                        #%%%%% Je laisse au cas où %%%%% v
                        coo_spur_halo = []
                        # [ [1615, 1665], [1685, 1480], [530, 260] ] # pix long, ds9 convention
                        flag = False
                        for ygal, xgal in coo_spur_halo:

                            dr = np.sqrt( (xgal - xco)**2 + (ygal - yco)**2 )
                            if (dr <= rc_pix) & (lvlo == 5):
                                flag = True
                        #%%%%%%% ^^^^^^^^^^^^^^^^^^^^^^^^^^
                            
                        if flag == False:
                            icl[ x_min : x_max, y_min : y_max ] += image
                            icl_dei[ x_min : x_max, y_min : y_max ] += det_err_image
                            icl_al.append([image, det_err_image, x_min, y_min, x_max, y_max, xco, yco, lvlo])
                            tot_icl_al.append([image, det_err_image, x_min, y_min, x_max, y_max, xco, yco, lvlo])
                            at_test.append([xco, yco])
                            continue
                            
                    else:
                        noticl_al.append([image, det_err_image, x_min, y_min, x_max, y_max, xco, yco, lvlo])
                else:
                    noticl_al.append([image, det_err_image, x_min, y_min, x_max, y_max, xco, yco, lvlo])
                    
            # Galaxies
            for j, (image, det_err_image, x_min, y_min, x_max, y_max, xco, yco, lvlo) in enumerate(noticl_al):
                
                # Satellites
                if (mscstar[xco, yco] != 1) & (lvlo < lvl_sep):

                    flag = False
                    for ygal, xgal in cat_gal:
                        dr = np.sqrt( (xgal - xco)**2 + (ygal - yco)**2 )
                        if dr <= rc_pix:
                            flag = True
                            gal[ x_min : x_max, y_min : y_max ] += image
                            gal_dei[ x_min : x_max, y_min : y_max ] += det_err_image
                            gal_al.append([image, det_err_image, x_min, y_min, x_max, y_max, xco, yco, lvlo])
                            tot_gal_al.append([image, det_err_image, x_min, y_min, x_max, y_max, xco, yco, lvlo])
                            break

                    # If not identified as galaxies --> test if BCG again
                    if flag == False:
                        unclass_al.append([image, det_err_image, x_min, y_min, x_max, y_max, xco, yco, lvlo])

                else:
                    unclass_al.append([image, det_err_image, x_min, y_min, x_max, y_max, xco, yco, lvlo])

            # Test for unclassified atoms --> sometimes extended BCG halo is missed because
            # of the nature of wavsep scheme.
            for j, (image, det_err_image, x_min, y_min, x_max, y_max, xco, yco, lvlo) in enumerate(unclass_al):
                
                # Case in which it is possible that it is BCG halo?
                if (lvl_sep > lvl_sep_bcg) & (lvlo >= lvl_sep_bcg) & (mscell[xco, yco] == 1) :
                    icl[ x_min : x_max, y_min : y_max ] += image
                    icl_dei[ x_min : x_max, y_min : y_max ] += det_err_image
                    icl_al.append([image, det_err_image, x_min, y_min, x_max, y_max, xco, yco, lvlo])
    
                #If not --> unclassified
                else:
                    im_unclass[ x_min : x_max, y_min : y_max ] += image
                    im_unclass_dei[ x_min : x_max, y_min : y_max ] += det_err_image
                    tot_unclass_al.append([image, det_err_image, x_min, y_min, x_max, y_max, xco, yco, lvlo])

    # Remove potential foreground star artifacts
    #gal[mscstar == 1.] = 0.
    
    at_test = np.array(at_test)

    if write_fits == True:
        print('\nWS + SF + SS -- ICL+BCG -- LVL_SEP = %d -- write fits as %s*'%(lvl_sep, nfp))

        hdu = fits.PrimaryHDU()
        hdu_icl = fits.ImageHDU(icl, name = 'ICL+BCG')
        hdu_gal = fits.ImageHDU(gal, name = 'SATELLITES')
        tot = gal + icl
        hdu_tot = fits.ImageHDU(tot, name = 'ICL+BCG+SATELLITES')
        hdu_icl_dei = fits.ImageHDU(icl_dei, name = 'ICL+BCG DET. ERR.')
        hdu_gal_dei = fits.ImageHDU(gal_dei, name = 'SAT DET. ERR.')
        tot_dei = icl_dei + gal_dei
        hdu_tot_dei = fits.ImageHDU(tot_dei, name = 'ICL+BCG+SAT DET. ERR.')
        hdu_unclass = fits.ImageHDU(im_unclass, name = 'UNCLASSIFIED')
        hdu_unclass_dei = fits.ImageHDU(im_unclass, name = 'UNCLASSIFIED DET. ERR.')

        
        hdul = fits.HDUList([ hdu, hdu_icl, hdu_gal, hdu_tot, hdu_unclass, hdu_icl_dei, hdu_gal_dei, hdu_tot_dei, hdu_unclass_dei ])
        hdul.writeto( nfp + 'synth.bcgwavsizesepmask_%03d_%03d.fits'%(lvl_sep, size_sep), overwrite = True )

    # Plot vignets
    if plot_vignet == True:

        interval = AsymmetricPercentileInterval(5, 99.5) # meilleur rendu que MinMax or ZScale pour images reconstruites
        fig, ax = plt.subplots(2, 2)
        poim = ax[0][0].imshow(gal, norm = ImageNormalize( gal, interval = interval, stretch = LogStretch()), cmap = 'binary', origin = 'lower')
        poim = ax[1][0].imshow(icl, norm = ImageNormalize( icl, interval = interval, stretch = LogStretch()), cmap = 'binary', origin = 'lower')
        #%
        rco = pyr.open(os.path.join(path_data, 'star_flags_polygon_pix_long.reg'))
        patch_list, artist_list = rco.get_mpl_patches_texts()
        for p in patch_list:
            ax[1][0].add_patch(p)
        for a in artist_list:
            ax[1][0].add_artist(a)
        for at in at_test:
            ax[1][0].plot(at[1], at[0], 'b+')
        #%
        poim = ax[0][1].imshow(im_unclass, norm = ImageNormalize( gal, interval = interval, stretch = LogStretch()), cmap = 'binary', origin = 'lower')
        poim = ax[1][1].imshow(im_art, norm = ImageNormalize( gal, interval = interval, stretch = LogStretch()), cmap = 'binary', origin = 'lower')

        #plt.show()
        plt.tight_layout()
        plt.savefig( nfp + 'results.bcgwavsizesepmask_%03d_%03d_testspur.png'%(lvl_sep, size_sep), format = 'png' )
        print('Write vignet to' + nfp + 'synth.bcgwavsizesepmask_%03d_%03d_testspur.png'%(lvl_sep, size_sep))
        plt.close('all')

    if measure_PR == True:
        print('start bootstrap')
        start = datetime.now()
        # Measure Fractions and uncertainties
        F_ICL_m, F_ICL_low, F_ICL_up, FICL_det_err, low_FICL_det_err, up_FICL_det_err, out_sed_icl =  selection_error(tot_icl_al, tot_unclass_al+tot_gal_al, M = N_err, percent = per_err, xs = xs, ys = ys, flux_lim = flux_lim, mscsedl = mscsedl)
        F_gal_m, F_gal_low, F_gal_up, Fgal_det_err, low_Fgal_det_err, up_Fgal_det_err, out_sed_gal =  selection_error(tot_gal_al, tot_unclass_al+tot_icl_al, M = N_err, percent = per_err, xs = xs, ys = ys, flux_lim = flux_lim, mscsedl = mscsedl)
        f_ICL_m = F_ICL_m / (F_ICL_m + F_gal_m)
        f_ICL_low = F_ICL_low / (F_ICL_low + F_gal_up)
        f_ICL_up = F_ICL_up / (F_ICL_up + F_gal_low)
        print(datetime.now() - start)
        
        print('\nWS + SF -- ICL+BCG --  z = %d'%lvl_sep)
        print('N = %4d   F_ICL = %f Mjy/sr  err_low = %f Mjy/sr  err_up = %f Mjy/sr'%(len(tot_icl_al), F_ICL_m, F_ICL_low, F_ICL_up))
        print('N = %4d   F_gal = %f Mjy/sr  err_low = %f Mjy/sr  err_up = %f Mjy/sr'%(len(tot_gal_al), F_gal_m, F_gal_low, F_gal_up))
        print('Det. error: deICL = %f ADU  deICL_low = %f  deICLup = %f ADU'%(FICL_det_err, low_FICL_det_err, up_FICL_det_err))
        print('f_ICL = %1.3f    f_ICL_low = %1.3f   f_ICL_up = %1.3f'%(f_ICL_m, f_ICL_low, f_ICL_up))
        
        # Measure Power ratio
        results_PR = PR_with_selection_error(atom_in_list = tot_icl_al, atom_out_list = tot_unclass_al+tot_gal_al, M = N_err, percent = per_err, R = R, xs = xs, ys = ys)
        PR_1_m, PR_1_up, PR_1_low = results_PR[0]
        PR_2_m, PR_2_up, PR_2_low = results_PR[1]
        PR_3_m, PR_3_up, PR_3_low = results_PR[2]
        PR_4_m, PR_4_up, PR_4_low = results_PR[3]
        
        print('PR_1_m = %1.2e    PR_1_low = %1.2e    PR_1_up = %1.2e'%(PR_1_m, PR_1_low, PR_1_up))
        print('PR_2_m = %1.2e    PR_2_low = %1.2e    PR_2_up = %1.2e'%(PR_2_m, PR_2_low, PR_2_up))
        print('PR_3_m = %1.2e    PR_3_low = %1.2e    PR_3_up = %1.2e'%(PR_3_m, PR_3_low, PR_3_up))
        print('PR_4_m = %1.2e    PR_4_low = %1.2e    PR_4_up = %1.2e'%(PR_4_m, PR_4_low, PR_4_up))
        
        return icl, gal, F_ICL_m, F_ICL_low, F_ICL_up, F_gal_m, F_gal_low, F_gal_up, f_ICL_m, f_ICL_low, f_ICL_up, PR_1_m, PR_1_up, PR_1_low, PR_2_m, PR_2_up, PR_2_low, PR_3_m, PR_3_up, PR_3_low, PR_4_m, PR_4_up, PR_4_low, out_sed_icl, out_sed_gal
    else:
        return icl, gal,  np.nan,    np.nan,   np.nan,  np.nan,    np.nan,   np.nan,  np.nan,    np.nan,   np.nan, np.nan,  np.nan,   np.nan, np.nan,  np.nan,   np.nan, np.nan,  np.nan,   np.nan, np.nan,  np.nan,   np.nan, [ np.nan ], [ np.nan ]

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def synthesis_wavsep_with_masks( nfp, lvl_sep, lvl_sep_max, lvl_sep_bcg, xs, ys, n_levels, mscstar, mscell, mscbcg, mscsedl, R, cat_gal, rc_pix, N_err, per_err, flux_lim, kurt_filt = True, plot_vignet = False, write_fits = True, measure_PR = False ):
    '''Wavelet Separation + Spatial filtering.
    ICL --> Atoms with z > lvl_sep and with maximum coordinates within ellipse mask 'mscell'
    Galaxies --> Satellites + BCG, so a bit complicated:
        - Atoms not classified as ICL but with maximum coordinates within ellipse mask 'mscbcg'
        - Atoms within radius 'rc' of a member galaxy position
        - In case lvl_sep > 5 (arbitrary...), atoms with z > 5 and within 'mscell' are BCG
    Unclassified --> rest of atoms
    '''
    # path, list & variables
    icl = np.zeros( (xs, ys) )
    icl_dei = np.zeros( (xs, ys) )
    gal = np.zeros( (xs, ys) )
    gal_dei = np.zeros( (xs, ys) )
    im_art = np.zeros( (xs, ys) )
    im_unclass = np.zeros( (xs, ys) )
    im_unclass_dei = np.zeros( (xs, ys) )


    tot_icl_al = []
    tot_gal_al = []
    #tot_noticl_al = []
    tot_unclass_al = []
    
    xc = xs / 2.
    yc = ys / 2.

    ######################################## MEMORY v
    opath = nfp + '*ol.it*.hdf5'
    itpath = nfp + '*itl.it*.hdf5'
    opathl = glob.glob(opath)
    opathl.sort()

    # Interscale tree lists
    itpathl = glob.glob(itpath)
    itpathl.sort()
     
    for i, ( op, itlp ) in enumerate( zip( opathl, itpathl )):
        print('Iteration %d' %(i), end ='\r')
        
        #ol = d.store_objects.read_ol_from_hdf5(op)
        #itl = d.store_objects.read_itl_from_hdf5(itlp)
        with h5py.File(op, "r") as f1, h5py.File(itlp, "r") as f2:
            gc.collect()
            icl_al = []
            gal_al = []
            noticl_al = []
            unclass_al = []
            for o, it in zip(f1.keys(), f2.keys()):
                ######################################## MEMORY ^

                x_min, y_min, x_max, y_max = f1[o]['bbox'][()]
                image = f1[o]['image'][()]
                det_err_image = f1[o]['det_err_image'][()]
                itm = f2[it]['interscale_maximum']
                xco = itm['x_max'][()]
                yco = itm['y_max'][()]
                lvlo = f1[o]['level'][()]

                if kurt_filt == True:
                    k = kurtosis(image.flatten(), fisher=True)
                    if k < 0:
                        im_art[ x_min : x_max, y_min : y_max ] += image
                        continue
        
                # Remove background
                if lvlo >= lvl_sep_max:
                    tot_unclass_al.append([image, det_err_image, x_min, y_min, x_max, y_max, xco, yco, lvlo])
                    continue
        
                # Only atoms within analysis radius
                dR = np.sqrt( (xc - xco)**2 + (yc - yco)**2 )
                if dR > R:
                    continue

                # ICL
                if (mscstar[xco, yco] != 1) & (mscell[xco, yco] == 1):
                    if lvlo >= lvl_sep:
                        icl[ x_min : x_max, y_min : y_max ] += image
                        icl_dei[ x_min : x_max, y_min : y_max ] += det_err_image
                        icl_al.append([image, det_err_image, x_min, y_min, x_max, y_max, xco, yco, lvlo])
                        tot_icl_al.append([image, det_err_image, x_min, y_min, x_max, y_max, xco, yco, lvlo])
                    else:
                        noticl_al.append([image, det_err_image, x_min, y_min, x_max, y_max, xco, yco, lvlo])
                else:
                    noticl_al.append([image, det_err_image, x_min, y_min, x_max, y_max, xco, yco, lvlo])
        
            # Galaxies
            for j, (image, det_err_image, x_min, y_min, x_max, y_max, xco, yco, lvlo) in enumerate(noticl_al):
        
                if mscstar[xco, yco] != 1:
        
                    # BCG
                    if mscbcg[xco, yco] == 1:
                        gal[ x_min : x_max, y_min : y_max ] += image
                        gal_dei[ x_min : x_max, y_min : y_max ] += det_err_image
                        gal_al.append([image, det_err_image, x_min, y_min, x_max, y_max, xco, yco, lvlo])
                        tot_gal_al.append([image, det_err_image, x_min, y_min, x_max, y_max, xco, yco, lvlo])
                        continue
        
                    # Satellites
                    if lvlo < lvl_sep:
        
                        flag = False
                        for ygal, xgal in cat_gal:
                            dr = np.sqrt( (xgal - xco)**2 + (ygal - yco)**2 )
                            if dr <= rc_pix:
                                flag = True
                                gal[ x_min : x_max, y_min : y_max ] += image
                                gal_dei[ x_min : x_max, y_min : y_max ] += det_err_image
                                gal_al.append([image, det_err_image, x_min, y_min, x_max, y_max, xco, yco, lvlo])
                                tot_gal_al.append([image, det_err_image, x_min, y_min, x_max, y_max, xco, yco, lvlo])
                                break
        
                        # If not identified as galaxies --> test if BCG again
                        if flag == False:
                            unclass_al.append([image, det_err_image, x_min, y_min, x_max, y_max, xco, yco, lvlo])

            # Remove potential foreground star artifacts
            #gal[mscstar == 1.] = 0.
        
            # Test for unclassified atoms --> sometimes extended BCG halo is missed because
            # of the nature of wavsep scheme.
            for j, (image, det_err_image, x_min, y_min, x_max, y_max, xco, yco, lvlo) in enumerate(unclass_al):
                
                # Case in which it is possible that it is BCG halo?
                if (lvl_sep > lvl_sep_bcg) & (lvlo >= lvl_sep_bcg) & (mscell[xco, yco] == 1) :
                    gal[ x_min : x_max, y_min : y_max ] += image
                    gal_dei[ x_min : x_max, y_min : y_max ] += det_err_image
                    gal_al.append([image, det_err_image, x_min, y_min, x_max, y_max, xco, yco, lvlo])
    
                #If not --> unclassified
                else:
                    im_unclass[ x_min : x_max, y_min : y_max ] += image
                    im_unclass_dei[ x_min : x_max, y_min : y_max ] += det_err_image
                    tot_unclass_al.append([image, det_err_image, x_min, y_min, x_max, y_max, xco, yco, lvlo])
                 

    if write_fits == True:
        print('\nWS + SF -- ICL -- write fits as %s*'%(nfp))

        hdu = fits.PrimaryHDU()
        hdu_icl = fits.ImageHDU(icl, name = 'ICL')
        hdu_gal = fits.ImageHDU(gal, name = 'BCG+SATELLITES')
        tot = gal + icl
        hdu_tot = fits.ImageHDU(tot, name = 'ICL+BCG+SATELLITES')
        hdu_icl_dei = fits.ImageHDU(icl_dei, name = 'ICL DET. ERR.')
        hdu_gal_dei = fits.ImageHDU(gal_dei, name = 'BCG+SAT DET. ERR.')
        tot_dei = icl_dei + gal_dei
        hdu_tot_dei = fits.ImageHDU(tot_dei, name = 'ICL+BCG+SAT DET. ERR.')
        hdu_unclass = fits.ImageHDU(im_unclass, name = 'UNCLASSIFIED')
        hdu_unclass_dei = fits.ImageHDU(im_unclass, name = 'UNCLASSIFIED DET. ERR.')

        
        hdul = fits.HDUList([ hdu, hdu_icl, hdu_gal, hdu_tot, hdu_unclass, hdu_icl_dei, hdu_gal_dei, hdu_tot_dei, hdu_unclass_dei ])
        hdul.writeto( nfp + 'synth.wavsepmask_%03d.fits'%(lvl_sep), overwrite = True )

    # Plot vignets
    if plot_vignet == True:

        interval = AsymmetricPercentileInterval(5, 99.5) # meilleur rendu que MinMax or ZScale pour images reconstruites
        fig, ax = plt.subplots(2, 2)
        poim = ax[0][0].imshow(gal, norm = ImageNormalize( gal, interval = interval, stretch = LogStretch()), cmap = 'binary', origin = 'lower')
        poim = ax[1][0].imshow(icl, norm = ImageNormalize( icl, interval = interval, stretch = LogStretch()), cmap = 'binary', origin = 'lower')
        #%
        rco = pyr.open(os.path.join(path_data, 'star_flags_polygon_pix_long.reg'))
        patch_list, artist_list = rco.get_mpl_patches_texts()
        for p in patch_list:
            ax[1][0].add_patch(p)
        for a in artist_list:
            ax[1][0].add_artist(a)
        for at in at_test:
            ax[1][0].plot(at[1], at[0], 'b+')
        #%
        poim = ax[0][1].imshow(im_unclass, norm = ImageNormalize( gal, interval = interval, stretch = LogStretch()), cmap = 'binary', origin = 'lower')
        poim = ax[1][1].imshow(im_art, norm = ImageNormalize( gal, interval = interval, stretch = LogStretch()), cmap = 'binary', origin = 'lower')

        #plt.show()
        plt.tight_layout()
        plt.savefig( nfp + 'results.wavsepmask_%03d.png'%lvl_sep, format = 'png' )
        print('Write vignet to' + nfp + 'synth.wavsepmask_%03d.png'%(lvl_sep))
        plt.close('all')

    if measure_PR == True:

        # Measure Fractions and uncertainties
        F_ICL_m, F_ICL_low, F_ICL_up, FICL_det_err, low_FICL_det_err, up_FICL_det_err, out_sed_icl =  selection_error(tot_icl_al, tot_unclass_al+tot_gal_al, M = N_err, percent = per_err, xs = xs, ys = ys, flux_lim = flux_lim, mscsedl = mscsedl)
        F_gal_m, F_gal_low, F_gal_up, Fgal_det_err, low_Fgal_det_err, up_Fgal_det_err, out_sed_gal =  selection_error(tot_gal_al, tot_unclass_al+tot_icl_al, M = N_err, percent = per_err, xs = xs, ys = ys, flux_lim = flux_lim, mscsedl = mscsedl)
        f_ICL_m = F_ICL_m / (F_ICL_m + F_gal_m)
        f_ICL_low = F_ICL_low / (F_ICL_low + F_gal_up)
        f_ICL_up = F_ICL_up / (F_ICL_up + F_gal_low)

        print('\nWS + SF -- ICL -- z = %d'%lvl_sep)
        print('N = %4d   F_ICL = %f Mjy/sr  err_low = %f Mjy/sr  err_up = %f Mjy/sr'%(len(icl_al), F_ICL_m, F_ICL_low, F_ICL_up))
        print('N = %4d   F_gal = %f Mjy/sr  err_low = %f Mjy/sr  err_up = %f Mjy/sr'%(len(gal_al), F_gal_m, F_gal_low, F_gal_up))
        print('f_ICL = %1.3f    f_ICL_low = %1.3f   f_ICL_up = %1.3f'%(f_ICL_m, f_ICL_low, f_ICL_up))

        # Measure Power ratio
        results_PR = PR_with_selection_error(atom_in_list = icl_al, atom_out_list = tot_unclass_al+tot_gal_al, M = N_err, percent = per_err, R = R, xs = xs, ys = ys)
        PR_1_m, PR_1_up, PR_1_low = results_PR[0]
        PR_2_m, PR_2_up, PR_2_low = results_PR[1]
        PR_3_m, PR_3_up, PR_3_low = results_PR[2]
        PR_4_m, PR_4_up, PR_4_low = results_PR[3]

        print('PR_1_m = %1.2e    PR_1_low = %1.2e    PR_1_up = %1.2e'%(PR_1_m, PR_1_low, PR_1_up))
        print('PR_2_m = %1.2e    PR_2_low = %1.2e    PR_2_up = %1.2e'%(PR_2_m, PR_2_low, PR_2_up))
        print('PR_3_m = %1.2e    PR_3_low = %1.2e    PR_3_up = %1.2e'%(PR_3_m, PR_3_low, PR_3_up))
        print('PR_4_m = %1.2e    PR_4_low = %1.2e    PR_4_up = %1.2e'%(PR_4_m, PR_4_low, PR_4_up))

        return icl, gal, F_ICL_m, F_ICL_low, F_ICL_up, F_gal_m, F_gal_low, F_gal_up, f_ICL_m, f_ICL_low, f_ICL_up, PR_1_m, PR_1_up, PR_1_low, PR_2_m, PR_2_up, PR_2_low, PR_3_m, PR_3_up, PR_3_low, PR_4_m, PR_4_up, PR_4_low, out_sed_icl, out_sed_gal
    else:
        return icl, gal,  np.nan,    np.nan,   np.nan,  np.nan,    np.nan,   np.nan,  np.nan,    np.nan,   np.nan, np.nan,  np.nan,   np.nan, np.nan,  np.nan,   np.nan, np.nan,  np.nan,   np.nan, np.nan,  np.nan,   np.nan, [ np.nan ], [ np.nan ]

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def synthesis_wavsizesep_with_masks( nfp, lvl_sep, lvl_sep_max, lvl_sep_bcg, size_sep, size_sep_pix, xs, ys, n_levels, mscstar, mscell, mscbcg, mscsedl, R, cat_gal, rc_pix, N_err, per_err, flux_lim, kurt_filt = True, plot_vignet = False, write_fits = True, measure_PR = False ):
    '''Wavelet Separation + Spatial filtering.
    ICL --> Atoms with z > lvl_sep, with maximum coordinates within ellipse mask 'mscell' and with size > size_sep_pix.
    Galaxies --> Satellites + BCG, so a bit complicated:
        - Atoms not classified as ICL but with maximum coordinates within ellipse mask 'mscbcg'
        - Atoms within radius 'rc' of a member galaxy position
        - In case lvl_sep > 5 (arbitrary...), atoms with z > 5 and within 'mscell' are BCG
    Unclassified --> rest of atoms
    '''
    # path, list & variables
    icl = np.zeros( (xs, ys) )
    icl_dei = np.zeros( (xs, ys) )
    gal = np.zeros( (xs, ys) )
    gal_dei = np.zeros( (xs, ys) )
    im_art = np.zeros( (xs, ys) )
    im_unclass = np.zeros( (xs, ys) )
    im_unclass_dei = np.zeros( (xs, ys) )

    tot_icl_al = []
    tot_gal_al = []
    #tot_noticl_al = []
    tot_unclass_al = []

    #%
    at_test = []
    #%

    xc = xs / 2.
    yc = ys / 2.

    ######################################## MEMORY v
    opath = nfp + '*ol.it*.hdf5'
    itpath = nfp + '*itl.it*.hdf5'
    opathl = glob.glob(opath)
    opathl.sort()

    # Interscale tree lists
    itpathl = glob.glob(itpath)
    itpathl.sort()
    
    for i, ( op, itlp ) in enumerate( zip( opathl, itpathl )):
        print('Iteration %d' %(i), end ='\r')
        
        #ol = d.store_objects.read_ol_from_hdf5(op)
        #itl = d.store_objects.read_itl_from_hdf5(itlp)
        with h5py.File(op, "r") as f1, h5py.File(itlp, "r") as f2:
            gc.collect()
            icl_al = []
            gal_al = []
            noticl_al = []
            unclass_al = []
            for o, it in zip(f1.keys(), f2.keys()):
                ######################################## MEMORY ^

                x_min, y_min, x_max, y_max = f1[o]['bbox'][()]
                image = f1[o]['image'][()]
                det_err_image = f1[o]['det_err_image'][()]
                itm = f2[it]['interscale_maximum']
                xco = itm['x_max'][()]
                yco = itm['y_max'][()]
                lvlo = f1[o]['level'][()]
                sx = x_max - x_min
                sy = y_max - y_min
            
                if kurt_filt == True:
                    k = kurtosis(image.flatten(), fisher=True)
                    if k < 0:
                        im_art[ x_min : x_max, y_min : y_max ] += image
                        continue
        
                # Remove background
                if lvlo >= lvl_sep_max:
                    tot_unclass_al.append([image, det_err_image, x_min, y_min, x_max, y_max, xco, yco, lvlo])
                    continue
        
                # Only atoms within analysis radius
                dR = np.sqrt( (xc - xco)**2 + (yc - yco)**2 )
                if dR > R:
                    continue

                # ICL
                if (mscstar[xco, yco] != 1) & (mscell[xco, yco] == 1):
                
                    if (lvlo >= lvl_sep) & (sx >= size_sep_pix) & (sy >= size_sep_pix):
                        icl[ x_min : x_max, y_min : y_max ] += image
                        icl_dei[ x_min : x_max, y_min : y_max ] += det_err_image
                        icl_al.append([image, det_err_image, x_min, y_min, x_max, y_max, xco, yco, lvlo])
                    else:
                        noticl_al.append([image, det_err_image, x_min, y_min, x_max, y_max, xco, yco, lvlo])
                else:
                    noticl_al.append([image, det_err_image, x_min, y_min, x_max, y_max, xco, yco, lvlo])

            # Galaxies
            for j, (image, det_err_image, x_min, y_min, x_max, y_max, xco, yco, lvlo) in enumerate(noticl_al):
        
                if mscstar[xco, yco] != 1:
        
                    # BCG
                    if mscbcg[xco, yco] == 1:
                        gal[ x_min : x_max, y_min : y_max ] += image
                        gal_dei[ x_min : x_max, y_min : y_max ] += det_err_image
                        gal_al.append([image, det_err_image, x_min, y_min, x_max, y_max, xco, yco, lvlo])
                        tot_gal_al.append([image, det_err_image, x_min, y_min, x_max, y_max, xco, yco, lvlo])
                        continue
        
                    # Satellites
                    if lvlo < lvl_sep:
        
                        flag = False
                        for ygal, xgal in cat_gal:
                            dr = np.sqrt( (xgal - xco)**2 + (ygal - yco)**2 )
                            if dr <= rc_pix:
                                flag = True
                                gal[ x_min : x_max, y_min : y_max ] += image
                                gal_dei[ x_min : x_max, y_min : y_max ] += det_err_image
                                gal_al.append([image, det_err_image, x_min, y_min, x_max, y_max, xco, yco, lvlo])
                                tot_gal_al.append([image, det_err_image, x_min, y_min, x_max, y_max, xco, yco, lvlo])
                                break
        
                        # If not identified as galaxies --> test if BCG again
                        if flag == False:
                            unclass_al.append([ image, det_err_image, x_min, y_min, x_max, y_max, xco, yco, lvlo ])

            # Remove potential foreground star artifacts
            #gal[mscstar == 1.] = 0.

            # Test for unclassified atoms --> sometimes extended BCG halo is missed because
            # of the nature of wavsep scheme.
            for j, (image, det_err_image, x_min, y_min, x_max, y_max, xco, yco, lvlo) in enumerate(unclass_al):
                
                # Case in which it is possible that it is BCG halo?
                if (lvl_sep > lvl_sep_bcg) & (lvlo >= lvl_sep_bcg) & (mscell[xco, yco] == 1) :
                    icl[ x_min : x_max, y_min : y_max ] += image
                    icl_dei[ x_min : x_max, y_min : y_max ] += det_err_image
                    icl_al.append([image, det_err_image, x_min, y_min, x_max, y_max, xco, yco, lvlo])
    
                #If not --> unclassified
                else:
                    im_unclass[ x_min : x_max, y_min : y_max ] += image
                    im_unclass_dei[ x_min : x_max, y_min : y_max ] += det_err_image
                    tot_unclass_al.append([image, det_err_image, x_min, y_min, x_max, y_max, xco, yco, lvlo])

    if write_fits == True:
        print('\nWS + SF + SS -- ICL -- write fits as %s*'%(nfp))

        hdu = fits.PrimaryHDU()
        hdu_icl = fits.ImageHDU(icl, name = 'ICL')
        hdu_gal = fits.ImageHDU(gal, name = 'BCG+SATELLITES')
        tot = gal + icl
        hdu_tot = fits.ImageHDU(tot, name = 'ICL+BCG+SATELLITES')
        hdu_icl_dei = fits.ImageHDU(icl_dei, name = 'ICL DET. ERR.')
        hdu_gal_dei = fits.ImageHDU(gal_dei, name = 'BCG+SAT DET. ERR.')
        tot_dei = icl_dei + gal_dei
        hdu_tot_dei = fits.ImageHDU(tot_dei, name = 'ICL+BCG+SAT DET. ERR.')
        hdu_unclass = fits.ImageHDU(im_unclass, name = 'UNCLASSIFIED')
        hdu_unclass_dei = fits.ImageHDU(im_unclass, name = 'UNCLASSIFIED DET. ERR.')

        
        hdul = fits.HDUList([ hdu, hdu_icl, hdu_gal, hdu_tot, hdu_unclass, hdu_icl_dei, hdu_gal_dei, hdu_tot_dei, hdu_unclass_dei ])
        hdul.writeto( nfp + 'synth.wavsizesepmask_%03d_%03d.fits'%(lvl_sep, size_sep), overwrite = True )


    # plot to debug --> masks & interscale maximum positions
    if plot_vignet == True:

        interval = AsymmetricPercentileInterval(5, 99.5) # meilleur rendu que MinMax or ZScale pour images reconstruites
        fig, ax = plt.subplots(2, 2)
        poim = ax[0][0].imshow(gal, norm = ImageNormalize( gal, interval = interval, stretch = LogStretch()), cmap = 'binary', origin = 'lower')
        poim = ax[1][0].imshow(icl, norm = ImageNormalize( icl, interval = interval, stretch = LogStretch()), cmap = 'binary', origin = 'lower')
        #%
        rco = pyr.open(os.path.join(path_data, 'star_flags_polygon_pix_long.reg'))
        patch_list, artist_list = rco.get_mpl_patches_texts()
        for p in patch_list:
            ax[1][0].add_patch(p)
        for a in artist_list:
            ax[1][0].add_artist(a)
        for at in at_test:
            ax[1][0].plot(at[1], at[0], 'b+')
        #%
        poim = ax[0][1].imshow(im_unclass, norm = ImageNormalize( gal, interval = interval, stretch = LogStretch()), cmap = 'binary', origin = 'lower')
        poim = ax[1][1].imshow(im_art, norm = ImageNormalize( gal, interval = interval, stretch = LogStretch()), cmap = 'binary', origin = 'lower')

        #plt.show()
        plt.tight_layout()
        plt.savefig( nfp + 'results.wavsizesepmask_%03d_%03d.png'%(lvl_sep, size_sep), format = 'png' )
        print('Write vignet to' + nfp + 'synth.wavsizesepmask_%03d_%03d.png'%(lvl_sep, size_sep))
        plt.close('all')

    if measure_PR == True:

        # Measure Fractions and uncertainties
        F_ICL_m, F_ICL_low, F_ICL_up, FICL_det_err, low_FICL_det_err, up_FICL_det_err, out_sed_icl =  selection_error(tot_icl_al, tot_unclass_al+tot_gal_al, M = N_err, percent = per_err, xs = xs, ys = ys, flux_lim = flux_lim, mscsedl = mscsedl)
        F_gal_m, F_gal_low, F_gal_up, Fgal_det_err, low_Fgal_det_err, up_Fgal_det_err, out_sed_gal =  selection_error(tot_gal_al, tot_unclass_al+tot_icl_al, M = N_err, percent = per_err, xs = xs, ys = ys, flux_lim = flux_lim, mscsedl = mscsedl)
        f_ICL_m = F_ICL_m / (F_ICL_m + F_gal_m)
        f_ICL_low = F_ICL_low / (F_ICL_low + F_gal_up)
        f_ICL_up = F_ICL_up / (F_ICL_up + F_gal_low)

        print('\nWS + SF + SS -- ICL -- z = %d    sisze_sep = %d'%(lvl_sep, size_sep))
        print('N = %4d   F_ICL = %f Mjy/sr  err_low = %f Mjy/sr  err_up = %f Mjy/sr'%(len(icl_al), F_ICL_m, F_ICL_low, F_ICL_up))
        print('N = %4d   F_gal = %f Mjy/sr  err_low = %f Mjy/sr  err_up = %f Mjy/sr'%(len(gal_al), F_gal_m, F_gal_low, F_gal_up))
        print('f_ICL = %1.3f    f_ICL_low = %1.3f   f_ICL_up = %1.3f'%(f_ICL_m, f_ICL_low, f_ICL_up))

        # Measure Power ratio
        results_PR = PR_with_selection_error(atom_in_list = icl_al, atom_out_list = tot_unclass_al+tot_gal_al, M = N_err, percent = per_err, R = R, xs = xs, ys = ys)
        PR_1_m, PR_1_up, PR_1_low = results_PR[0]
        PR_2_m, PR_2_up, PR_2_low = results_PR[1]
        PR_3_m, PR_3_up, PR_3_low = results_PR[2]
        PR_4_m, PR_4_up, PR_4_low = results_PR[3]

        print('PR_1_m = %1.2e    PR_1_low = %1.2e    PR_1_up = %1.2e'%(PR_1_m, PR_1_low, PR_1_up))
        print('PR_2_m = %1.2e    PR_2_low = %1.2e    PR_2_up = %1.2e'%(PR_2_m, PR_2_low, PR_2_up))
        print('PR_3_m = %1.2e    PR_3_low = %1.2e    PR_3_up = %1.2e'%(PR_3_m, PR_3_low, PR_3_up))
        print('PR_4_m = %1.2e    PR_4_low = %1.2e    PR_4_up = %1.2e'%(PR_4_m, PR_4_low, PR_4_up))

        return icl, gal, F_ICL_m, F_ICL_low, F_ICL_up, F_gal_m, F_gal_low, F_gal_up, f_ICL_m, f_ICL_low, f_ICL_up, PR_1_m, PR_1_up, PR_1_low, PR_2_m, PR_2_up, PR_2_low, PR_3_m, PR_3_up, PR_3_low, PR_4_m, PR_4_up, PR_4_low, out_sed_icl, out_sed_gal
    else:
        return icl, gal,  np.nan,    np.nan,   np.nan,  np.nan,    np.nan,   np.nan,  np.nan,    np.nan,   np.nan, np.nan,  np.nan,   np.nan, np.nan,  np.nan,   np.nan, np.nan,  np.nan,   np.nan, np.nan,  np.nan,   np.nan, [ np.nan ], [ np.nan ]

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
@ray.remote
def make_results_cluster( sch, oim, nfp, filt, size_sep, size_sep_pix, lvl_sep, lvl_sep_max, lvl_sep_bcg, xs, ys, n_levels, mscstar, mscell, mscbcg, mscsedl, R_kpc, R_pix, cat_gal, rc_pix, N_err, per_err, flux_lim, kurt_filt, plot_vignet, write_fits, measure_PR ):
    '''
    Runs all classification schemes for a single cluster. Performed by a single ray worker.
    '''
    hkw = ['m', 'low', 'up'] # To correctly name SED column names
    output_df = []

    # Full field ---------------------------------------------------------------
    if sch == 'fullfield':
        output = synthesis_fullfield( oim, nfp, xs, ys, write_fits )
        filler_sed = np.empty( 3 * len(mscsedl)) # fill SED data
        filler_sed[:] = np.nan
        out_sed_icl_df = pd.DataFrame([filler_sed], columns = [ 'reg_%d_%s'%(i/3, hkw[i%3]) for i in range(3 * len(mscsedl))])# create df with all SED flux for all regions with correctly numbered column names, for ICL & galaxies
        output_df = pd.DataFrame( [[ nf, filt, sch, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan ]], \
                        columns = [ 'nf', 'filter', 'Atom selection scheme', 'R_kpc', 'R_pix', 'lvl_sep', 'size_sep', 'F_ICL_m', 'F_ICL_low', 'F_ICL_up', 'F_gal_m', 'F_gal_low', 'F_gal_up', 'f_ICL_m', 'f_ICL_low', 'f_ICL_up', 'PR_1_m', 'PR_1_up', 'PR_1_low', 'PR_2_m', 'PR_2_up', 'PR_2_low', 'PR_3_m', 'PR_3_up', 'PR_3_low', 'PR_4_m', 'PR_4_up', 'PR_4_low'  ])
        
    # ICL -- WS -----------------------------------------------------------------
    if sch == 'WS':
        output = synthesis_wavsep( nfp, lvl_sep, xs, ys, n_levels, kurt_filt = kurt_filt, plot_vignet = plot_vignet, write_fits = write_fits )
        filler_sed = np.empty( 3 * len(mscsedl)) # fill SED data
        filler_sed[:] = np.nan
        out_sed_icl_df = pd.DataFrame([filler_sed], columns = [ 'reg_%d_%s'%(i/3, hkw[i%3]) for i in range(3 * len(mscsedl))])# create df with all SED flux for all regions with correctly numbered column names
        output_df = pd.DataFrame( [[ nf, filt, sch, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan ]], \
                        columns = [ 'nf', 'filter', 'Atom selection scheme', 'R_kpc', 'R_pix', 'lvl_sep', 'size_sep', 'F_ICL_m', 'F_ICL_low', 'F_ICL_up', 'F_gal_m', 'F_gal_low', 'F_gal_up', 'f_ICL_m', 'f_ICL_low', 'f_ICL_up', 'PR_1_m', 'PR_1_up', 'PR_1_low', 'PR_2_m', 'PR_2_up', 'PR_2_low', 'PR_3_m', 'PR_3_up', 'PR_3_low', 'PR_4_m', 'PR_4_up', 'PR_4_low'  ])
        
    # ICL -- WS + SF -----------------------------------------------------------
    if sch == 'WS+SF':
        output = synthesis_wavsep_with_masks( nfp = nfp, lvl_sep = lvl_sep, lvl_sep_max = lvl_sep_max, lvl_sep_bcg = lvl_sep_bcg, xs = xs, ys = ys, \
                n_levels = n_levels, mscstar = mscstar, mscell = mscell, mscbcg = mscbcg, mscsedl = mscsedl, R = R_pix, cat_gal = cat_gal, rc_pix = rc_pix,\
                N_err = N_err, per_err = per_err, flux_lim = flux_lim, kurt_filt = kurt_filt, plot_vignet = plot_vignet, write_fits = write_fits, measure_PR = measure_PR )
        F_ICL_m, F_ICL_low, F_ICL_up, F_gal_m, F_gal_low, F_gal_up, f_ICL_m, f_ICL_low, f_ICL_up, PR_1_m, PR_1_up, PR_1_low, PR_2_m, PR_2_up, PR_2_low, PR_3_m, PR_3_up, PR_3_low, PR_4_m, PR_4_up, PR_4_low = output[2:-2]
        out_sed_icl_df = pd.DataFrame( [output[-1]], columns = [ 'reg_%d_%s'%(i/3, hkw[i%3]) for i in range(len(output[-1]))]) # create df with all SED flux for all regions with correctly numbered column names
        output_df = pd.DataFrame( [[ nf, filt, sch, R_kpc, R_pix, lvl_sep, np.nan, F_ICL_m, F_ICL_low, F_ICL_up, F_gal_m, F_gal_low, F_gal_up, f_ICL_m, f_ICL_low, f_ICL_up, PR_1_m, PR_1_up, PR_1_low, PR_2_m, PR_2_up, PR_2_low, PR_3_m, PR_3_up, PR_3_low, PR_4_m, PR_4_up, PR_4_low ]], \
                        columns = [ 'nf', 'filter', 'Atom selection scheme', 'R_kpc', 'R_pix', 'lvl_sep', 'size_sep', 'F_ICL_m', 'F_ICL_low', 'F_ICL_up', 'F_gal_m', 'F_gal_low', 'F_gal_up', 'f_ICL_m', 'f_ICL_low', 'f_ICL_up', 'PR_1_m', 'PR_1_up', 'PR_1_low', 'PR_2_m', 'PR_2_up', 'PR_2_low', 'PR_3_m', 'PR_3_up', 'PR_3_low', 'PR_4_m', 'PR_4_up', 'PR_4_low'  ])
        output_df = pd.concat( [output_df, out_sed_icl_df], axis = 1)

    # ICL+BCG -- WS + SF -------------------------------------------------------
    if sch == 'WS+BCGSF':
        output = synthesis_bcgwavsep_with_masks( nfp = nfp, lvl_sep = lvl_sep, lvl_sep_max = lvl_sep_max, lvl_sep_bcg = lvl_sep_bcg, xs = xs, ys = ys, \
                n_levels = n_levels, mscstar = mscstar, mscell = mscell, mscbcg = mscbcg, mscsedl = mscsedl, R = R_pix, cat_gal = cat_gal, rc_pix = rc_pix,\
                N_err = N_err, per_err = per_err, flux_lim = flux_lim, kurt_filt = kurt_filt, plot_vignet = plot_vignet, write_fits = write_fits, measure_PR = measure_PR )
        F_ICL_m, F_ICL_low, F_ICL_up, F_gal_m, F_gal_low, F_gal_up, f_ICL_m, f_ICL_low, f_ICL_up, PR_1_m, PR_1_up, PR_1_low, PR_2_m, PR_2_up, PR_2_low, PR_3_m, PR_3_up, PR_3_low, PR_4_m, PR_4_up, PR_4_low = output[2:-2]
        out_sed_icl_df = pd.DataFrame( [output[-1]], columns = [ 'reg_%d_%s'%(i/3, hkw[i%3]) for i in range(len(output[-1]))]) # create df with all SED flux for all regions with correctly numbered column names
        output_df = pd.DataFrame( [[ nf, filt, sch, R_kpc, R_pix, lvl_sep, np.nan, F_ICL_m, F_ICL_low, F_ICL_up, F_gal_m, F_gal_low, F_gal_up, f_ICL_m, f_ICL_low, f_ICL_up, PR_1_m, PR_1_up, PR_1_low, PR_2_m, PR_2_up, PR_2_low, PR_3_m, PR_3_up, PR_3_low, PR_4_m, PR_4_up, PR_4_low ]], \
                        columns = [ 'nf', 'filter', 'Atom selection scheme', 'R_kpc', 'R_pix', 'lvl_sep', 'size_sep', 'F_ICL_m', 'F_ICL_low', 'F_ICL_up', 'F_gal_m', 'F_gal_low', 'F_gal_up', 'f_ICL_m', 'f_ICL_low', 'f_ICL_up', 'PR_1_m', 'PR_1_up', 'PR_1_low', 'PR_2_m', 'PR_2_up', 'PR_2_low', 'PR_3_m', 'PR_3_up', 'PR_3_low', 'PR_4_m', 'PR_4_up', 'PR_4_low'  ])
        output_df = pd.concat( [output_df, out_sed_icl_df], axis = 1)
        
    # ICL -- WS + SF + SS ------------------------------------------------------
    if sch == 'WS+SF+SS':
        output = synthesis_wavsizesep_with_masks( nfp = nfp, lvl_sep = lvl_sep, lvl_sep_max = lvl_sep_max, lvl_sep_bcg = lvl_sep_bcg, size_sep = size_sep, size_sep_pix =  size_sep_pix, xs = xs, ys = ys, \
                n_levels = n_levels, mscstar = mscstar, mscell = mscell, mscbcg = mscbcg, mscsedl = mscsedl, R = R_pix, cat_gal = cat_gal, rc_pix = rc_pix,\
                N_err = N_err, per_err = per_err, flux_lim = flux_lim, kurt_filt = kurt_filt, plot_vignet = plot_vignet, write_fits = write_fits, measure_PR = measure_PR )
        F_ICL_m, F_ICL_low, F_ICL_up, F_gal_m, F_gal_low, F_gal_up, f_ICL_m, f_ICL_low, f_ICL_up, PR_1_m, PR_1_up, PR_1_low, PR_2_m, PR_2_up, PR_2_low, PR_3_m, PR_3_up, PR_3_low, PR_4_m, PR_4_up, PR_4_low = output[2:-2]
        out_sed_icl_df = pd.DataFrame( [output[-1]], columns = [ 'reg_%d_%s'%(i/3, hkw[i%3]) for i in range(len(output[-1]))]) # create df with all SED flux for all regions with correctly numbered column names
        output_df = pd.DataFrame( [[ nf, filt, sch, R_kpc, R_pix, lvl_sep, size_sep, F_ICL_m, F_ICL_low, F_ICL_up, F_gal_m, F_gal_low, F_gal_up, f_ICL_m, f_ICL_low, f_ICL_up, PR_1_m, PR_1_up, PR_1_low, PR_2_m, PR_2_up, PR_2_low, PR_3_m, PR_3_up, PR_3_low, PR_4_m, PR_4_up, PR_4_low ]], \
                        columns = [ 'nf', 'filter', 'Atom selection scheme', 'R_kpc', 'R_pix', 'lvl_sep', 'size_sep','F_ICL_m', 'F_ICL_low', 'F_ICL_up', 'F_gal_m', 'F_gal_low', 'F_gal_up', 'f_ICL_m', 'f_ICL_low', 'f_ICL_up', 'PR_1_m', 'PR_1_up', 'PR_1_low', 'PR_2_m', 'PR_2_up', 'PR_2_low', 'PR_3_m', 'PR_3_up', 'PR_3_low', 'PR_4_m', 'PR_4_up', 'PR_4_low'  ])
        output_df = pd.concat( [output_df, out_sed_icl_df], axis = 1)

    # ICL+BCG -- WS + SF + SS --------------------------------------------------
    if sch == 'WS+BCGSF+SS':
        output = synthesis_bcgwavsizesep_with_masks( nfp = nfp, size_sep = size_sep, size_sep_pix = size_sep_pix, lvl_sep = lvl_sep, lvl_sep_max = lvl_sep_max, lvl_sep_bcg = lvl_sep_bcg, xs = xs, ys = ys, \
                n_levels = n_levels, mscstar = mscstar, mscell = mscell, mscbcg = mscbcg, mscsedl = mscsedl, R = R_pix, cat_gal = cat_gal, rc_pix = rc_pix,\
                N_err = N_err, per_err = per_err, flux_lim = flux_lim, kurt_filt = kurt_filt, plot_vignet = plot_vignet, write_fits = write_fits, measure_PR = measure_PR )
        F_ICL_m, F_ICL_low, F_ICL_up, F_gal_m, F_gal_low, F_gal_up, f_ICL_m, f_ICL_low, f_ICL_up, PR_1_m, PR_1_up, PR_1_low, PR_2_m, PR_2_up, PR_2_low, PR_3_m, PR_3_up, PR_3_low, PR_4_m, PR_4_up, PR_4_low = output[2:-2]
        out_sed_icl_df = pd.DataFrame( [output[-1]], columns = [ 'reg_%d_%s'%(i/3, hkw[i%3]) for i in range(len(output[-1]))]) # create df with all SED flux for all regions with correctly numbered column names
        output_df = pd.DataFrame( [[ nf, filt, sch, R_kpc, R_pix, lvl_sep, np.nan, F_ICL_m, F_ICL_low, F_ICL_up, F_gal_m, F_gal_low, F_gal_up, f_ICL_m, f_ICL_low, f_ICL_up, PR_1_m, PR_1_up, PR_1_low, PR_2_m, PR_2_up, PR_2_low, PR_3_m, PR_3_up, PR_3_low, PR_4_m, PR_4_up, PR_4_low ]], \
                        columns = [ 'nf', 'filter', 'Atom selection scheme', 'R_kpc', 'R_pix', 'lvl_sep', 'size_sep', 'F_ICL_m', 'F_ICL_low', 'F_ICL_up', 'F_gal_m', 'F_gal_low', 'F_gal_up', 'f_ICL_m', 'f_ICL_low', 'f_ICL_up', 'PR_1_m', 'PR_1_up', 'PR_1_low', 'PR_2_m', 'PR_2_up', 'PR_2_low', 'PR_3_m', 'PR_3_up', 'PR_3_low', 'PR_4_m', 'PR_4_up', 'PR_4_low'  ])
        output_df = pd.concat( [output_df, out_sed_icl_df], axis = 1)

    return output_df

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if __name__ == '__main__':

    # Paths, lists & variables
    path_data = '/n03data/ellien/JWST/data/'
    path_scripts = '/n03data/ellien/JWST/JWST_scripts'
    path_wavelets = '/n03data/ellien/JWST/wavelets/out20/'
    path_plots = '/n03data/ellien/JWST/plots'
    path_analysis = '/home/ellien/JWST/analysis/'
    
    '''path_data = '/home/aellien/JWST/data/'
    path_scripts = '/home/aellien/JWST/JWST_scripts'
    path_wavelets = '/home/aellien/JWST/wavelets/out20/'
    path_plots = '/home/aellien/JWST/plots'
    path_analysis = '/home/aellien/JWST/analysis/'
    '''
    nfl = [ {'nf':'jw02736001001_f090w_bkg_rot_crop_warp_nobkg2.fits', 'chan':'short', 'pix_scale':0.063, 'pixar_sr':9.31E-14, 'n_levels':11, 'lvl_sep_max':8, 'mu_lim':999 }, \
            {'nf':'jw02736001001_f150w_bkg_rot_crop_warp_nobkg2.fits', 'chan':'short', 'pix_scale':0.063, 'pixar_sr':9.31E-14, 'n_levels':11, 'lvl_sep_max':999, 'mu_lim':999 }, \
            {'nf':'jw02736001001_f200w_bkg_rot_crop_warp_nobkg2.fits', 'chan':'short', 'pix_scale':0.063, 'pixar_sr':9.31E-14, 'n_levels':11, 'lvl_sep_max':999, 'mu_lim':999 }, \
            {'nf':'jw02736001001_f356w_bkg_rot_crop_input.fits', 'chan':'long', 'pix_scale':0.063, 'pixar_sr':9.31E-14, 'n_levels':11, 'lvl_sep_max':999, 'mu_lim':999 }, \
            {'nf':'jw02736001001_f444w_bkg_rot_crop_input.fits', 'chan':'long', 'pix_scale':0.063, 'pixar_sr':9.31E-14, 'n_levels':11, 'lvl_sep_max':999, 'mu_lim':999 }, \
            {'nf':'jw02736001001_f277w_bkg_rot_crop_input.fits', 'chan':'long', 'pix_scale':0.063, 'pixar_sr':9.31E-14, 'n_levels':11, 'lvl_sep_max':999, 'mu_lim':999 } ]

            # out13
            #{'nf':'jw02736001001_f090w_bkg_rot_crop_warp_det_nosky_input.fits', 'chan':'long', 'pix_scale':0.063, 'n_levels':10, 'lvl_sep_max':999 }, \
            #{'nf':'jw02736001001_f150w_bkg_rot_crop_warp_det_nosky_input.fits', 'chan':'long', 'pix_scale':0.063, 'n_levels':10, 'lvl_sep_max':999 }, \
            #{'nf':'jw02736001001_f200w_bkg_rot_crop_warp_det_nosky_input.fits', 'chan':'long', 'pix_scale':0.063, 'n_levels':10, 'lvl_sep_max':999 } ]

            # out12
            #{'nf':'jw02736001001_f090w_bkg_rot_crop_input.fits', 'chan':'short', 'pix_scale':0.031, 'n_levels':11, 'lvl_sep_max':8 }, \
            #{'nf':'jw02736001001_f150w_bkg_rot_crop_input.fits', 'chan':'short', 'pix_scale':0.031, 'n_levels':11, 'lvl_sep_max':8 }, \
            #{'nf':'jw02736001001_f200w_bkg_rot_crop_input.fits', 'chan':'short', 'pix_scale':0.031, 'n_levels':11, 'lvl_sep_max':8 }

    lvl_sepl = [ 3, 4, 5, 6, 7 ] # wavelet scale separation
    size_sepl = [ 60, 80, 100, 140, 200 ] # size separation [kpc]
    R_kpcl = [ 128, 200, 400 ] # radius in which quantities are measured [kpc]
    physcale = 5.3 # kpc/"
    gamma = 0.5
    lvl_sep_big = 5
    lvl_sep_bcg = 6
    rm_gamma_for_big = True
    coo_spur_halo = [ [1615, 1665], [1685, 1480], [530, 260] ] # pix long, ds9 convention

    rc = 10 # kpc, distance to center to be classified as gal
    N_err = 100
    per_err = 0.1

    sed_n_ann = 10 # number of annuli regions, SED
    sed_n_str = 6 # number of tidal stream regions, SED

    kurt_filt = True
    plot_vignet = False
    write_fits = True
    measure_PR = True
    write_dataframe = True
    resume = True # set to true to start from where it stoped

    results = []
    ray_refs = []
    ray_outputs = []

    # ray hyperparameters
    n_cpus = 8
    ray.init(num_cpus = n_cpus)
    print('Ray OK.')
    
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

    # Read SED extraction regions
    rsedl = []
    for i in range(1, sed_n_ann + 1):
        rsedl.append(pyr.open(os.path.join(path_data, 'ellipse_annuli_pix_long_%d.reg'%i)))
    for i in range(1, sed_n_str + 1):
        rsedl.append(pyr.open(os.path.join(path_data, 'streams_flags_pix_long_%d.reg'%i)))
    rsedl.append(rell)

    # Masks LONG
    hdu = fits.open(os.path.join(path_data, 'jw02736001001_f277w_bkg_rot_crop_input.fits')) # Arbitrary
    mscell = rell.get_mask(hdu = hdu[0]) # not python convention
    mscstar = rstar.get_mask(hdu = hdu[0]) # not python convention
    mscbcg = rbcg.get_mask(hdu = hdu[0]) # not python convention
    mscsedl = [] # SED
    for rsed in rsedl:
        msc = rsed.get_mask(hdu = hdu[0])
        mscsedl.append(msc)
    id_mscsedl_long = ray.put(mscsedl)
    id_mscell_long = ray.put(mscell)
    id_mscstar_long = ray.put(mscstar)
    id_mscbcg_long = ray.put(mscbcg)
    
    # Masks SHORT
    hdu = fits.open(os.path.join(path_data, 'jw02736001001_f200w_bkg_rot_crop_warp_nobkg2.fits')) # Arbitrary
    mscell = rell.get_mask(hdu = hdu[0]) # not python convention
    mscstar = rstar.get_mask(hdu = hdu[0]) # not python convention
    mscbcg = rbcg.get_mask(hdu = hdu[0]) # not python convention
    mscsedl = [] # SED
    for rsed in rsedl:
        msc = rsed.get_mask(hdu = hdu[0])
        mscsedl.append(msc)
    id_mscsedl_short = ray.put(mscsedl)
    id_mscell_short = ray.put(mscell)
    id_mscstar_short = ray.put(mscstar)
    id_mscbcg_short = ray.put(mscbcg)

    for R_kpc in R_kpcl:

        # Iterate over dictionary list
        for nfd in nfl:

            # Read image data from dic
            nf = nfd['nf']
            filt = nf.split('_')[1]
            n_levels = nfd['n_levels']
            pix_scale = nfd['pix_scale']
            lvl_sep_max = nfd['lvl_sep_max']
            pixar_sr = nfd['pixar_sr']
            mu_lim = nfd['mu_lim']
            rc_pix = rc / physcale / pix_scale # pixels
            R_pix = R_kpc / physcale / pix_scale # pixels
            id_R_pix = ray.put(R_pix)
            print(nf)
            
            if nfd['chan'] == 'long':
                id_mscsedl = id_mscsedl_long
                id_mscell = id_mscell_long
                id_mscstar = id_mscstar_long
                id_mscbcg = id_mscbcg_long
            elif nfd['chan'] == 'short':
                id_mscsedl = id_mscsedl_short
                id_mscell = id_mscell_short
                id_mscstar = id_mscstar_short
                id_mscbcg = id_mscbcg_short

            # Photometry for limiting depth
            ZP_AB = -6.10 - 2.5 * np.log10(pixar_sr)
            flux_lim = 10**( (ZP_AB - mu_lim) / 2.5 )

            # Read image file
            nfp = os.path.join( path_wavelets, nf[:-4] )
            oim_file = os.path.join( path_data, nf )
            hdu = fits.open(oim_file)
            oim = hdu[0].data
            id_oim = ray.put(oim)
            xs, ys = oim.shape

            # Full field ------------------------------------------------
            lvl_sep = np.nan
            size_sep = np.nan
            size_sep_pix = np.nan
            
            if (resume == False) or (os.path.isfile(nfp + 'synth.full_field.fits') == False):
                ray_refs.append( make_results_cluster.remote(sch = 'fullfield', \
                                                 oim = id_oim, \
                                                 nfp = nfp, \
                                                 filt = filt, \
                                                 lvl_sep = lvl_sep, \
                                                 lvl_sep_max = lvl_sep_max, \
                                                 lvl_sep_bcg = lvl_sep_bcg, \
                                                 size_sep = size_sep, \
                                                 size_sep_pix = size_sep_pix, \
                                                 xs = xs, \
                                                 ys = ys, \
                                                 n_levels = n_levels, \
                                                 mscstar = id_mscstar, \
                                                 mscell = id_mscell, \
                                                 mscbcg = id_mscbcg, \
                                                 mscsedl = id_mscsedl, \
                                                 R_pix = id_R_pix, \
                                                 R_kpc = R_kpc,\
                                                 cat_gal = cat_gal, \
                                                 rc_pix = rc_pix,\
                                                 N_err = N_err, \
                                                 per_err = per_err, \
                                                 flux_lim = flux_lim, \
                                                 kurt_filt = kurt_filt, \
                                                 plot_vignet = plot_vignet, \
                                                 write_fits = write_fits, \
                                                 measure_PR = measure_PR ))
                        
            
            # ICL -- WS ------------------------------------------------
            for lvl_sep in lvl_sepl:
                if (resume == False) or (os.path.isfile(nfp + 'synth.wavsep_%03d.fits'%lvl_sep) == False):
                    ray_refs.append( make_results_cluster.remote(sch = 'WS', \
                                                oim = id_oim, \
                                                nfp = nfp, \
                                                filt = filt, \
                                                lvl_sep = lvl_sep, \
                                                lvl_sep_max = lvl_sep_max, \
                                                lvl_sep_bcg = lvl_sep_bcg, \
                                                size_sep = size_sep, \
                                                size_sep_pix = size_sep_pix, \
                                                xs = xs, \
                                                ys = ys, \
                                                n_levels = n_levels, \
                                                mscstar = id_mscstar, \
                                                mscell = id_mscell, \
                                                mscbcg = id_mscbcg, \
                                                mscsedl = id_mscsedl, \
                                                R_pix = id_R_pix, \
                                                R_kpc = R_kpc,\
                                                cat_gal = cat_gal, \
                                                rc_pix = rc_pix,\
                                                N_err = N_err, \
                                                per_err = per_err, \
                                                flux_lim = flux_lim, \
                                                kurt_filt = kurt_filt, \
                                                plot_vignet = plot_vignet, \
                                                write_fits = write_fits, \
                                                measure_PR = measure_PR ))

            # ICL -- WS + SF -------------------------------------------
            for lvl_sep in lvl_sepl:
                if (resume == False) or (os.path.isfile(nfp + 'synth.wavsepmask_%03d.fits'%(lvl_sep)) == False):
                    ray_refs.append( make_results_cluster.remote(sch = 'WS+SF', \
                                                oim = id_oim, \
                                                nfp = nfp, \
                                                filt = filt, \
                                                lvl_sep = lvl_sep, \
                                                lvl_sep_max = lvl_sep_max, \
                                                lvl_sep_bcg = lvl_sep_bcg, \
                                                size_sep = size_sep, \
                                                size_sep_pix = size_sep_pix, \
                                                xs = xs, \
                                                ys = ys, \
                                                n_levels = n_levels, \
                                                mscstar = id_mscstar, \
                                                mscell = id_mscell, \
                                                mscbcg = id_mscbcg, \
                                                mscsedl = id_mscsedl, \
                                                R_kpc = R_kpc,\
                                                R_pix = id_R_pix, \
                                                cat_gal = cat_gal, \
                                                rc_pix = rc_pix,\
                                                N_err = N_err, \
                                                per_err = per_err, \
                                                flux_lim = flux_lim, \
                                                kurt_filt = kurt_filt, \
                                                plot_vignet = plot_vignet, \
                                                write_fits = write_fits, \
                                                measure_PR = measure_PR ))
            
            # ICL+BCG -- WS + SF ---------------------------------------
            for lvl_sep in lvl_sepl:
                if (resume == False) or (os.path.isfile(nfp + 'synth.bcgavsepmask_%03d.fits'%(lvl_sep)) == False):
                    ray_refs.append( make_results_cluster.options(memory = 8 * 1024 * 1024 * 1024).remote(sch = 'WS+BCGSF', \
                                                oim = oim, \
                                                nfp = nfp, \
                                                filt = filt, \
                                                lvl_sep = lvl_sep, \
                                                lvl_sep_max = lvl_sep_max, \
                                                lvl_sep_bcg = lvl_sep_bcg, \
                                                size_sep = size_sep, \
                                                size_sep_pix = size_sep_pix, \
                                                xs = xs, \
                                                ys = ys, \
                                                n_levels = n_levels, \
                                                mscstar = id_mscstar, \
                                                mscell = id_mscell, \
                                                mscbcg = id_mscbcg, \
                                                mscsedl = id_mscsedl, \
                                                R_kpc = R_kpc,\
                                                R_pix = id_R_pix, \
                                                cat_gal = cat_gal, \
                                                rc_pix = rc_pix,\
                                                N_err = N_err, \
                                                per_err = per_err, \
                                                flux_lim = flux_lim, \
                                                kurt_filt = kurt_filt, \
                                                plot_vignet = plot_vignet, \
                                                write_fits = write_fits, \
                                                measure_PR = measure_PR ))
            
            # ICL -- WS + SF + SS --------------------------------------
            for lvl_sep in lvl_sepl:
                for size_sep in size_sepl:
                    size_sep_pix = size_sep / physcale / pix_scale # pixels
                    if (resume == False) or (os.path.isfile(nfp + 'synth.wavsizesepmask_%03d_%03d.fits'%(lvl_sep, size_sep)) == False):
                        ray_refs.append( make_results_cluster.remote(sch = 'WS+SF+SS', \
                                                    oim = oim, \
                                                    nfp = nfp, \
                                                    filt = filt, \
                                                    lvl_sep = lvl_sep, \
                                                    lvl_sep_max = lvl_sep_max, \
                                                    lvl_sep_bcg = lvl_sep_bcg, \
                                                    size_sep = size_sep, \
                                                    size_sep_pix = size_sep_pix, \
                                                    xs = xs, \
                                                    ys = ys, \
                                                    n_levels = n_levels, \
                                                    mscstar = id_mscstar, \
                                                    mscell = id_mscell, \
                                                    mscbcg = id_mscbcg, \
                                                    mscsedl = id_mscsedl, \
                                                    R_kpc = R_kpc,\
                                                    R_pix = id_R_pix, \
                                                    cat_gal = cat_gal, \
                                                    rc_pix = rc_pix,\
                                                    N_err = N_err, \
                                                    per_err = per_err, \
                                                    flux_lim = flux_lim, \
                                                    kurt_filt = kurt_filt, \
                                                    plot_vignet = plot_vignet, \
                                                    write_fits = write_fits, \
                                                    measure_PR = measure_PR ))
            
            # ICL+BCG -- WS + SF + SS ----------------------------------
            for lvl_sep in lvl_sepl:
                for size_sep in size_sepl:
                    size_sep_pix = size_sep / physcale / pix_scale # pixels
                    if (resume == False) or (os.path.isfile(nfp + 'synth.bcgwavsizesepmask_%03d_%03d.fits'%(lvl_sep, size_sep)) == False):
                        ray_refs.append( make_results_cluster.options(memory = 8 * 1024 * 1024 * 1024).remote(sch = 'WS+BCGSF+SS', \
                                                    oim = oim, \
                                                    nfp = nfp, \
                                                    filt = filt, \
                                                    lvl_sep = lvl_sep, \
                                                    lvl_sep_max = lvl_sep_max, \
                                                    lvl_sep_bcg = lvl_sep_bcg, \
                                                    size_sep = size_sep, \
                                                    size_sep_pix = size_sep_pix, \
                                                    xs = xs, \
                                                    ys = ys, \
                                                    n_levels = n_levels, \
                                                    mscstar = id_mscstar, \
                                                    mscell = id_mscell, \
                                                    mscbcg = id_mscbcg, \
                                                    mscsedl = id_mscsedl, \
                                                    R_kpc = R_kpc, \
                                                    R_pix = id_R_pix, \
                                                    cat_gal = cat_gal, \
                                                    rc_pix = rc_pix,\
                                                    N_err = N_err, \
                                                    per_err = per_err, \
                                                    flux_lim = flux_lim, \
                                                    kurt_filt = kurt_filt, \
                                                    plot_vignet = plot_vignet, \
                                                    write_fits = write_fits, \
                                                    measure_PR = measure_PR ))
    # Collect ray outputs
    for ref in ray_refs:
        ray_outputs.append(ray.get(ref))

    ray.shutdown()

    if write_dataframe == True:
        results_df = ray_outputs[0]
        for output_df in ray_outputs[1:]:
            results_df = pd.concat( [ results_df, output_df], ignore_index = True )
        ofp = os.path.join(path_analysis, 'results_out5b.xlsx')
        print('Write results to %s'%ofp)
        results_df.to_excel(ofp)

    print('Done')
