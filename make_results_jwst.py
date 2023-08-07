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
from skimage.morphology import binary_dilation
from scipy.stats import kurtosis

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def read_image_atoms( nfp, filter_it = None, verbose = False ):

    # Object lists
    if filter_it == None:
        opath = nfp + '*ol.it*.pkl'
        itpath = nfp + '*itl.it*.pkl'
    else:
        opath = nfp + '*ol.it' + filter_it  + '.pkl'
        itpath = nfp + '*itl.it' + filter_it + '.pkl'

    opathl = glob.glob(opath)
    opathl.sort()

    # Interscale tree lists

    itpathl = glob.glob(itpath)
    itpathl.sort()

    tol = []
    titl = []

    if verbose:
        print('Reading %s.'%(opath))
        print('Reading %s.'%(itpath))

    for i, ( op, itlp ) in enumerate( zip( opathl, itpathl )):

        if verbose :
            print('Iteration %d' %(i), end ='\r')

        ol = d.read_objects_from_pickle( op )
        itl = d.read_interscale_trees_from_pickle( itlp )

        for j, o in enumerate(ol):

            tol.append(o)
            titl.append(itl[j])

    return tol, titl

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def synthesis_fullfield( oim, nfp, gamma, lvl_sep_big, xs, ys, n_levels, rm_gamma_for_big = True ):
    '''Synthesis of the full astronomical field (e.g. sum of all atoms)
    --- Args:
    oim         # Original astronomical field
    nfp         # root path of *.pkl
    gamma       # attenuation factor
    lvl_sep_big # wavelet scale at which gamma set to 1
    lvl_sep     # wavelet scale threshold for the separation
    xs, ys      # image size
    n_levels    # number of wavelet scales
    plot_vignet # plot pdf vignet of output
    --- Output:
    rec         # synthesis image with all atoms
    res         # residuals (original - rec)
    '''
    # path, list & variables
    res = np.zeros( (xs, ys) )
    rec = np.zeros( (xs, ys) )
    wei = np.zeros( (xs, ys) )
    xc = xs / 2.
    yc = ys / 2.

    # Read atoms
    ol, itl = read_image_atoms( nfp, verbose = False )

    for j, o in enumerate(ol):

        lvlo = o.level
        x_min, y_min, x_max, y_max = o.bbox

        if (o.level >= lvl_sep_big) & (rm_gamma_for_big == True):
            rec[ x_min : x_max, y_min : y_max ] += o.image
        else:
            rec[ x_min : x_max, y_min : y_max ] += o.image * gamma

        # atom weight map
        o.image[o.image > 0.] = 1.
        wei[ x_min : x_max, y_min : y_max ] += o.image

    res = oim - rec

    hduo = fits.PrimaryHDU(rec)
    hduo.writeto( nfp + 'synth.restored.fits', overwrite = True )

    hduo = fits.PrimaryHDU(res)
    hduo.writeto( nfp + 'synth.residuals.fits', overwrite = True )

    hduo = fits.PrimaryHDU(wei)
    hduo.writeto( nfp + 'synth.weight.fits', overwrite = True )

    return rec, res, wei

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def selection_error(atom_in_list, atom_out_list, M, percent, lvl_sep_big, gamma):
    '''Computation of classification error on flux.
    '''
    size_sample = np.random.uniform(low = int( len(atom_in_list) * (1. - percent)), \
                               high = int( len(atom_in_list) + len(atom_in_list) * percent ), \
                               size = M).astype(int)
    replace_sample = []
    for s in size_sample:
        replace_sample.append(int(np.random.uniform(low = 0, high = int( s * percent ))))
    replace_sample = np.array(replace_sample)

    flux_sample = []
    for s, r in zip(size_sample, replace_sample):

        if s < len(atom_in_list):

            flux = 0
            draw = random.sample(atom_in_list, s)
            for (o, xco, yco) in draw:
                if o.level >= lvl_sep_big:
                    flux += np.sum(o.image)
                else:
                    flux += np.sum(o.image) * gamma
            flux_sample.append(flux)

        if s >= len(atom_in_list):

            flux = 0
            draw1 = random.sample(atom_in_list, len(atom_in_list) - r)
            draw2 = random.sample(atom_out_list, s - len(atom_in_list) + r)
            draw = draw1 + draw2
            for (o, xco, yco) in draw:
                if o.level >= lvl_sep_big:
                    flux += np.sum(o.image)
                else:
                    flux += np.sum(o.image) * gamma
            flux_sample.append(flux)

    flux_sample = np.array(flux_sample)
    mean_flux = np.median(flux_sample)
    up_err = np.percentile(flux_sample, 95)
    low_err = np.percentile(flux_sample, 5)

    #plt.figure()
    #plt.hist(flux_sample, bins = 10)
    #plt.show()

    return mean_flux, low_err, up_err

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def PR_with_selection_error(atom_in_list, atom_out_list, M, percent, lvl_sep_big, gamma, R, xs, ys):
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
            for (o, xco, yco) in draw:
                x_min, y_min, x_max, y_max = o.bbox
                if o.level >= lvl_sep_big:
                    im[ x_min : x_max, y_min : y_max ] += o.image
                else:
                    im[ x_min : x_max, y_min : y_max ] += o.image * gamma
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
            for (o, xco, yco) in draw:
                x_min, y_min, x_max, y_max = o.bbox
                if o.level >= lvl_sep_big:
                    im[ x_min : x_max, y_min : y_max ] += o.image
                else:
                    im[ x_min : x_max, y_min : y_max ] += o.image * gamma
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
def synthesis_wavsep( nfp, gamma, lvl_sep_big, lvl_sep, xs, ys, n_levels, rm_gamma_for_big = True, kurt_filt = False, plot_vignet = False ):
    '''Simple separation based on wavelet scale, given by parameter 'lvl_sep'.
    --- Args:
    nfp         # root path of *.pkl
    gamma       # attenuation factor
    lvl_sep_big # wavelet scale at which gamma set to 1
    lvl_sep     # wavelet scale threshold for the separation
    xs, ys      # image size
    n_levels    # number of wavelet scales
    plot_vignet # plot pdf vignet of output
    --- Output:
    icl         # synthesis image with atoms at wavelet scale >= lvl_sep
    gal         # synthesis image with atoms at wavelet scale < lvl_sep
    '''
    # path, list & variables
    icl = np.zeros( (xs, ys) )
    gal = np.zeros( (xs, ys) )
    im_art = np.zeros((xs, ys))

    xc = xs / 2.
    yc = ys / 2.

    # Read atoms
    ol, itl = read_image_atoms( nfp, verbose = False )

    onb = len(ol)
    filtered_onb = 0

    for j, o in enumerate(ol):

        x_min, y_min, x_max, y_max = o.bbox

        if kurt_filt == True:
            k = kurtosis(o.image.flatten(), fisher=True)
            if k < 0:
                filtered_onb += 1
                if (o.level >= lvl_sep_big) & (rm_gamma_for_big == True):
                    im_art[ x_min : x_max, y_min : y_max ] += o.image
                else:
                    im_art[ x_min : x_max, y_min : y_max ] += o.image * gamma
                continue

        lvlo = o.level

        if (o.level >= lvl_sep_big) & (rm_gamma_for_big == True):

            if o.level >= lvl_sep:
                icl[ x_min : x_max, y_min : y_max ] += o.image
            else:
                gal[ x_min : x_max, y_min : y_max ] += o.image

        else:

            if o.level >= lvl_sep:
                icl[ x_min : x_max, y_min : y_max ] += o.image * gamma
            else:
                gal[ x_min : x_max, y_min : y_max ] += o.image * gamma

    print('Kurtosis filtered: %d/%d'%(filtered_onb,onb))

    hduo = fits.PrimaryHDU(icl)
    hduo.writeto( nfp + 'synth.icl.wavsep_%03d.fits'%lvl_sep, overwrite = True )

    hduo = fits.PrimaryHDU(gal)
    hduo.writeto( nfp + 'synth.gal.wavsep_%03d.fits'%lvl_sep, overwrite = True )

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
def synthesis_mbrgal_wavsizesep_with_masks( nfp, gamma, lvl_sep_big, lvl_sep, size_sep, size_sep_pix, xs, ys, n_levels, mscoim, cat_gal, rc_pix, rm_gamma_for_big = True, kurt_filt = True, plot_vignet = False ):
    '''Synthesis of galaxies from a catalog using separation on wavelet scale and spatial filtering.
    '''
    # path, list & variables
    gal = np.zeros( (xs, ys) )
    im_art = np.zeros( (xs, ys) )

    xc = xs / 2.
    yc = ys / 2.

    # Read atoms
    ol, itl = read_image_atoms( nfp, verbose = False )

    for j, o in enumerate(ol):

        x_min, y_min, x_max, y_max = o.bbox
        itm = itl[j].interscale_maximum
        xco = itm.x_max
        yco = itm.y_max

        if kurt_filt == True:
            k = kurtosis(o.image.flatten(), fisher=True)
            if k < 0:
                if (o.level >= lvl_sep_big) & (rm_gamma_for_big == True):
                    im_art[ x_min : x_max, y_min : y_max ] += o.image
                else:
                    im_art[ x_min : x_max, y_min : y_max ] += o.image * gamma
                continue

        lvlo = o.level
        xco = itl[j].interscale_maximum.x_max
        yco = itl[j].interscale_maximum.y_max

        # Galaxy members
        if mscoim[xco, yco] != 1:

            if o.level < lvl_sep:

                flag = False
                for ygal, xgal in cat_gal:

                    if flag == False:
                        dr = np.sqrt( (xgal - xco)**2 + (ygal - yco)**2 )
                        if dr <= rc_pix:
                            Flag = True
                            sx = x_max - x_min
                            sy = y_max - y_min
                            if (sx < size_sep_pix) & (sy < size_sep_pix):
                                if o.level >= lvl_sep_big:
                                    gal[ x_min : x_max, y_min : y_max ] += o.image
                                else:
                                    gal[ x_min : x_max, y_min : y_max ] += o.image * gamma
                    else:
                        break

    # write to fits
    hduo = fits.PrimaryHDU(gal)
    hduo.writeto( nfp + 'synth.mbrgal.wavsizesepmask_%03d_%03d.fits'%(lvl_sep, size_sep), overwrite = True )

    return gal

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def synthesis_sizesep_with_masks( nfp, gamma, lvl_sep_big, size_sep, size_sep_pix, xs, ys, n_levels, mscoim, mscell, rm_gamma_for_big = True, kurt_filt = True, plot_vignet = False ):

    # Paths, list & variables
    icl = np.zeros((xs, ys))
    gal = np.zeros((xs, ys))
    im_art = np.zeros((xs, ys))

    # Read atoms and interscale trees
    ol, itl = read_image_atoms( nfp, verbose = False )
    for j, o in enumerate(ol):

        x_min, y_min, x_max, y_max = o.bbox
        itm = itl[j].interscale_maximum
        xco = itm.x_max
        yco = itm.y_max

        if kurt_filt == True:
            k = kurtosis(o.image.flatten(), fisher=True)
            if k < 0:
                if (o.level >= lvl_sep_big) & (rm_gamma_for_big == True):
                    im_art[ x_min : x_max, y_min : y_max ] += o.image
                else:
                    im_art[ x_min : x_max, y_min : y_max ] += o.image * gamma
                continue

        sx = x_max - x_min
        sy = y_max - y_min

        if (mscoim[xco, yco] != 1) & (mscell[xco, yco] == 1):
            if o.level >= lvl_sep_big:
                if (sx >= size_sep_pix) or (sy >= size_sep_pix):
                    icl[ x_min : x_max, y_min : y_max ] += o.image
                else:
                    gal[ x_min : x_max, y_min : y_max ] += o.image
            else:
                if (sx >= size_sep_pix) or (sy >= size_sep_pix):
                    icl[ x_min : x_max, y_min : y_max ] += o.image * gamma
                else:
                    gal[ x_min : x_max, y_min : y_max ] += o.image * gamma

    hduo = fits.PrimaryHDU(icl)
    hduo.writeto( nfp + 'synth.icl.sizesepmask_%03d.fits'%size_sep, overwrite = True )

    hduo = fits.PrimaryHDU(gal)
    hduo.writeto( nfp + 'synth.gal.sizesepmask_%03d.fits'%size_sep, overwrite = True )

    if plot_vignet == True:

        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(gal, norm = ImageNormalize(gal, interval = MinMaxInterval(), stretch = LogStretch() ), cmap = 'binary')
        ax[1].imshow(icl, norm = ImageNormalize(icl, interval = MinMaxInterval(), stretch = LogStretch() ), cmap = 'binary')
        plt.tight_layout()
        plt.savefig( nfp + 'synth.sizesepmask_%03d.png'%size_sep, format = 'png' )
        plt.close('all')

    return icl, gal

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def synthesis_bcgwavsep_with_masks( nfp, gamma, lvl_sep_big, lvl_sep, lvl_sep_max, lvl_sep_bcg, xs, ys, n_levels, mscoim, mscell, mscbcg, R, cat_gal, rc_pix, N_err, per_err, rm_gamma_for_big = True, kurt_filt = True, plot_vignet = False ):
    '''Simple separation based on wavelet scale, given by parameter 'lvl_sep'.
    '''

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

    # Kurtosis + BCG + ICL
    for j, o in enumerate(ol):

        x_min, y_min, x_max, y_max = o.bbox
        itm = itl[j].interscale_maximum
        xco = itm.x_max
        yco = itm.y_max

        if kurt_filt == True:
            k = kurtosis(o.image.flatten(), fisher=True)
            if k < 0:
                if (o.level >= lvl_sep_big) & (rm_gamma_for_big == True):
                    im_art[ x_min : x_max, y_min : y_max ] += o.image
                else:
                    im_art[ x_min : x_max, y_min : y_max ] += o.image * gamma
                continue

        lvlo = o.level
        xco = itl[j].interscale_maximum.x_max
        yco = itl[j].interscale_maximum.y_max

        # Remove background
        if o.level >= lvl_sep_max:
            #unclass_al.append([o, xco, yco])
            continue

        # Only atoms within analysis radius
        dR = np.sqrt( (xc - xco)**2 + (yc - yco)**2 )
        if dR > R:
            continue

        # ICL+BCG
        if (mscoim[xco, yco] != 1) & (mscell[xco, yco] == 1):

            # BCG
            if mscbcg[xco, yco] == 1:

                if o.level >= lvl_sep_big:
                    icl[ x_min : x_max, y_min : y_max ] += o.image
                    icl_al.append([o, xco, yco])
                else:
                    icl[ x_min : x_max, y_min : y_max ] += o.image * gamma
                    icl_al.append([o, xco, yco])

            # ICL
            else:

                if o.level >= lvl_sep_big:

                    if o.level >= lvl_sep:
                        icl[ x_min : x_max, y_min : y_max ] += o.image
                        icl_al.append([o, xco, yco])
                        at_test.append([xco, yco])
                    else:
                        #gal[ x_min : x_max, y_min : y_max ] += o.image
                        noticl_al.append([o, xco, yco])

                else:

                    if o.level >= lvl_sep:
                        icl[ x_min : x_max, y_min : y_max ] += o.image * gamma
                        icl_al.append([o, xco, yco])
                        #%
                        at_test.append([xco, yco])
                        #%
                    else:
                        #gal[ x_min : x_max, y_min : y_max ] += o.image * gamma
                        noticl_al.append([ o, xco, yco ])

        else:
            noticl_al.append([ o, xco, yco ])

    # Galaxies
    for j, (o, xco, yco) in enumerate(noticl_al):

        x_min, y_min, x_max, y_max = o.bbox
        if mscoim[xco, yco] != 1:

            # BCG
            if mscbcg[xco, yco] == 1:
                if o.level >= lvl_sep_big:
                    gal[ x_min : x_max, y_min : y_max ] += o.image
                    gal_al.append([o, xco, yco])
                else:
                    gal[ x_min : x_max, y_min : y_max ] += o.image * gamma
                    gal_al.append([o, xco, yco])
                continue

            # Satellites
            if o.level < lvl_sep:

                flag = False
                for ygal, xgal in cat_gal:
                    dr = np.sqrt( (xgal - xco)**2 + (ygal - yco)**2 )
                    if dr <= rc_pix:
                        flag = True
                        if o.level >= lvl_sep_big:
                            gal[ x_min : x_max, y_min : y_max ] += o.image
                            gal_al.append([o, xco, yco])
                        else:
                            gal[ x_min : x_max, y_min : y_max ] += o.image * gamma
                            gal_al.append([o, xco, yco])
                        break

                # If not identified as galaxies --> test if BCG again
                if flag == False:
                    unclass_al.append([ o, xco, yco ])

    # Test for unclassified atoms --> sometimes extended BCG halo is missed because
    # of the nature of wavsep scheme.
    for j, (o, xco, yco) in enumerate(unclass_al):

        x_min, y_min, x_max, y_max = o.bbox

        # Case in which it is possible that it is BCG halo?
        if lvl_sep > lvl_sep_bcg:

            #BCG extended halo?
            if (o.level >= lvl_sep_bcg) & (mscell[xco, yco] == 1) :
                if o.level >= lvl_sep_big:
                    icl[ x_min : x_max, y_min : y_max ] += o.image
                    icl_al.append([o, xco, yco])
                else:
                    icl[ x_min : x_max, y_min : y_max ] += o.image * gamma
                    icl_al.append([o, xco, yco])

            #If not --> unclassified
            else:
                if o.level >= lvl_sep_big:
                    im_unclass[ x_min : x_max, y_min : y_max ] += o.image
                else:
                    im_unclass[ x_min : x_max, y_min : y_max ] += o.image * gamma

        #If not --> unclassified
        else:
            if o.level >= lvl_sep_big:
                im_unclass[ x_min : x_max, y_min : y_max ] += o.image
            else:
                im_unclass[ x_min : x_max, y_min : y_max ] += o.image * gamma

    # Remove potential foreground star artifacts
    #gal[mscoim == 1.] = 0.

    # Measure Fractions and uncertainties
    F_ICL_m, F_ICL_low, F_ICL_up =  selection_error(icl_al, unclass_al, M = N_err, percent = per_err, lvl_sep_big = lvl_sep_big, gamma = gamma)
    F_gal_m, F_gal_low, F_gal_up =  selection_error(gal_al, unclass_al, M = N_err, percent = per_err, lvl_sep_big = lvl_sep_big, gamma = gamma)
    f_ICL_m = np.sum(icl)/(np.sum(icl + gal))
    f_ICL_low = F_ICL_low / (F_ICL_low + F_gal_up)
    f_ICL_up = F_ICL_up / (F_ICL_up + F_gal_low)

    print('\nWS + SF -- ICL+BCG --  z = %d'%lvl_sep)
    print('N = %4d   F_ICL = %f Mjy/sr  err_low = %f Mjy/sr  err_up = %f Mjy/sr'%(len(icl_al), F_ICL_m, F_ICL_low, F_ICL_up))
    print('N = %4d   F_gal = %f Mjy/sr  err_low = %f Mjy/sr  err_up = %f Mjy/sr'%(len(gal_al), F_gal_m, F_gal_low, F_gal_up))
    print('f_ICL = %1.3f    f_ICL_low = %1.3f   f_ICL_up = %1.3f'%(f_ICL_m, f_ICL_low, f_ICL_up))
    #%
    at_test = np.array(at_test)
    #%

    # Measure Power ratio
    results_PR = PR_with_selection_error(atom_in_list = icl_al, atom_out_list = unclass_al, M = N_err, percent = per_err, lvl_sep_big = lvl_sep_big, gamma = gamma, R = R, xs = xs, ys = ys)
    PR_1_m, PR_1_up, PR_1_low = results_PR[0]
    PR_2_m, PR_2_up, PR_2_low = results_PR[1]
    PR_3_m, PR_3_up, PR_3_low = results_PR[2]
    PR_4_m, PR_4_up, PR_4_low = results_PR[3]

    print('PR_1_m = %1.2e    PR_1_low = %1.2e    PR_1_up = %1.2e'%(PR_1_m, PR_1_low, PR_1_up))
    print('PR_2_m = %1.2e    PR_2_low = %1.2e    PR_2_up = %1.2e'%(PR_2_m, PR_2_low, PR_2_up))
    print('PR_3_m = %1.2e    PR_3_low = %1.2e    PR_3_up = %1.2e'%(PR_3_m, PR_3_low, PR_3_up))
    print('PR_4_m = %1.2e    PR_4_low = %1.2e    PR_4_up = %1.2e'%(PR_4_m, PR_4_low, PR_4_up))

    # write to fits
    hduo = fits.PrimaryHDU(icl)
    hduo.writeto( nfp + 'synth.icl.bcgwavsepmask_%03d.fits'%lvl_sep, overwrite = True )

    hduo = fits.PrimaryHDU(gal)
    hduo.writeto( nfp + 'synth.gal.bcgwavsepmask_%03d.fits'%lvl_sep, overwrite = True )

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
        plt.savefig( nfp + 'results.bcgwavsepmask_%03d.png'%lvl_sep, format = 'png' )
        print('Write vignet to' + nfp + 'synth.bcgwavsepmask_%03d.png'%(lvl_sep))
        plt.close('all')

    return icl, gal, F_ICL_m, F_ICL_low, F_ICL_up, F_gal_m, F_gal_low, F_gal_up, f_ICL_m, f_ICL_low, f_ICL_up, PR_1_m, PR_1_up, PR_1_low, PR_2_m, PR_2_up, PR_2_low, PR_3_m, PR_3_up, PR_3_low, PR_4_m, PR_4_up, PR_4_low

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def synthesis_bcgwavsizesep_with_masks( nfp, chan, gamma, lvl_sep_big, lvl_sep, lvl_sep_max, lvl_sep_bcg, size_sep, size_sep_pix, xs, ys, n_levels, mscoim, mscell, mscbcg, R, cat_gal, rc_pix, N_err, per_err, rm_gamma_for_big = True, kurt_filt = True, plot_vignet = False ):
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

    # Kurtosis + ICL+BCG
    for j, o in enumerate(ol):

        x_min, y_min, x_max, y_max = o.bbox
        sx = x_max - x_min
        sy = y_max - y_min
        itm = itl[j].interscale_maximum
        xco = itm.x_max
        yco = itm.y_max
        lvlo = o.level

        if kurt_filt == True:
            k = kurtosis(o.image.flatten(), fisher=True)
            if k < 0:
                if (o.level >= lvl_sep_big) & (rm_gamma_for_big == True):
                    im_art[ x_min : x_max, y_min : y_max ] += o.image
                else:
                    im_art[ x_min : x_max, y_min : y_max ] += o.image * gamma
                continue

        # Remove background
        if o.level >= lvl_sep_max:
            continue

        # Only atoms within analysis radius
        dR = np.sqrt( (xc - xco)**2 + (yc - yco)**2 )
        if dR > R:
            continue

        # ICL + BCG
        if (mscoim[xco, yco] != 1) & (mscell[xco, yco] == 1):

            # BCG
            if chan == 'long':xbcg, ybcg = [ 1050, 980 ] # pix long, ds9 convention
            if chan == 'short':xbcg, ybcg = [ 2130, 2000 ] # pix short, ds9 convention
            if mscbcg[xco, yco] == 1:

                dr = np.sqrt( (xbcg - xco)**2 + (ybcg - yco)**2 )
                if (o.level <= 3) & (dr < rc_pix):

                    if o.level >= lvl_sep_big:
                        icl[ x_min : x_max, y_min : y_max ] += o.image
                        icl_al.append([o, xco, yco])
                    else:
                        icl[ x_min : x_max, y_min : y_max ] += o.image * gamma
                        icl_al.append([o, xco, yco])

                elif o.level >= 4:
                    if o.level >= lvl_sep_big:
                        icl[ x_min : x_max, y_min : y_max ] += o.image
                        icl_al.append([o, xco, yco])
                    else:
                        icl[ x_min : x_max, y_min : y_max ] += o.image * gamma
                        icl_al.append([o, xco, yco])

            # ICL
            else:
                if o.level >= lvl_sep_big:

                    if (o.level >= lvl_sep) & (sx >= size_sep_pix) & (sy >= size_sep_pix):

                        #%%%%%
                        if chan == 'long':coo_spur_halo = [ [1615, 1665], [1685, 1480], [530, 260] ] # pix long, ds9 convention
                        if chan == 'short':coo_spur_halo = [ [3300, 3375], [3345, 3000], [1100, 550] ] # pix short, ds9 convention
                        flag = False
                        for ygal, xgal in coo_spur_halo:

                            dr = np.sqrt( (xgal - xco)**2 + (ygal - yco)**2 )
                            if (dr <= rc_pix) & (o.level == 5):
                                flag = True

                        if flag == False:
                            #%
                            icl[ x_min : x_max, y_min : y_max ] += o.image
                            icl_al.append([o, xco, yco])
                            at_test.append([xco, yco])
                    else:
                        #gal[ x_min : x_max, y_min : y_max ] += o.image
                        noticl_al.append([o, xco, yco])

                else:

                    if (o.level >= lvl_sep) & (sx >= size_sep_pix) & (sy >= size_sep_pix):

                        #%%%%%
                        if chan == 'long':coo_spur_halo = [ [1615, 1665], [1685, 1480], [530, 260] ] # pix long, ds9 convention
                        if chan == 'short':coo_spur_halo = [ [3300, 3375], [3345, 3000], [1100, 550] ] # pix short, ds9 convention
                        flag = False
                        for ygal, xgal in coo_spur_halo:

                            dr = np.sqrt( (xgal - xco)**2 + (ygal - yco)**2 )
                            if (dr <= rc_pix) & (o.level == 5):
                                flag = True

                        if flag == False:
                            #%
                            icl[ x_min : x_max, y_min : y_max ] += o.image * gamma
                            icl_al.append([o, xco, yco])
                            #%
                            at_test.append([xco, yco])
                            #%
                    else:
                        noticl_al.append([ o, xco, yco ])

        else:
            noticl_al.append([ o, xco, yco ])

    # Galaxies
    for j, (o, xco, yco) in enumerate(noticl_al):

        x_min, y_min, x_max, y_max = o.bbox
        if mscoim[xco, yco] != 1:

            # Satellites
            if o.level < lvl_sep:

                flag = False
                for ygal, xgal in cat_gal:
                    dr = np.sqrt( (xgal - xco)**2 + (ygal - yco)**2 )
                    if dr <= rc_pix:
                        flag = True
                        if o.level >= lvl_sep_big:
                            gal[ x_min : x_max, y_min : y_max ] += o.image
                            gal_al.append([o, xco, yco])
                        else:
                            gal[ x_min : x_max, y_min : y_max ] += o.image * gamma
                            gal_al.append([o, xco, yco])
                        break

                # If not identified as galaxies --> test if BCG again
                if flag == False:
                    unclass_al.append([ o, xco, yco ])

    # Test for unclassified atoms --> sometimes extended BCG halo is missed because
    # of the nature of wavsep scheme.
    for j, (o, xco, yco) in enumerate(unclass_al):

        x_min, y_min, x_max, y_max = o.bbox

        # Case in which it is possible that it is BCG halo?
        if lvl_sep > lvl_sep_bcg:

            #BCG extended halo?
            if (o.level >= lvl_sep_bcg) & (mscell[xco, yco] == 1) :
                if o.level >= lvl_sep_big:
                    icl[ x_min : x_max, y_min : y_max ] += o.image
                    icl_al.append([o, xco, yco])
                else:
                    icl[ x_min : x_max, y_min : y_max ] += o.image * gamma
                    icl_al.append([o, xco, yco])

            #If not --> unclassified
            else:
                if o.level >= lvl_sep_big:
                    im_unclass[ x_min : x_max, y_min : y_max ] += o.image
                else:
                    im_unclass[ x_min : x_max, y_min : y_max ] += o.image * gamma

        #If not --> unclassified
        else:
            if o.level >= lvl_sep_big:
                im_unclass[ x_min : x_max, y_min : y_max ] += o.image
            else:
                im_unclass[ x_min : x_max, y_min : y_max ] += o.image * gamma

    # Remove potential foreground star artifacts
    #gal[mscoim == 1.] = 0.

    # Measure Fractions and uncertainties
    F_ICL_m, F_ICL_low, F_ICL_up =  selection_error(icl_al, unclass_al, M = N_err, percent = per_err, lvl_sep_big = lvl_sep_big, gamma = gamma)
    F_gal_m, F_gal_low, F_gal_up =  selection_error(gal_al, unclass_al, M = N_err, percent = per_err, lvl_sep_big = lvl_sep_big, gamma = gamma)
    f_ICL_m = F_ICL_m / (F_ICL_m + F_gal_m)
    f_ICL_low = F_ICL_low / (F_ICL_low + F_gal_up)
    f_ICL_up = F_ICL_up / (F_ICL_up + F_gal_low)

    print('\nWS + SF + SS --  z = %d    sise_sep = %d'%(lvl_sep, size_sep))
    print('N = %4d   F_ICL = %f Mjy/sr  err_low = %f Mjy/sr  err_up = %f Mjy/sr'%(len(icl_al), F_ICL_m, F_ICL_low, F_ICL_up))
    print('N = %4d   F_gal = %f Mjy/sr  err_low = %f Mjy/sr  err_up = %f Mjy/sr'%(len(gal_al), F_gal_m, F_gal_low, F_gal_up))
    print('f_ICL = %1.3f    f_ICL_low = %1.3f   f_ICL_up = %1.3f'%(f_ICL_m, f_ICL_low, f_ICL_up))

    # Measure Power ratio
    results_PR = PR_with_selection_error(atom_in_list = icl_al, atom_out_list = unclass_al, M = N_err, percent = per_err, lvl_sep_big = lvl_sep_big, gamma = gamma, R = R, xs = xs, ys = ys)
    PR_1_m, PR_1_up, PR_1_low = results_PR[0]
    PR_2_m, PR_2_up, PR_2_low = results_PR[1]
    PR_3_m, PR_3_up, PR_3_low = results_PR[2]
    PR_4_m, PR_4_up, PR_4_low = results_PR[3]

    print('PR_1_m = %1.2e    PR_1_low = %1.2e    PR_1_up = %1.2e'%(PR_1_m, PR_1_low, PR_1_up))
    print('PR_2_m = %1.2e    PR_2_low = %1.2e    PR_2_up = %1.2e'%(PR_2_m, PR_2_low, PR_2_up))
    print('PR_3_m = %1.2e    PR_3_low = %1.2e    PR_3_up = %1.2e'%(PR_3_m, PR_3_low, PR_3_up))
    print('PR_4_m = %1.2e    PR_4_low = %1.2e    PR_4_up = %1.2e'%(PR_4_m, PR_4_low, PR_4_up))

    #%
    at_test = np.array(at_test)
    #%

    # write to fits
    hduo = fits.PrimaryHDU(icl)
    hduo.writeto( nfp + 'synth.icl.bcgwavsizesepmask_%03d_%03d.fits'%(lvl_sep, size_sep), overwrite = True )

    hduo = fits.PrimaryHDU(gal)
    hduo.writeto( nfp + 'synth.gal.bcgwavsizesepmask_%03d_%03d.fits'%(lvl_sep, size_sep), overwrite = True )

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
        plt.savefig( nfp + 'results.bcgwavsizesepmask_%03d_%03d_testspur.png'%(lvl_sep, size_sep), format = 'png' )
        print('Write vignet to' + nfp + 'synth.bcgwavsizesepmask_%03d_%03d_testspur.png'%(lvl_sep, size_sep))
        plt.close('all')

    return icl, gal, F_ICL_m, F_ICL_low, F_ICL_up, F_gal_m, F_gal_low, F_gal_up, f_ICL_m, f_ICL_low, f_ICL_up, PR_1_m, PR_1_up, PR_1_low, PR_2_m, PR_2_up, PR_2_low, PR_3_m, PR_3_up, PR_3_low, PR_4_m, PR_4_up, PR_4_low

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def synthesis_wavsep_with_masks( nfp, gamma, lvl_sep_big, lvl_sep, lvl_sep_max, lvl_sep_bcg, xs, ys, n_levels, mscoim, mscell, mscbcg, R, cat_gal, rc_pix, N_err, per_err, rm_gamma_for_big = True, kurt_filt = True, plot_vignet = False ):
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
        itm = itl[j].interscale_maximum
        xco = itm.x_max
        yco = itm.y_max

        if kurt_filt == True:
            k = kurtosis(o.image.flatten(), fisher=True)
            if k < 0:
                if (o.level >= lvl_sep_big) & (rm_gamma_for_big == True):
                    im_art[ x_min : x_max, y_min : y_max ] += o.image
                else:
                    im_art[ x_min : x_max, y_min : y_max ] += o.image * gamma
                continue

        lvlo = o.level
        xco = itl[j].interscale_maximum.x_max
        yco = itl[j].interscale_maximum.y_max

        # Remove background
        if o.level >= lvl_sep_max:
            #unclass_al.append([o, xco, yco])
            continue

        # Only atoms within analysis radius
        dR = np.sqrt( (xc - xco)**2 + (yc - yco)**2 )
        if dR > R:
            continue

        # ICL
        if (mscoim[xco, yco] != 1) & (mscell[xco, yco] == 1):

            if o.level >= lvl_sep_big:

                if o.level >= lvl_sep:
                    icl[ x_min : x_max, y_min : y_max ] += o.image
                    icl_al.append([o, xco, yco])
                    at_test.append([xco, yco])
                else:
                    #gal[ x_min : x_max, y_min : y_max ] += o.image
                    noticl_al.append([o, xco, yco])

            else:

                if o.level >= lvl_sep:
                    icl[ x_min : x_max, y_min : y_max ] += o.image * gamma
                    icl_al.append([o, xco, yco])
                    #%
                    at_test.append([xco, yco])
                    #%
                else:
                    #gal[ x_min : x_max, y_min : y_max ] += o.image * gamma
                    noticl_al.append([ o, xco, yco ])

        else:
            noticl_al.append([ o, xco, yco ])

    # Galaxies
    for j, (o, xco, yco) in enumerate(noticl_al):

        x_min, y_min, x_max, y_max = o.bbox
        if mscoim[xco, yco] != 1:

            # BCG
            if mscbcg[xco, yco] == 1:
                if o.level >= lvl_sep_big:
                    gal[ x_min : x_max, y_min : y_max ] += o.image
                    gal_al.append([o, xco, yco])
                else:
                    gal[ x_min : x_max, y_min : y_max ] += o.image * gamma
                    gal_al.append([o, xco, yco])
                continue

            # Satellites
            if o.level < lvl_sep:

                flag = False
                for ygal, xgal in cat_gal:
                    dr = np.sqrt( (xgal - xco)**2 + (ygal - yco)**2 )
                    if dr <= rc_pix:
                        flag = True
                        if o.level >= lvl_sep_big:
                            gal[ x_min : x_max, y_min : y_max ] += o.image
                            gal_al.append([o, xco, yco])
                        else:
                            gal[ x_min : x_max, y_min : y_max ] += o.image * gamma
                            gal_al.append([o, xco, yco])
                        break

                # If not identified as galaxies --> test if BCG again
                if flag == False:
                    unclass_al.append([ o, xco, yco ])

    # Remove potential foreground star artifacts
    gal[mscoim == 1.] = 0.

    # Test for unclassified atoms --> sometimes extended BCG halo is missed because
    # of the nature of wavsep scheme.
    for j, (o, xco, yco) in enumerate(unclass_al):

        x_min, y_min, x_max, y_max = o.bbox

        # Case in which it is possible that it is BCG halo?
        if lvl_sep > lvl_sep_bcg:

            #BCG extended halo?
            if (o.level >= lvl_sep_bcg) & (mscell[xco, yco] == 1) :
                if o.level >= lvl_sep_big:
                    gal[ x_min : x_max, y_min : y_max ] += o.image
                    gal_al.append([o, xco, yco])
                else:
                    gal[ x_min : x_max, y_min : y_max ] += o.image * gamma
                    gal_al.append([o, xco, yco])

            #If not --> unclassified
            else:
                if o.level >= lvl_sep_big:
                    im_unclass[ x_min : x_max, y_min : y_max ] += o.image
                else:
                    im_unclass[ x_min : x_max, y_min : y_max ] += o.image * gamma

        #If not --> unclassified
        else:
            if o.level >= lvl_sep_big:
                im_unclass[ x_min : x_max, y_min : y_max ] += o.image
            else:
                im_unclass[ x_min : x_max, y_min : y_max ] += o.image * gamma

    # Measure Fractions and uncertainties
    F_ICL_m, F_ICL_low, F_ICL_up =  selection_error(icl_al, unclass_al, M = N_err, percent = per_err, lvl_sep_big = lvl_sep_big, gamma = gamma)
    F_gal_m, F_gal_low, F_gal_up =  selection_error(gal_al, unclass_al, M = N_err, percent = per_err, lvl_sep_big = lvl_sep_big, gamma = gamma)
    f_ICL_m = np.sum(icl)/(np.sum(icl + gal))
    f_ICL_low = F_ICL_low / (F_ICL_low + F_gal_up)
    f_ICL_up = F_ICL_up / (F_ICL_up + F_gal_low)

    print('\nWS + SF -- ICL -- z = %d'%lvl_sep)
    print('N = %4d   F_ICL = %f Mjy/sr  err_low = %f Mjy/sr  err_up = %f Mjy/sr'%(len(icl_al), F_ICL_m, F_ICL_low, F_ICL_up))
    print('N = %4d   F_gal = %f Mjy/sr  err_low = %f Mjy/sr  err_up = %f Mjy/sr'%(len(gal_al), F_gal_m, F_gal_low, F_gal_up))
    print('f_ICL = %1.3f    f_ICL_low = %1.3f   f_ICL_up = %1.3f'%(f_ICL_m, f_ICL_low, f_ICL_up))

    #%
    at_test = np.array(at_test)
    #%

    # Measure Power ratio
    results_PR = PR_with_selection_error(atom_in_list = icl_al, atom_out_list = unclass_al, M = N_err, percent = per_err, lvl_sep_big = lvl_sep_big, gamma = gamma, R = R, xs = xs, ys = ys)
    PR_1_m, PR_1_up, PR_1_low = results_PR[0]
    PR_2_m, PR_2_up, PR_2_low = results_PR[1]
    PR_3_m, PR_3_up, PR_3_low = results_PR[2]
    PR_4_m, PR_4_up, PR_4_low = results_PR[3]

    print('PR_1_m = %1.2e    PR_1_low = %1.2e    PR_1_up = %1.2e'%(PR_1_m, PR_1_low, PR_1_up))
    print('PR_2_m = %1.2e    PR_2_low = %1.2e    PR_2_up = %1.2e'%(PR_2_m, PR_2_low, PR_2_up))
    print('PR_3_m = %1.2e    PR_3_low = %1.2e    PR_3_up = %1.2e'%(PR_3_m, PR_3_low, PR_3_up))
    print('PR_4_m = %1.2e    PR_4_low = %1.2e    PR_4_up = %1.2e'%(PR_4_m, PR_4_low, PR_4_up))

    # write to fits
    hduo = fits.PrimaryHDU(icl)
    hduo.writeto( nfp + 'synth.icl.wavsepmask_%03d.fits'%lvl_sep, overwrite = True )

    hduo = fits.PrimaryHDU(gal)
    hduo.writeto( nfp + 'synth.gal.wavsepmask_%03d.fits'%lvl_sep, overwrite = True )

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
        plt.savefig( nfp + 'results.wavsepmask_%03d.png'%lvl_sep, format = 'png' )
        print('Write vignet to' + nfp + 'synth.wavsepmask_%03d.png'%(lvl_sep))
        plt.close('all')

    return icl, gal, F_ICL_m, F_ICL_low, F_ICL_up, F_gal_m, F_gal_low, F_gal_up, f_ICL_m, f_ICL_low, f_ICL_up, PR_1_m, PR_1_up, PR_1_low, PR_2_m, PR_2_up, PR_2_low, PR_3_m, PR_3_up, PR_3_low, PR_4_m, PR_4_up, PR_4_low

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def synthesis_wavsizesep_with_masks( nfp, gamma, lvl_sep_big, lvl_sep, lvl_sep_max, lvl_sep_bcg, size_sep, size_sep_pix, xs, ys, n_levels, mscoim, mscell, mscbcg, R, cat_gal, rc_pix, N_err, per_err, rm_gamma_for_big = True, kurt_filt = True, plot_vignet = False ):
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

        if kurt_filt == True:
            k = kurtosis(o.image.flatten(), fisher=True)
            if k < 0:
                if (o.level >= lvl_sep_big) & (rm_gamma_for_big == True):
                    im_art[ x_min : x_max, y_min : y_max ] += o.image
                else:
                    im_art[ x_min : x_max, y_min : y_max ] += o.image * gamma
                continue

        # Remove background
        if o.level >= lvl_sep_max:
            continue

        # Only atoms within analysis radius
        dR = np.sqrt( (xc - xco)**2 + (yc - yco)**2 )
        if dR > R:
            continue

        # ICL
        if (mscoim[xco, yco] != 1) & (mscell[xco, yco] == 1):

            if o.level >= lvl_sep_big:

                if (o.level >= lvl_sep) & (sx >= size_sep_pix) & (sy >= size_sep_pix):
                    icl[ x_min : x_max, y_min : y_max ] += o.image
                    icl_al.append([o, xco, yco])
                    at_test.append([xco, yco])
                else:
                    #gal[ x_min : x_max, y_min : y_max ] += o.image
                    noticl_al.append([o, xco, yco])

            else:

                if (o.level >= lvl_sep) & (sx >= size_sep_pix) & (sy >= size_sep_pix):
                    icl[ x_min : x_max, y_min : y_max ] += o.image * gamma
                    icl_al.append([o, xco, yco])
                    #%
                    at_test.append([xco, yco])
                    #%
                else:
                    noticl_al.append([ o, xco, yco ])

        else:
            noticl_al.append([ o, xco, yco ])

    # Galaxies
    for j, (o, xco, yco) in enumerate(noticl_al):

        x_min, y_min, x_max, y_max = o.bbox
        if mscoim[xco, yco] != 1:

            # BCG
            if mscbcg[xco, yco] == 1:
                if o.level >= lvl_sep_big:
                    gal[ x_min : x_max, y_min : y_max ] += o.image
                    gal_al.append([o, xco, yco])
                else:
                    gal[ x_min : x_max, y_min : y_max ] += o.image * gamma
                    gal_al.append([o, xco, yco])
                continue

            # Satellites
            if o.level < lvl_sep:

                flag = False
                for ygal, xgal in cat_gal:
                    dr = np.sqrt( (xgal - xco)**2 + (ygal - yco)**2 )
                    if dr <= rc_pix:
                        flag = True
                        if o.level >= lvl_sep_big:
                            gal[ x_min : x_max, y_min : y_max ] += o.image
                            gal_al.append([o, xco, yco])
                        else:
                            gal[ x_min : x_max, y_min : y_max ] += o.image * gamma
                            gal_al.append([o, xco, yco])
                        break

                # If not identified as galaxies --> test if BCG again
                if flag == False:
                    unclass_al.append([ o, xco, yco ])

    # Remove potential foreground star artifacts
    gal[mscoim == 1.] = 0.

    # Test for unclassified atoms --> sometimes extended BCG halo is missed because
    # of the nature of wavsep scheme.
    for j, (o, xco, yco) in enumerate(unclass_al):

        x_min, y_min, x_max, y_max = o.bbox

        # Case in which it is possible that it is BCG halo?
        if lvl_sep > lvl_sep_bcg:

            #BCG extended halo?
            if (o.level >= lvl_sep_bcg) & (mscell[xco, yco] == 1) :
                if o.level >= lvl_sep_big:
                    gal[ x_min : x_max, y_min : y_max ] += o.image
                    gal_al.append([o, xco, yco])
                else:
                    gal[ x_min : x_max, y_min : y_max ] += o.image * gamma
                    gal_al.append([o, xco, yco])

            #If not --> unclassified
            else:
                if o.level >= lvl_sep_big:
                    im_unclass[ x_min : x_max, y_min : y_max ] += o.image
                else:
                    im_unclass[ x_min : x_max, y_min : y_max ] += o.image * gamma

        #If not --> unclassified
        else:
            if o.level >= lvl_sep_big:
                im_unclass[ x_min : x_max, y_min : y_max ] += o.image
            else:
                im_unclass[ x_min : x_max, y_min : y_max ] += o.image * gamma

    # Measure Fractions and uncertainties
    F_ICL_m, F_ICL_low, F_ICL_up =  selection_error(icl_al, unclass_al, M = N_err, percent = per_err, lvl_sep_big = lvl_sep_big, gamma = gamma)
    F_gal_m, F_gal_low, F_gal_up =  selection_error(gal_al, unclass_al, M = N_err, percent = per_err, lvl_sep_big = lvl_sep_big, gamma = gamma)
    f_ICL_m = F_ICL_m / (F_ICL_m + F_gal_m)
    f_ICL_low = F_ICL_low / (F_ICL_low + F_gal_up)
    f_ICL_up = F_ICL_up / (F_ICL_up + F_gal_low)

    print('\nWS + SF + SS --  z = %d    sisze_sep = %d'%(lvl_sep, size_sep))
    print('N = %4d   F_ICL = %f Mjy/sr  err_low = %f Mjy/sr  err_up = %f Mjy/sr'%(len(icl_al), F_ICL_m, F_ICL_low, F_ICL_up))
    print('N = %4d   F_gal = %f Mjy/sr  err_low = %f Mjy/sr  err_up = %f Mjy/sr'%(len(gal_al), F_gal_m, F_gal_low, F_gal_up))
    print('f_ICL = %1.3f    f_ICL_low = %1.3f   f_ICL_up = %1.3f'%(f_ICL_m, f_ICL_low, f_ICL_up))

    # Measure Power ratio
    results_PR = PR_with_selection_error(atom_in_list = icl_al, atom_out_list = unclass_al, M = 50, percent = 0.1, lvl_sep_big = lvl_sep_big, gamma = gamma, R = R, xs = xs, ys = ys)
    PR_1_m, PR_1_up, PR_1_low = results_PR[0]
    PR_2_m, PR_2_up, PR_2_low = results_PR[1]
    PR_3_m, PR_3_up, PR_3_low = results_PR[2]
    PR_4_m, PR_4_up, PR_4_low = results_PR[3]

    print('PR_1_m = %1.2e    PR_1_low = %1.2e    PR_1_up = %1.2e'%(PR_1_m, PR_1_low, PR_1_up))
    print('PR_2_m = %1.2e    PR_2_low = %1.2e    PR_2_up = %1.2e'%(PR_2_m, PR_2_low, PR_2_up))
    print('PR_3_m = %1.2e    PR_3_low = %1.2e    PR_3_up = %1.2e'%(PR_3_m, PR_3_low, PR_3_up))
    print('PR_4_m = %1.2e    PR_4_low = %1.2e    PR_4_up = %1.2e'%(PR_4_m, PR_4_low, PR_4_up))

    #%
    at_test = np.array(at_test)
    #%

    # write to fits
    hduo = fits.PrimaryHDU(icl)
    hduo.writeto( nfp + 'synth.icl.wavsizesepmask_%03d_%03d.fits'%(lvl_sep, size_sep), overwrite = True )

    hduo = fits.PrimaryHDU(gal)
    hduo.writeto( nfp + 'synth.gal.wavsizesepmask_%03d_%03d.fits'%(lvl_sep, size_sep), overwrite = True )

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

    return icl, gal, F_ICL_m, F_ICL_low, F_ICL_up, F_gal_m, F_gal_low, F_gal_up, f_ICL_m, f_ICL_low, f_ICL_up, PR_1_m, PR_1_up, PR_1_low, PR_2_m, PR_2_up, PR_2_low, PR_3_m, PR_3_up, PR_3_low, PR_4_m, PR_4_up, PR_4_low

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
@ray.remote
def make_results_cluster( sch, oim, nfp, chan, filt, gamma, size_sep, size_sep_pix, lvl_sep_big, lvl_sep, lvl_sep_max, lvl_sep_bcg, xs, ys, n_levels, mscoim, mscell, mscbcg, R_kpc, R_pix, cat_gal, rc_pix, N_err, per_err, rm_gamma_for_big, kurt_filt, plot_vignet):
    '''
    Runs all classification schemes for a single cluster. Performed by a single ray worker.
    '''

    # Full field ---------------------------------------------------------------
    if sch == 'fullfield':
        synthesis_fullfield( oim, nfp, gamma, lvl_sep_big, xs, ys, n_levels, rm_gamma_for_big = True )
        return None

    # ICL -- WS -----------------------------------------------------------------
    if sch == 'WS':
        output = synthesis_wavsep( nfp, gamma, lvl_sep_big, lvl_sep, xs, ys, n_levels, rm_gamma_for_big, kurt_filt = True, plot_vignet = plot_vignet )
        return None

    # ICL -- WS + SF -----------------------------------------------------------
    if sch == 'WS+SF':
        output = synthesis_wavsep_with_masks( nfp = nfp, gamma = gamma, \
                lvl_sep_big = lvl_sep_big, lvl_sep = lvl_sep, lvl_sep_max = lvl_sep_max, lvl_sep_bcg = lvl_sep_bcg, xs = xs, ys = ys, \
                n_levels = n_levels, mscoim = mscoim, mscell = mscell, mscbcg = mscbcg, R = R_pix, cat_gal = cat_gal, rc_pix = rc_pix,\
                N_err = N_err, per_err = per_err, rm_gamma_for_big = rm_gamma_for_big, kurt_filt = kurt_filt, plot_vignet = plot_vignet )
        F_ICL_m, F_ICL_low, F_ICL_up, F_gal_m, F_gal_low, F_gal_up, f_ICL_m, f_ICL_low, f_ICL_up, PR_1_m, PR_1_up, PR_1_low, PR_2_m, PR_2_up, PR_2_low, PR_3_m, PR_3_up, PR_3_low, PR_4_m, PR_4_up, PR_4_low = output[2:]
        output_df = pd.DataFrame( [[ nf, chan, filt, sch, R_kpc, R_pix, lvl_sep, np.nan, F_ICL_m, F_ICL_low, F_ICL_up, F_gal_m, F_gal_low, F_gal_up, f_ICL_m, f_ICL_low, f_ICL_up, PR_1_m, PR_1_up, PR_1_low, PR_2_m, PR_2_up, PR_2_low, PR_3_m, PR_3_up, PR_3_low, PR_4_m, PR_4_up, PR_4_low ]], \
                        columns = [ 'nf', 'chan', 'filter', 'Atom selection scheme', 'R_kpc', 'R_pix', 'lvl_sep', 'size_sep', 'F_ICL_m', 'F_ICL_low', 'F_ICL_up', 'F_gal_m', 'F_gal_low', 'F_gal_up', 'f_ICL_m', 'f_ICL_low', 'f_ICL_up', 'PR_1_m', 'PR_1_up', 'PR_1_low', 'PR_2_m', 'PR_2_up', 'PR_2_low', 'PR_3_m', 'PR_3_up', 'PR_3_low', 'PR_4_m', 'PR_4_up', 'PR_4_low'  ])

    # ICL+BCG -- WS + SF -------------------------------------------------------
    if sch == 'WS+BCGSF':
        output = synthesis_bcgwavsep_with_masks( nfp = nfp, gamma = gamma, \
                lvl_sep_big = lvl_sep_big, lvl_sep = lvl_sep, lvl_sep_max = lvl_sep_max, lvl_sep_bcg = lvl_sep_bcg, xs = xs, ys = ys, \
                n_levels = n_levels, mscoim = mscoim, mscell = mscell, mscbcg = mscbcg, R = R_pix, cat_gal = cat_gal, rc_pix = rc_pix,\
                N_err = N_err, per_err = per_err, rm_gamma_for_big = rm_gamma_for_big, kurt_filt = kurt_filt, plot_vignet = plot_vignet )

        F_ICL_m, F_ICL_low, F_ICL_up, F_gal_m, F_gal_low, F_gal_up, f_ICL_m, f_ICL_low, f_ICL_up, PR_1_m, PR_1_up, PR_1_low, PR_2_m, PR_2_up, PR_2_low, PR_3_m, PR_3_up, PR_3_low, PR_4_m, PR_4_up, PR_4_low = output[2:]
        output_df = pd.DataFrame( [[ nf, chan, filt, sch, R_kpc, R_pix, lvl_sep, np.nan, F_ICL_m, F_ICL_low, F_ICL_up, F_gal_m, F_gal_low, F_gal_up, f_ICL_m, f_ICL_low, f_ICL_up, PR_1_m, PR_1_up, PR_1_low, PR_2_m, PR_2_up, PR_2_low, PR_3_m, PR_3_up, PR_3_low, PR_4_m, PR_4_up, PR_4_low ]], \
                        columns = [ 'nf', 'chan', 'filter', 'Atom selection scheme', 'R_kpc', 'R_pix', 'lvl_sep', 'size_sep', 'F_ICL_m', 'F_ICL_low', 'F_ICL_up', 'F_gal_m', 'F_gal_low', 'F_gal_up', 'f_ICL_m', 'f_ICL_low', 'f_ICL_up', 'PR_1_m', 'PR_1_up', 'PR_1_low', 'PR_2_m', 'PR_2_up', 'PR_2_low', 'PR_3_m', 'PR_3_up', 'PR_3_low', 'PR_4_m', 'PR_4_up', 'PR_4_low'  ])

    # ICL -- WS + SF + SS ------------------------------------------------------
    if sch == 'WS+SF+SS':
        output = synthesis_wavsizesep_with_masks( nfp = nfp, gamma = gamma, \
                lvl_sep_big = lvl_sep_big, lvl_sep = lvl_sep, lvl_sep_max = lvl_sep_max, lvl_sep_bcg = lvl_sep_bcg, size_sep = size_sep, size_sep_pix =  size_sep_pix, xs = xs, ys = ys, \
                n_levels = n_levels, mscoim = mscoim, mscell = mscell, mscbcg = mscbcg, R = R_pix, cat_gal = cat_gal, rc_pix = rc_pix,\
                N_err = N_err, per_err = per_err, rm_gamma_for_big = rm_gamma_for_big, kurt_filt = kurt_filt, plot_vignet = plot_vignet )
        F_ICL_m, F_ICL_low, F_ICL_up, F_gal_m, F_gal_low, F_gal_up, f_ICL_m, f_ICL_low, f_ICL_up, PR_1_m, PR_1_up, PR_1_low, PR_2_m, PR_2_up, PR_2_low, PR_3_m, PR_3_up, PR_3_low, PR_4_m, PR_4_up, PR_4_low = output[2:]
        output_df = pd.DataFrame( [[ nf, chan, filt, sch, R_kpc, R_pix, lvl_sep, size_sep, F_ICL_m, F_ICL_low, F_ICL_up, F_gal_m, F_gal_low, F_gal_up, f_ICL_m, f_ICL_low, f_ICL_up, PR_1_m, PR_1_up, PR_1_low, PR_2_m, PR_2_up, PR_2_low, PR_3_m, PR_3_up, PR_3_low, PR_4_m, PR_4_up, PR_4_low ]], \
                        columns = [ 'nf', 'chan', 'filter', 'Atom selection scheme', 'R_kpc', 'R_pix', 'lvl_sep', 'size_sep','F_ICL_m', 'F_ICL_low', 'F_ICL_up', 'F_gal_m', 'F_gal_low', 'F_gal_up', 'f_ICL_m', 'f_ICL_low', 'f_ICL_up', 'PR_1_m', 'PR_1_up', 'PR_1_low', 'PR_2_m', 'PR_2_up', 'PR_2_low', 'PR_3_m', 'PR_3_up', 'PR_3_low', 'PR_4_m', 'PR_4_up', 'PR_4_low'  ])

    # ICL+BCG -- WS + SF + SS --------------------------------------------------
    if sch == 'WS+BCGSF+SS':
        output = synthesis_bcgwavsizesep_with_masks( nfp = nfp, chan = chan, gamma = gamma, size_sep = size_sep, size_sep_pix = size_sep_pix,\
                lvl_sep_big = lvl_sep_big, lvl_sep = lvl_sep, lvl_sep_max = lvl_sep_max, lvl_sep_bcg = lvl_sep_bcg, xs = xs, ys = ys, \
                n_levels = n_levels, mscoim = mscoim, mscell = mscell, mscbcg = mscbcg, R = R_pix, cat_gal = cat_gal, rc_pix = rc_pix,\
                N_err = N_err, per_err = per_err, rm_gamma_for_big = rm_gamma_for_big, kurt_filt = kurt_filt, plot_vignet = plot_vignet )
        F_ICL_m, F_ICL_low, F_ICL_up, F_gal_m, F_gal_low, F_gal_up, f_ICL_m, f_ICL_low, f_ICL_up, PR_1_m, PR_1_up, PR_1_low, PR_2_m, PR_2_up, PR_2_low, PR_3_m, PR_3_up, PR_3_low, PR_4_m, PR_4_up, PR_4_low = output[2:]
        output_df = pd.DataFrame( [[ nf, chan, filt, sch, R_kpc, R_pix, lvl_sep, size_sep, F_ICL_m, F_ICL_low, F_ICL_up, F_gal_m, F_gal_low, F_gal_up, f_ICL_m, f_ICL_low, f_ICL_up, PR_1_m, PR_1_up, PR_1_low, PR_2_m, PR_2_up, PR_2_low, PR_3_m, PR_3_up, PR_3_low, PR_4_m, PR_4_up, PR_4_low ]], \
                        columns = [ 'nf', 'chan', 'filter', 'Atom selection scheme', 'R_kpc', 'R_pix', 'lvl_sep', 'size_sep', 'F_ICL_m', 'F_ICL_low', 'F_ICL_up', 'F_gal_m', 'F_gal_low', 'F_gal_up', 'f_ICL_m', 'f_ICL_low', 'f_ICL_up', 'PR_1_m', 'PR_1_up', 'PR_1_low', 'PR_2_m', 'PR_2_up', 'PR_2_low', 'PR_3_m', 'PR_3_up', 'PR_3_low', 'PR_4_m', 'PR_4_up', 'PR_4_low'  ])

    return output_df

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if __name__ == '__main__':

    # Paths, lists & variables
    path_data = '/n03data/ellien/JWST/data/'
    path_scripts = '/n03data/ellien/JWST/JWST_scripts'
    path_wavelets = '/n03data/ellien/JWST/wavelets/out14/'
    path_plots = '/n03data/ellien/JWST/plots'

    nfl = [ {'nf':'jw02736001001_f090w_bkg_rot_crop_test_clean_det_nosky_input.fits', 'chan':'long', 'pix_scale':0.063, 'n_levels':10, 'lvl_sep_max':999 } ]

            # out13
            #{'nf':'jw02736001001_f356w_bkg_rot_crop_input.fits', 'chan':'long', 'pix_scale':0.063, 'n_levels':10, 'lvl_sep_max':999 }, \
            #{'nf':'jw02736001001_f444w_bkg_rot_crop_input.fits', 'chan':'long', 'pix_scale':0.063, 'n_levels':10, 'lvl_sep_max':999 }, \
            #{'nf':'jw02736001001_f277w_bkg_rot_crop_input.fits', 'chan':'long', 'pix_scale':0.063, 'n_levels':10, 'lvl_sep_max':999 }, \
            #{'nf':'jw02736001001_f090w_bkg_rot_crop_warp_det_nosky_input.fits', 'chan':'long', 'pix_scale':0.063, 'n_levels':10, 'lvl_sep_max':999 }, \
            #{'nf':'jw02736001001_f150w_bkg_rot_crop_warp_det_nosky_input.fits', 'chan':'long', 'pix_scale':0.063, 'n_levels':10, 'lvl_sep_max':999 }, \
            #{'nf':'jw02736001001_f200w_bkg_rot_crop_warp_det_nosky_input.fits', 'chan':'long', 'pix_scale':0.063, 'n_levels':10, 'lvl_sep_max':999 } ]

            # out12
            #{'nf':'jw02736001001_f090w_bkg_rot_crop_input.fits', 'chan':'short', 'pix_scale':0.031, 'n_levels':11, 'lvl_sep_max':8 }, \
            #{'nf':'jw02736001001_f150w_bkg_rot_crop_input.fits', 'chan':'short', 'pix_scale':0.031, 'n_levels':11, 'lvl_sep_max':8 }, \
            #{'nf':'jw02736001001_f200w_bkg_rot_crop_input.fits', 'chan':'short', 'pix_scale':0.031, 'n_levels':11, 'lvl_sep_max':8 }

    lvl_sepl = [ 3, 4, 5, 6, 7 ] # wavelet scale separation
    size_sepl = [ 60, 80, 100, 140, 200 ] # size separation [kpc]
    R_kpcl = [ 128, 200, 400 ] # radius in which quantities are measured [kpc]
    physcale = 5.3 # kpc/"
    gamma = 0.5
    lvl_sep_big = 5
    lvl_sep_bcg = 6
    rm_gamma_for_big = True

    rc = 10 # kpc, distance to center to be classified as gal
    N_err = 50
    per_err = 0.1

    results = []
    ray_refs = []
    ray_outputs = []

    # ray hyperparameters
    n_cpus = 12
    ray.init(num_cpus = n_cpus)

    for chan in [ 'long' ]:

        for R_kpc in R_kpcl:

            # Read galaxy catalog
            rgal = pyr.open(os.path.join(path_data, 'mahler_noirot_merged_member_gal_ra_dec_pix_%s.reg'%chan))
            cat_gal = []
            for gal in rgal:
                cat_gal.append(gal.coord_list)
            cat_gal = np.array(cat_gal)

            # Read star region files
            rco = pyr.open(os.path.join(path_data, 'star_flags_polygon_pix_%s.reg'%chan))
            rell = pyr.open(os.path.join(path_data, 'icl_flags_ellipse_pix_%s.reg'%chan))
            rbcg = pyr.open(os.path.join(path_data, 'bcg_flags_ellipse_pix_%s.reg'%chan))

            for nfd in nfl:

                if nfd['chan'] == chan:

                    # Read image data from dic
                    nf = nfd['nf']
                    filt = nf.split('_')[1]
                    n_levels = nfd['n_levels']
                    pix_scale = nfd['pix_scale']
                    lvl_sep_max = nfd['lvl_sep_max']
                    rc_pix = rc / physcale / pix_scale # pixels
                    R_pix = R_kpc / physcale / pix_scale # pixels
                    id_R_pix = ray.put(R_pix)
                    print(nf)

                    # Read image file
                    nfp = os.path.join( path_wavelets, nf[:-4] )
                    oim_file = os.path.join( path_data, nf )
                    hdu = fits.open(oim_file)
                    oim = hdu[0].data
                    id_oim = ray.put(oim)
                    xs, ys = oim.shape

                    # mask image
                    mscell = rell.get_mask(hdu = hdu[0]) # not python convention
                    mscoim = rco.get_mask(hdu = hdu[0]) # not python convention
                    mscbcg = rbcg.get_mask(hdu = hdu[0]) # not python convention

                    # Full field ------------------------------------------------
                    lvl_sep = np.nan
                    size_sep = np.nan
                    size_sep_pix = np.nan

                    ray_refs.append( make_results_cluster.remote(sch = 'fullfield', \
                                                 oim = id_oim, \
                                                 nfp = nfp, \
                                                 chan = chan, \
                                                 filt = filt, \
                                                 gamma = gamma, \
                                                 lvl_sep_big = lvl_sep_big, \
                                                 lvl_sep = lvl_sep, \
                                                 lvl_sep_max = lvl_sep_max, \
                                                 lvl_sep_bcg = lvl_sep_bcg, \
                                                 size_sep = size_sep, \
                                                 size_sep_pix = size_sep_pix, \
                                                 xs = xs, \
                                                 ys = ys, \
                                                 n_levels = n_levels, \
                                                 mscoim = mscoim, \
                                                 mscell = mscell, \
                                                 mscbcg = mscbcg, \
                                                 R_pix = id_R_pix, \
                                                 R_kpc = R_kpc,\
                                                 cat_gal = cat_gal, \
                                                 rc_pix = rc_pix,\
                                                 N_err = N_err, \
                                                 per_err = per_err, \
                                                 rm_gamma_for_big = rm_gamma_for_big, \
                                                 kurt_filt = True, \
                                                 plot_vignet = False ))


                    # ICL -- WS ------------------------------------------------
                    for lvl_sep in lvl_sepl:
                        ray_refs.append( make_results_cluster.remote(sch = 'WS', \
                                                        oim = id_oim, \
                                                        nfp = nfp, \
                                                        chan = chan, \
                                                        filt = filt, \
                                                        gamma = gamma, \
                                                        lvl_sep_big = lvl_sep_big, \
                                                        lvl_sep = lvl_sep, \
                                                        lvl_sep_max = lvl_sep_max, \
                                                        lvl_sep_bcg = lvl_sep_bcg, \
                                                        size_sep = size_sep, \
                                                        size_sep_pix = size_sep_pix, \
                                                        xs = xs, \
                                                        ys = ys, \
                                                        n_levels = n_levels, \
                                                        mscoim = mscoim, \
                                                        mscell = mscell, \
                                                        mscbcg = mscbcg, \
                                                        R_pix = id_R_pix, \
                                                        R_kpc = R_kpc,\
                                                        cat_gal = cat_gal, \
                                                        rc_pix = rc_pix,\
                                                        N_err = N_err, \
                                                        per_err = per_err, \
                                                        rm_gamma_for_big = rm_gamma_for_big, \
                                                        kurt_filt = True, \
                                                        plot_vignet = False ))

                    # ICL -- WS + SF -------------------------------------------
                    for lvl_sep in lvl_sepl:
                        ray_refs.append( make_results_cluster.remote(sch = 'WS+SF', \
                                                        oim = id_oim, \
                                                        nfp = nfp, \
                                                        chan = chan, \
                                                        filt = filt, \
                                                        gamma = gamma, \
                                                        lvl_sep_big = lvl_sep_big, \
                                                        lvl_sep = lvl_sep, \
                                                        lvl_sep_max = lvl_sep_max, \
                                                        lvl_sep_bcg = lvl_sep_bcg, \
                                                        size_sep = size_sep, \
                                                        size_sep_pix = size_sep_pix, \
                                                        xs = xs, \
                                                        ys = ys, \
                                                        n_levels = n_levels, \
                                                        mscoim = mscoim, \
                                                        mscell = mscell, \
                                                        mscbcg = mscbcg, \
                                                        R_kpc = R_kpc,\
                                                        R_pix = id_R_pix, \
                                                        cat_gal = cat_gal, \
                                                        rc_pix = rc_pix,\
                                                        N_err = N_err, \
                                                        per_err = per_err, \
                                                        rm_gamma_for_big = rm_gamma_for_big, \
                                                        kurt_filt = True, \
                                                        plot_vignet = False ))

                    # ICL+BCG -- WS + SF ---------------------------------------
                    for lvl_sep in lvl_sepl:
                        ray_refs.append( make_results_cluster.remote(sch = 'WS+BCGSF', \
                                                        oim = oim, \
                                                        nfp = nfp, \
                                                        chan = chan, \
                                                        filt = filt, \
                                                        gamma = gamma, \
                                                        lvl_sep_big = lvl_sep_big, \
                                                        lvl_sep = lvl_sep, \
                                                        lvl_sep_max = lvl_sep_max, \
                                                        lvl_sep_bcg = lvl_sep_bcg, \
                                                        size_sep = size_sep, \
                                                        size_sep_pix = size_sep_pix, \
                                                        xs = xs, \
                                                        ys = ys, \
                                                        n_levels = n_levels, \
                                                        mscoim = mscoim, \
                                                        mscell = mscell, \
                                                        mscbcg = mscbcg, \
                                                        R_kpc = R_kpc,\
                                                        R_pix = id_R_pix, \
                                                        cat_gal = cat_gal, \
                                                        rc_pix = rc_pix,\
                                                        N_err = N_err, \
                                                        per_err = per_err, \
                                                        rm_gamma_for_big = rm_gamma_for_big, \
                                                        kurt_filt = True, \
                                                        plot_vignet = False ))

                    # ICL -- WS + SF + SS --------------------------------------
                    for lvl_sep in lvl_sepl:
                        for size_sep in size_sepl:
                            size_sep_pix = size_sep / physcale / pix_scale # pixels
                            ray_refs.append( make_results_cluster.remote(sch = 'WS+SF+SS', \
                                                            oim = oim, \
                                                            nfp = nfp, \
                                                            chan = chan, \
                                                            filt = filt, \
                                                            gamma = gamma, \
                                                            lvl_sep_big = lvl_sep_big, \
                                                            lvl_sep = lvl_sep, \
                                                            lvl_sep_max = lvl_sep_max, \
                                                            lvl_sep_bcg = lvl_sep_bcg, \
                                                            size_sep = size_sep, \
                                                            size_sep_pix = size_sep_pix, \
                                                            xs = xs, \
                                                            ys = ys, \
                                                            n_levels = n_levels, \
                                                            mscoim = mscoim, \
                                                            mscell = mscell, \
                                                            mscbcg = mscbcg, \
                                                            R_kpc = R_kpc,\
                                                            R_pix = id_R_pix, \
                                                            cat_gal = cat_gal, \
                                                            rc_pix = rc_pix,\
                                                            N_err = N_err, \
                                                            per_err = per_err, \
                                                            rm_gamma_for_big = rm_gamma_for_big, \
                                                            kurt_filt = True, \
                                                            plot_vignet = False ))

                    # ICL+BCG -- WS + SF + SS ----------------------------------
                    for lvl_sep in lvl_sepl:

                        for size_sep in size_sepl:

                            size_sep_pix = size_sep / physcale / pix_scale # pixels
                            ray_refs.append( make_results_cluster.remote(sch = 'WS+BCGSF+SS', \
                                                            oim = oim, \
                                                            nfp = nfp, \
                                                            chan = chan, \
                                                            filt = filt, \
                                                            gamma = gamma, \
                                                            lvl_sep_big = lvl_sep_big, \
                                                            lvl_sep = lvl_sep, \
                                                            lvl_sep_max = lvl_sep_max, \
                                                            lvl_sep_bcg = lvl_sep_bcg, \
                                                            size_sep = size_sep, \
                                                            size_sep_pix = size_sep_pix, \
                                                            xs = xs, \
                                                            ys = ys, \
                                                            n_levels = n_levels, \
                                                            mscoim = mscoim, \
                                                            mscell = mscell, \
                                                            mscbcg = mscbcg, \
                                                            R_kpc = R_kpc, \
                                                            R_pix = id_R_pix, \
                                                            cat_gal = cat_gal, \
                                                            rc_pix = rc_pix,\
                                                            N_err = N_err, \
                                                            per_err = per_err, \
                                                            rm_gamma_for_big = rm_gamma_for_big, \
                                                            kurt_filt = True, \
                                                            plot_vignet = False ))

    for ref in ray_refs:
        ray_outputs.append(ray.get(ref))

    ray.shutdown()

    results_df = ray_outputs[0]
    for output_df in ray_outputs[1:]:
        results_df = pd.concat( [ results_df, output_df], ignore_index = True )

    results_df.to_excel('/home/ellien/JWST/analysis/results_out13.xlsx')
