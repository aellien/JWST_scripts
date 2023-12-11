import dawis as d
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
import pyregion as pyr
from glob import glob
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import LinearStretch, LogStretch
from astropy.visualization import ZScaleInterval, MinMaxInterval
from astropy.visualization import AsymmetricPercentileInterval
from astropy.visualization import ImageNormalize
from astropy import units as u
from skimage.morphology import (erosion, dilation, closing, opening,
                                area_closing, area_opening)
sys.path.append('/home/ellien/LSST_ICL/scripts')
from atom_props import *
# MPL PARAMS
#mpl.rcParams['xtick.major.size'] = 0
#mpl.rcParams['ytick.major.size'] = 0
#mpl.rcParams['xtick.minor.size'] = 0
#mpl.rcParams['ytick.minor.size'] = 0
#mpl.rcParams['xtick.labelbottom'] = False
#mpl.rcParams['ytick.labelleft'] = False

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def plot_atom_props(ol, itl, indexes):

    eccl = []
    wl = []
    s = []
    extl = []
    for i in l_at:
        x_min, y_min, x_max, y_max = ol[i].bbox
        lvlo = ol[i].level
        itm = itl[i].interscale_maximum
        xco = itm.x_max
        yco = itm.y_max
        ecc = itm.eccentricity
        extent = itm.extent

        eccl.append(ecc)
        wl.append(lvlo)
        s.append( (x_max - x_min) / 2. + ( y_max - y_min ) / 2.  )
        extl.append(extent)

    fig, ax = plt.subplots(1, 3)
    ax[0].plot(eccl, extl, 'o')
    ax[0].set_title('eccentricity vs extent')

    ax[1].plot(eccl, s, 'o')
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[1].set_title('eccentricity vs size')

    ax[2].plot(extl, s, 'o')
    ax[2].set_title('extent vs size')

    plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def plot_wavelet_scales(ol, itl, indexes, n_levels, xs, ys, gamma):

    wdc = np.zeros((xs, ys, n_levels))

    for i in indexes:
        x_min, y_min, x_max, y_max = ol[i].bbox
        lvlo = ol[i].level

        wdc[ x_min : x_max, y_min : y_max, lvlo ] += ol[i].image * gamma

    fig, ax = plt.subplots(3, 3)
    interval = AsymmetricPercentileInterval(0, 99.5)
    lvl = 0
    for i in range(3):
        for j in range(3):
            ax[i][j].imshow(wdc[:,:,lvl], cmap = 'binary_r', origin = 'lower',\
                            norm = ImageNormalize( wdc[:,:,lvl], \
                            interval = interval, \
                            stretch = LinearStretch()))
            for k in indexes:
                if ol[k].level == lvl:
                    itm = itl[k].interscale_maximum
                    xco = itm.x_max
                    yco = itm.y_max
                    ax[i][j].plot(yco, xco, 'b+')
            lvl += 1

    plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def plot_test():

    # Paths, lists & variables
    path_data = '/home/ellien/JWST/data/'
    path_scripts = '/home/ellien/JWST/JWST_scripts'
    path_wavelets = '/home/ellien/JWST/wavelets/out2/'
    path_plots = '/home/ellien/JWST/plots'

    nfl = [ 'f277_rot.fits', 'f356_rot.fits', 'f444_rot.fits' ]

    # Read original image
    hdu = fits.open( os.path.join(path_data, 'f277_rot.fits') )
    header = hdu[0].header
    oim = hdu[0].data
    wcs = WCS(header)

    # Read star region files
    r = pyr.open(os.path.join(path_data, 'star_flags_cross.reg'))
    #r = pyr.parse(r).as_imagecoord(header)
    patch_list, artist_list = r.get_mpl_patches_texts()

    # plots
    fig, ax = plt.subplots(1,1)
    ax.imshow( oim, cmap = 'binary_r', origin = 'lower',
                    norm = ImageNormalize( oim, \
                    interval = ZScaleInterval(), \
                    stretch = LinearStretch()) )

    for p in patch_list:
        ax.add_patch(p)

    # filter
    mask = r.get_mask(hdu = hdu[0])
    plt.figure()
    plt.imshow(mask)
    plt.show()

    return None

if __name__ == '__main__':

    # Paths, lists & variables
    path_data = '/home/ellien/JWST/data/'
    path_scripts = '/home/ellien/JWST/JWST_scripts'
    path_wavelets = '/home/ellien/JWST/wavelets/out11/'
    path_plots = '/home/ellien/JWST/plots'

    fltl = [ 'f277w', 'f356w', 'f444w' ]

    gamma = 0.5
    lvl_sep_big = 5

    for flt in fltl:

        # Read original image
        hdu = fits.open( os.path.join(path_data, 'jw02736001001_%s_bkg_rot_crop_det_nosky.fits'%flt) )
        header = hdu[0].header
        oim = hdu[0].data
        xs, ys = oim.shape
        xc = xs / 2.
        yc = ys / 2.

        # Read star region files
        rcr = pyr.open(os.path.join(path_data, 'star_flags_cross.reg'))
        rco = pyr.open(os.path.join(path_data, 'star_flags_core.reg'))
        r = pyr.open(os.path.join(path_data, 'star_flags_polygon.reg'))
        patch_list, artist_list = r.get_mpl_patches_texts()

        # mask image
        mscrim = rcr.get_mask(hdu = hdu[0]) # not python convention
        mscoim = rco.get_mask(hdu = hdu[0]) # not python convention
        ms = r.get_mask(hdu = hdu[0])

        # star image initialisation
        sim = np.zeros(oim.shape)
        rim = np.zeros(oim.shape)

        # Read atoms
        nfp = os.path.join( path_wavelets,  'jw02736001001_%s_bkg_rot_crop_det_nosky'%flt )
        ol, itl = read_image_atoms( nfp, verbose = True )

        # Select atom in masks
        l_at = []
        for j, o in enumerate(ol):

            x_min, y_min, x_max, y_max = o.bbox
            lvlo = o.level
            itm = itl[j].interscale_maximum
            xco = itm.x_max
            yco = itm.y_max
            ecc = itm.eccentricity
            extent = itm.extent
            size = (x_max - x_min) / 2. + ( y_max - y_min ) / 2.

            if lvlo >= lvl_sep_big:
                rim[ x_min : x_max, y_min : y_max ] += o.image
            else:
                rim[ x_min : x_max, y_min : y_max ] += o.image * gamma

            if ms[xco, yco] == 1:
                if lvlo >= lvl_sep_big:
                    sim[ x_min : x_max, y_min : y_max ] += o.image
                else:
                    sim[ x_min : x_max, y_min : y_max ] += o.image * gamma
                l_at.append(j)

            #if mscrim[xco, yco] == 1:
            #    if lvlo >= lvl_sep_big:
            #        sim[ x_min : x_max, y_min : y_max ] += o.image
            #    else:
            #        sim[ x_min : x_max, y_min : y_max ] += o.image * gamma
            #    l_at.append(j)

            #elif mscoim[xco, yco] == 1:
            #    if lvlo >= lvl_sep_big:
            #        sim[ x_min : x_max, y_min : y_max ] += o.image
            #    else:
            #        sim[ x_min : x_max, y_min : y_max ] += o.image * gamma
            #    l_at.append(j)

        #sim[np.where(sim < 1E-2)] = 0.

        # to surface brightness
        sim[ np.where(sim == 0.) ] = 1E-10
        sim_mu = - 2.5 * np.log10( sim / 4.25 * 1E-4 ) + 8.906

        # morphological features
        #sim = area_opening(area_closing(sim, 1000), 1000)

        # plot
        fig, ax = plt.subplots(2, 2)
        #interval = AsymmetricPercentileInterval(0, 99.5)
        ax[0][0].imshow( sim, cmap = 'binary_r', origin = 'lower',
                        norm = ImageNormalize( oim, \
                        interval = ZScaleInterval(), \
                        stretch = LinearStretch()))
        ax[0][0].set_title('Synthesis star')

        ax[0][1].imshow( oim - sim, cmap = 'binary_r', origin = 'lower',
                        norm = ImageNormalize( oim, \
                        interval = ZScaleInterval(), \
                        stretch = LinearStretch()))
        ax[0][1].set_title('Star-subtracted image')

        ax[1][0].imshow( rim - sim, cmap = 'binary_r', origin = 'lower',
                        norm = ImageNormalize( oim, \
                        interval = ZScaleInterval(), \
                        stretch = LinearStretch()))
        ax[1][0].set_title('Modeled star-subtracted image')

        ax[1][1].imshow( oim - rim, cmap = 'binary_r', origin = 'lower',
                        norm = ImageNormalize( oim - rim, \
                        interval = ZScaleInterval(), \
                        stretch = LinearStretch()))
        ax[1][1].set_title('Residuals')

        plt.title('%s' %flt)
        plt.tight_layout()
        plt.show()

        # Write files
        hduo = fits.PrimaryHDU(sim, header = header)
        hduo.writeto( '/home/ellien/JWST/gallery/star_image_%s_MJy.fits'%flt, overwrite = True )

        hduo = fits.PrimaryHDU(sim_mu, header = header)
        hduo.writeto( '/home/ellien/JWST/gallery/star_image_%s_mu.fits'%flt, overwrite = True )

        hduo = fits.PrimaryHDU(oim - sim, header = header)
        hduo.writeto( '/home/ellien/JWST/gallery/star_subtracted_image_%s_MJy.fits'%flt, overwrite = True )

        hduo = fits.PrimaryHDU(rim - sim, header = header)
        hduo.writeto( '/home/ellien/JWST/gallery/star_subtracted_synth_image_%s_MJy.fits'%flt, overwrite = True )
