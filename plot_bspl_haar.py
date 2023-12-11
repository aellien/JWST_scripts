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
from astropy.visualization import AsymmetricPercentileInterval
from astropy.visualization import ImageNormalize
from astropy import units as u
sys.path.append('/home/ellien/LSST_ICL/scripts')
from atom_props import *

# MPL PARAMS
mpl.rcParams['xtick.major.size'] = 0
mpl.rcParams['ytick.major.size'] = 0
mpl.rcParams['xtick.minor.size'] = 0
mpl.rcParams['ytick.minor.size'] = 0
mpl.rcParams['xtick.labelbottom'] = False
mpl.rcParams['ytick.labelleft'] = False

if __name__ == '__main__':

    # Paths, lists & variables
    path_data = '/home/ellien/JWST/data/'
    path_scripts = '/home/ellien/JWST/JWST_scripts'
    path_wavelets = '/home/ellien/JWST/wavelets/out11/'
    path_plots = '/home/ellien/JWST/plots'

    fltl = [ 'f277w', 'f356w', 'f444w' ]

    gamma = 0.2

    for flt in fltl:

        # read image
        hdu = fits.open( os.path.join(path_data, 'jw02736001001_%s_bkg_rot_crop_det_nosky.fits'%flt) )
        im = hdu[0].data

        # Read atoms
        nfp = os.path.join( path_wavelets, 'jw02736001001_%s_bkg_rot_crop_det_nosky'%flt )
        ol, itl = read_image_atoms( nfp, verbose = True )

        onb = len(ol)
        onb_bspl = 0
        onb_haar = 0

        bspl = np.zeros(im.shape)
        haar = np.zeros(im.shape)

        # separate atoms between haar and bspl
        for o, it in zip(ol, itl):

            x_min, y_min, x_max, y_max = o.bbox
            lvlo = o.level

            if o.filter == 'BSPL':
                bspl[ x_min : x_max, y_min : y_max ] += o.image * gamma
                onb_bspl += 1

            elif o.filter == 'HAAR':
                haar[ x_min : x_max, y_min : y_max ] += o.image * gamma
                onb_haar += 1

        print('Haar atoms : %d/%d'%(onb_haar, onb))

        # plots
        interval = AsymmetricPercentileInterval(0, 99.5) # meilleur rendu que MinMax or ZScale pour images reconstruites
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow( bspl, cmap = 'binary_r', origin = 'lower',
                    norm = ImageNormalize( bspl, \
                    interval = interval, \
                    stretch = LinearStretch()))
        ax[1].imshow( haar, cmap = 'binary_r', origin = 'lower',
                    norm = ImageNormalize( haar, \
                    interval = interval, \
                    stretch = LinearStretch()))
        plt.title('%s'%flt)
        plt.show()
