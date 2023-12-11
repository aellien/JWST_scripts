import numpy as np
from astropy.io import fits
import os
import glob as glob

def MJy_to_mu_montes(im_MJy, mu_lim):
    # MJy to surface brightness AB (source Montes, M. 2022)
    im_MJy[ np.where(im_MJy == 0.) ] = 1E-10
    im_mu = - 2.5 * np.log10( im_MJy / 4.25 * 1E-4 ) + 8.906
    im_mu[im_mu > mu_lim] = mu_lim
    return im_mu

def MJy_to_mu(im_MJy, pixar_sr, mu_lim):
    im_MJy[ np.where(im_MJy == 0.) ] = 1E-10
    ZP_AB = -6.10 - 2.5 * np.log10(pixar_sr)
    im_mu = -2.5 * np.log10( im_MJy) + ZP_AB
    im_mu[im_mu > mu_lim] = mu_lim
    return im_mu

if __name__ == '__main__':

    # Paths, lists & variables
    path_data = '/home/ellien/JWST/data/'
    path_scripts = '/home/ellien/JWST/JWST_scripts'
    path_wavelets = '/home/ellien/JWST/wavelets/out12/'
    path_plots = '/home/ellien/JWST/plots'

    nfl = glob.glob( os.path.join( path_wavelets, '*f090w*synth*[!ut].fits'))
    for nf in nfl:

        # Read MJy image
        print(nf)
        hdu = fits.open(nf)
        head = hdu[0].header

        # MJy to surface brightness AB (source Montes, M. 2022)
        im_MJy = hdu[0].data
        im_MJy[ np.where(im_MJy == 0.) ] = 1E-10
        im_mu = - 2.5 * np.log10( im_MJy / 4.25 * 1E-4 ) + 8.906
        im_mu[im_mu > 31.1] = 31.1 # SB limit from Montes 2022

        # Raw surface brightness map
        hduo = fits.PrimaryHDU(im_mu, header = head)
        hduo.writeto( os.path.join( path_wavelets, nf[:-5] + '_mu.fits' ), overwrite = True )

        # Cutoff surface brightness map
        # im_mu_cut = np.copy(im_mu)
        # im_mu_cut[np.where(im_mu_cut > 28.)] = 1000
        # hduo = fits.PrimaryHDU(im_mu_cut, header = head)
        # hduo.writeto( os.path.join( path_wavelets, nf[:-5] + '_mu_cut.fits' ), overwrite = True )
