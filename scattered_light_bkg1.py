# conda activate photutils
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clip
from astropy.stats import SigmaClip
from astropy.visualization import *
from scipy.ndimage import gaussian_filter
from photutils.background import Background2D, MedianBackground, ModeEstimatorBackground
import os
import tqdm
import matplotlib.pyplot as plt
import pyregion as pyr

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if __name__ == '__main__':

    plt.ion()

    # Paths, lists & variables
    path_data = '/home/ellien/JWST/data/'
    path_scripts = '/home/ellien/JWST/JWST_scripts'
    path_wavelets = '/home/ellien/JWST/wavelets/out12/'
    path_plots = '/home/ellien/JWST/plots'

    sig = 3
    xc = 2 * 1085 # estimated by hand
    yc = 2 * 1027 # estimated by hand

    nfdl = [ {'filt':'f090w', 'nf':'jw02736001001_f090w_bkg_rot_crop_input.fits', 'nwf':'jw02736001001_f090w_bkg_rot_crop_input.synth.gal.wavsep_007.fits', 'chan':'short', 'pix_scale':0.031, 'bkg_scaling':1, 'bkg_meshsize':200 }, \
             {'filt':'f150w', 'nf':'jw02736001001_f150w_bkg_rot_crop_input.fits', 'nwf':'jw02736001001_f150w_bkg_rot_crop_input.synth.gal.wavsep_007.fits', 'chan':'short', 'pix_scale':0.031, 'bkg_scaling':1, 'bkg_meshsize':200}, \
             {'filt':'f200w', 'nf':'jw02736001001_f200w_bkg_rot_crop_input.fits', 'nwf':'jw02736001001_f200w_bkg_rot_crop_input.synth.gal.wavsep_007.fits', 'chan':'short', 'pix_scale':0.031, 'bkg_scaling':1, 'bkg_meshsize':200 } ]

            #{'filt':'f090w', 'nf':'jw02736001001_f090w_bkg_rot_crop_warp.fits', 'nwf':'jw02736001001_f090w_bkg_rot_crop_warp_nobkg1_det_nosky_input.synth.gal.wavsep_007.fits', 'chan':'short', 'pix_scale':0.063, 'bkg_scaling':0.75, 'bkg_meshsize':50 }, \
            #{'filt':'f150w', 'nf':'jw02736001001_f150w_bkg_rot_crop_warp.fits', 'nwf':'jw02736001001_f150w_bkg_rot_crop_warp_nobkg1_det_nosky_input.synth.gal.wavsep_008.fits', 'chan':'short', 'pix_scale':0.063, 'bkg_scaling':0.75, 'bkg_meshsize':50}, \
            #{'filt':'f200w', 'nf':'jw02736001001_f200w_bkg_rot_crop_warp.fits', 'nwf':'jw02736001001_f200w_bkg_rot_crop_warp_nobkg1_det_nosky_input.synth.gal.wavsep_008.fits', 'chan':'short', 'pix_scale':0.063, 'bkg_scaling':0.75, 'bkg_meshsize':50 } ]

    for nfd in nfdl:

        print(nfd['filt'])

        # Read files
        nfp = os.path.join( path_wavelets, nfd['nwf'])
        hdu = fits.open(nfp)
        recim = hdu[0].data
        xs, ys = recim.shape

        nfp = os.path.join( path_data, nfd['nf'])
        hdu = fits.open(nfp)
        im = hdu[1].data

        # scattered ligt + ICL
        bkg_icl = im - recim
        res = np.copy(bkg_icl)

        # First median bkg
        mask1 = np.zeros(res.shape)
        noise = sigma_clip(res, sig, sig, False).data
        det_thresh = np.nanmedian(noise) + np.nanmedian(np.absolute(noise - np.nanmedian(noise)))
        print(np.nanmedian(noise), np.nanmedian(np.absolute(noise - np.nanmedian(noise))), det_thresh)
        mask1[res >= det_thresh] = 1.

        # Median interpolated background
        bkg0 = Background2D(res, (25, 25), filter_size = (3, 3), sigma_clip = SigmaClip(sig), mask = mask1.astype(bool)).background
        res -= bkg0

        # Scattered light map
        bkg1 = np.zeros(res.shape)
        for i in tqdm.tqdm(range(ys)):

            # median measured on half line/col
            pixl = sigma_clip(res[:xc,i], sig, sig, False)
            bkg_med = np.nanmedian(pixl.data)
            bkg1[:xc,i] += bkg_med

            pixl = sigma_clip(res[xc:,i], sig, sig, False)
            bkg_med = np.nanmedian(pixl.data)
            bkg1[xc:,i] += bkg_med

        res -= bkg1
        for i in tqdm.tqdm(range(xs)):

            # median measured on half line/col
            pixl = sigma_clip(res[i,:yc], sig, sig, False)
            bkg_med = np.nanmedian(pixl.data)
            bkg1[i,:yc] += bkg_med

            pixl = sigma_clip(res[i,yc:], sig, sig, False)
            bkg_med = np.nanmedian(pixl.data)
            bkg1[i,yc:] += bkg_med


        #%
        # bkg = gaussian_filter(bkg, 3)
        #%

        # Remove scattered light from image
        clean = im - bkg1 + bkg0
        icl = bkg_icl - bkg1 + bkg0

        # Mask reminder of ICL
        mask2 = np.zeros(icl.shape)
        r = pyr.open('/home/ellien/JWST/data/icl_flags_ellipse_pix_short.reg')
        maskreg = r.get_mask(hdu = hdu[1])
        noise = sigma_clip(icl, sig, sig, False).data
        det_thresh = np.nanmedian(noise) + np.nanmedian(np.absolute(noise - np.nanmedian(noise)))
        mask2[icl >= det_thresh] = 1.
        mask2 *= maskreg

        # Median interpolated background
        bkg_est = ModeEstimatorBackground(median_factor = 2.5, mean_factor = 1.5, sigma_clip = SigmaClip(sigma = sig))
        bkg2 = Background2D(icl, (nfd['bkg_meshsize'], nfd['bkg_meshsize']), filter_size = (3, 3), sigma_clip = SigmaClip(sig), bkg_estimator = bkg_est, mask = mask2.astype(bool)).background
        combbkg = (bkg1 + bkg2) * nfd['bkg_scaling']
        nobkg2 = im - combbkg

        fig, ax = plt.subplots(3, 3)
        norm = ImageNormalize( bkg_icl, interval = ZScaleInterval(), stretch = LinearStretch())
        cmap = 'gray'
        ax[0][0].imshow(im, origin = 'lower', cmap = cmap, norm = norm )
        ax[0][0].set_title('im')
        ax[0][1].imshow(bkg_icl, origin = 'lower', cmap = cmap, norm = norm )
        ax[0][1].set_title('bkg_icl')
        ax[0][2].imshow(mask1, origin = 'lower', cmap = cmap )
        ax[0][2].set_title('mask1')
        ax[1][0].imshow(bkg0, origin = 'lower', cmap = cmap, norm = norm )
        ax[1][0].set_title('bkg0')
        ax[1][1].imshow(bkg1, origin = 'lower', cmap = cmap, norm = norm )
        ax[1][1].set_title('bkg1')
        ax[1][2].imshow(icl, origin = 'lower', cmap = cmap, norm = norm )
        ax[1][2].set_title('icl')
        ax[2][0].imshow(mask2, origin = 'lower', cmap = cmap )
        ax[2][0].set_title('mask2')
        ax[2][1].imshow(bkg2, origin = 'lower', cmap = cmap, norm = norm )
        ax[2][1].set_title('bkg2')
        ax[2][2].imshow(combbkg, origin = 'lower', cmap = cmap, norm = norm )
        ax[2][2].set_title('combbkg')

        #plt.show(block = True)

        # Write results to files
        hduo = fits.PrimaryHDU(clean, header = hdu[0].header)
        hduo.writeto(os.path.join(path_data, nfd['nf'][:-5] + '_nobkg1.fits'), overwrite = True)

        hduo = fits.PrimaryHDU(bkg1, header = hdu[0].header)
        hduo.writeto(os.path.join(path_data, nfd['nf'][:-5] + '_bkg1.fits'), overwrite = True)

        hduo = fits.PrimaryHDU(bkg2, header = hdu[0].header)
        hduo.writeto(os.path.join(path_data, nfd['nf'][:-5] + '_bkg2.fits'), overwrite = True)

        hduo = fits.PrimaryHDU(bkg_icl, header = hdu[0].header)
        hduo.writeto(os.path.join(path_data, nfd['nf'][:-5] + '_res1.fits'), overwrite = True)

        hduo = fits.PrimaryHDU(icl, header = hdu[0].header)
        hduo.writeto(os.path.join(path_data, nfd['nf'][:-5] + '_res2.fits'), overwrite = True)

        hduo = fits.PrimaryHDU(combbkg, header = hdu[0].header)
        hduo.writeto(os.path.join(path_data, nfd['nf'][:-5] + '_combbkg.fits'), overwrite = True)

        hduo = fits.PrimaryHDU(nobkg2, header = hdu[0].header)
        hduo.writeto(os.path.join(path_data, nfd['nf'][:-5] + '_nobkg2.fits'), overwrite = True)
