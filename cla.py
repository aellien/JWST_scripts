import sys
import glob as glob
import os
import numpy as np
import matplotlib.pyplot as plt
import tqdm as tqdm
from astropy.io import fits
from astropy.visualization import LinearStretch, LogStretch
from astropy.visualization import ZScaleInterval, MinMaxInterval, AsymmetricPercentileInterval
from astropy.visualization import ImageNormalize
from plot_pub import rebin
from scipy.signal import correlate2d
from mpl_toolkits.axes_grid1 import ImageGrid

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def normalize_image(im):

    out = np.copy(im)
    out = np.around((100 * (out - np.min(out) ) / ( np.max(out) - np.min(out))), decimals = 2)
    return out


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if __name__ == '__main__':

    # Paths, lists & variables
    path_data = '/home/ellien/JWST/power_ratio/'
    path_scripts = '/home/ellien/JWST/JWST_scripts'
    path_wavelets = '/home/ellien/JWST/wavelets/out12/'
    path_plots = '/home/ellien/JWST/plots'

    nf1 = 'jw02736001001_f277w_bkg_rot_crop_input.synth.icl.wavsizesepmask_005_080.fits'
    nf2 = 'jw02736001001_f356w_bkg_rot_crop_input.synth.icl.wavsizesepmask_005_080.fits'
    nf3 = 'jw02736001001_f444w_bkg_rot_crop_input.synth.icl.wavsizesepmask_005_080.fits'

    nfp1 = os.path.join(path_wavelets, nf1)
    nfp2 = os.path.join(path_wavelets, nf2)
    nfp3 = os.path.join(path_wavelets, nf3)

    n_bin = 16
    mu_lim = 27
    MJy_lim = 10**((mu_lim - 8.906) / -2.5) * 4.25 * 1E4
    #im_mu = - 2.5 * np.log10( im_MJy / 4.25 * 1E-4 ) + 8.906

    hdu1 = fits.open(nfp1)
    im1 = hdu1[0].data
    #im1[im1 < MJy_lim] = 0.
    im1 = rebin(im1, xbin = n_bin, ybin = n_bin, type = 'SUM')

    hdu2 = fits.open(nfp2)
    im2 = hdu2[0].data
    #im2[im2 < MJy_lim] = 0.
    im2 = rebin(im2, xbin = n_bin, ybin = n_bin, type = 'SUM')

    hdu3 = fits.open(nfp3)
    im3 = hdu3[0].data
    #im3[im3 < MJy_lim] = 0.
    im3 = rebin(im3, xbin = n_bin, ybin = n_bin, type = 'SUM')

    # Normalize
    # im1 = normalize_image(im1)
    # im2 = normalize_image(im2)
    # im3 = normalize_image(im3)
    iml = [normalize_image(im1), normalize_image(im2), normalize_image(im3) ]

    # CLA
    clal = []
    for i, im1 in enumerate(iml):
        for j, im2 in enumerate(iml):
            cla = np.zeros(im1.shape)
            it = np.nditer(im1, flags=['multi_index'])
            for pix in tqdm.tqdm(it):
                x, y = it.multi_index
                idx = np.where( im2 == pix )
                idx = np.array(idx)
                if idx.size != 0:
                    dr = np.sqrt( ( x - idx[0] )**2 + ( y - idx[1] )**2 )
                    dr = dr[dr > 0]

                    if dr.size != 0:
                        closest_idx = np.nanargmin(dr)
                        x_clos = idx[0][closest_idx]
                        y_clos = idx[1][closest_idx]

                        #print(pix, x, y, x_clos, y_clos, dr[np.argmin(dr)])
                        cla[x, y] = dr[np.argmin(dr)]

                else:
                    cla[x,y] = np.nan
            cla[np.where(np.isnan(cla))] = np.nanmax(cla)
            clal.append(cla)


    # Correlation
    # corr12 = correlate2d(im1, im2)
    # corr13 = correlate2d(im1, im3)
    # corr23 = correlate2d(im2, im3)

    # Plots
    interval = AsymmetricPercentileInterval(5, 99.5)
    fig = plt.figure()
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols = (3, 3),  # creates 2x2 grid of axes
                 axes_pad = 0.1, label_mode="L",  # pad between axes in inch.
                 cbar_mode='each', cbar_location='right', cbar_pad=None, cbar_size='5%')
    for ax, cax, cla in zip(grid, grid.cbar_axes, clal):
        #ax[0][0].imshow(im1, origin = 'lower', norm = ImageNormalize( im1, interval = interval, stretch = LogStretch()))
        #ax[0][1].imshow(im2, origin = 'lower', norm = ImageNormalize( im2, interval = interval, stretch = LogStretch()))
        #ax[0][2].imshow(im3, origin = 'lower', norm = ImageNormalize( im3, interval = interval, stretch = LogStretch()))
        im = ax.imshow(cla, origin = 'lower')
        cb = cax.colorbar(im)
        #ax[1][1].imshow(cla13, origin = 'lower')
        #ax[1][2].imshow(cla23, origin = 'lower')
        #ax[2][0].imshow(corr12, origin = 'lower', norm = ImageNormalize( corr12, interval = interval, stretch = LinearStretch()))
        #ax[2][1].imshow(corr13, origin = 'lower', norm = ImageNormalize( corr13, interval = interval, stretch = LinearStretch()))
        #ax[2][2].imshow(corr23, origin = 'lower', norm = ImageNormalize( corr23, interval = interval, stretch = LinearStretch()))
    plt.show()
