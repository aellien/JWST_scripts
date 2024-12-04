import numpy as np
from astropy.io import fits
import os
import glob as glob
import pandas as pd
import dawis as d
import matplotlib.pyplot as plt
from matplotlib import colors, colorbar
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import matplotlib.image as mpimg
from astropy.visualization import *
from astropy.wcs import WCS
from astropy.visualization import make_lupton_rgb
from def_sls_cmap import *
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.offsetbox import AnchoredText
from scipy.ndimage import gaussian_filter
from MJy_to_mu import *
from scipy.ndimage import zoom
import pyregion as pyr
import cmasher
import matplotlib as mpl
import pyregion as pyr

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def SED_tidal_streams():

    plt.ion()

    # Paths, lists & variables
    path_data = '/home/aellien/JWST/data/'
    path_scripts = '/home/aellien/JWST/JWST_scripts'
    path_wavelets = '/home/aellien/JWST/wavelets/out15/'
    path_plots = '/home/aellien/JWST/plots'
    path_analysis = '/home/aellien/JWST/analysis/sed/'

    nfdl = [ {'nf':'jw02736001001_f356w_bkg_rot_crop_input.synth.icl.bcgwavsizesepmask_005_080.fits', 'filt':'f356w', 'path_data':'/home/aellien/JWST/wavelets/out15/', 'chan':'long', 'pix_scale':0.063, 'pixar_sr':9.31E-14, 'phot_corr':0.163, 'n_levels':10 }, \
            {'nf':'jw02736001001_f444w_bkg_rot_crop_input.synth.icl.bcgwavsizesepmask_005_080.fits', 'filt':'f444w', 'path_data':'/home/aellien/JWST/wavelets/out15/', 'chan':'long', 'pix_scale':0.063, 'pixar_sr':9.31E-14, 'phot_corr':0.162, 'n_levels':10 }, \
            {'nf':'jw02736001001_f277w_bkg_rot_crop_input.synth.icl.bcgwavsizesepmask_005_080.fits', 'filt':'f277w', 'path_data':'/home/aellien/JWST/wavelets/out15/', 'chan':'long', 'pix_scale':0.063, 'pixar_sr':9.31E-14, 'phot_corr':0.223, 'n_levels':10 }, \
            {'nf':'jw02736001001_f090w_bkg_rot_crop_warp_nobkg2.synth.icl.bcgwavsizesepmask_005_080.fits', 'path_data':'/home/aellien/JWST/wavelets/out15/', 'filt':'f090w', 'chan':'short', 'pix_scale':0.031, 'phot_corr':-0.174, 'pixar_sr':2.29E-14, 'n_levels':10 }, \
            {'nf':'jw02736001001_f150w_bkg_rot_crop_warp_nobkg2.synth.icl.bcgwavsizesepmask_005_080.fits', 'path_data':'/home/aellien/JWST/wavelets/out15/', 'filt':'f150w', 'chan':'short', 'pix_scale':0.031, 'phot_corr':-0.047, 'pixar_sr':2.31E-14, 'n_levels':10 }, \
            {'nf':'jw02736001001_f200w_bkg_rot_crop_warp_nobkg2.synth.icl.bcgwavsizesepmask_005_080.fits', 'path_data':'/home/aellien/JWST/wavelets/out15/', 'filt':'f200w', 'chan':'short', 'pix_scale':0.031, 'phot_corr':-0.114, 'pixar_sr':2.29E-14, 'n_levels':10 } ]


    filterl = [ 'f090w', 'f150w', 'f200w', 'f277w', 'f356w', 'f444w']
    colors = [ 'gold', 'green', 'paleturquoise', 'mediumaquamarine', 'dodgerblue' ]
    z = 0.3877

    plt.figure()
    ofn = os.path.join(path_analysis, 'streams.txt')

    with open(ofn, 'w+') as of:
        of.write('# id redshift jwst.nircam.F090W jwst.nircam.F090W_err jwst.nircam.F150W jwst.nircam.F150W_err jwst.nircam.F200W jwst.nircam.F200W_err jwst.nircam.F277W jwst.nircam.F277W_err jwst.nircam.F356W jwst.nircam.F356W_err jwst.nircam.F444W jwst.nircam.F444W_err\n')
        for i in range(1, 6):

            # Read region files
            r = pyr.open(os.path.join(path_data, 'streams_flags_pix_long_%d.reg'%i))

            # Read files
            fluxl = []
            outline = '%d %f'%(i, z)
            for filt in filterl:
                for nfd in nfdl:
                    if nfd['filt'] == filt:

                        nfp = os.path.join(nfd['path_data'], nfd['nf'])
                        hdu = fits.open(nfp)
                        recim = hdu[0].data

                        # Mask image
                        mscr = r.get_mask(hdu = hdu[0]) # not python convention
                        n = np.sum(mscr)

                        # Measure flux
                        recim[mscr != True] = 0.
                        ZP_AB = -6.10 - 2.5 * np.log10(nfd['pixar_sr'])
                        flux_err = np.sqrt(n) * np.std(recim[mscr == True])
                        mag_err = -2.5 * np.log10(flux_err) + ZP_AB

                        flux = np.sum(recim)
                        mag = -2.5 * np.log10(flux) + ZP_AB
                        if nfd['chan'] == 'short':
                            mag_corr = mag + nfd['phot_corr']
                            flux_corr = 10**( (ZP_AB - mag_corr) / 2.5 )
                        else:
                            mag_corr = mag
                            flux_corr = flux
                        flux *= nfd['pixar_sr']
                        flux_corr *= nfd['pixar_sr']
                        flux_err *= nfd['pixar_sr']
                        print('%d %s %1.2e %2.1f %1.2e %2.1f %1.2e %2.1f' %(i, filt, flux, mag, flux_err, mag_err, flux_corr, mag_corr))
                        fluxl.append(flux_corr)
                        outline = outline + ' %1.4e'%flux_corr
                        outline = outline + ' %1.4e'%flux_err

            fluxl = np.array(fluxl)
            outline = outline + '\n'
            of.write(outline)

            plt.plot(filterl, \
                     fluxl, \
                     marker = 'o', markerfacecolor = colors[i-1], \
                     markersize = 10, \
                     markeredgecolor = colors[i-1], \
                     linewidth = 2, \
                     color = colors[i-1])

    plt.show(block=True)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def plot_and_make_sed():

    # Paths, lists & variables
    path_data = '/home/aellien/JWST/data/'
    path_scripts = '/home/aellien/JWST/JWST_scripts'
    path_wavelets = '/home/aellien/JWST/wavelets/out15/'
    path_plots = '/home/aellien/JWST/plots'
    path_analysis = '/home/aellien/JWST/analysis/'

    sch = 'WS+BCGSF'
    size_sep = 80
    R_kpc = 400
    lvl_sep = 5
    pixar_sr = np.array([ 2.29E-14, 2.31E-14, 2.29E-14, 9.31E-14, 9.31E-14, 9.31E-14 ])
    ZP_AB = -6.10 - 2.5 * np.log10(pixar_sr)
    phot_corr = np.array([-0.174, -0.047, -0.114, 0, 0, 0])
    #phot_corr = np.array([0, 0, 0., 0, 0, 0])
    #phot_corr = np.array([-0.174, -0.047, -0.114, 0.223, 0.163, 0.162])

    colors = [ 'dodgerblue', 'mediumaquamarine', 'paleturquoise' , 'white' ]

    r = pd.read_excel('/home/aellien/JWST/analysis/results_out5b.xlsx')
    r = r.sort_values(by = 'filter') # Sort values so filters goes from low wavelength to high wavelength

    sed_n_ann = 10 # number of annuli regions, SED
    sed_n_str = 6 # number of tidal stream regions, SED
    z = 0.3877

    # Read SED extraction regions
    areal = []
    hdu = fits.open(os.path.join(path_data, 'jw02736001001_f277w_bkg_rot_crop_input.fits')) # Arbitrary
    for i in range(1, sed_n_ann + 1):
        reg = pyr.open(os.path.join(path_data, 'ellipse_annuli_pix_long_%d.reg'%i))
        ms = reg.get_mask(hdu = hdu[0])
        area = np.sum( ms )
        areal.append(area)
    for i in range(1, sed_n_str + 1):
        reg = pyr.open(os.path.join(path_data, 'streams_flags_pix_long_%d.reg'%i))
        ms = reg.get_mask(hdu = hdu[0])
        area = np.sum( ms )
        areal.append(area)

    plt.figure(figsize=(8,8))
    ofn = os.path.join(path_analysis, 'sed_all_regions.txt')

    with open(ofn, 'w+') as of:
        of.write('# id redshift jwst.nircam.F090W jwst.nircam.F090W_err jwst.nircam.F150W jwst.nircam.F150W_err jwst.nircam.F200W jwst.nircam.F200W_err jwst.nircam.F277W jwst.nircam.F277W_err jwst.nircam.F356W jwst.nircam.F356W_err jwst.nircam.F444W jwst.nircam.F444W_err')

        #---------------------------------------------------------------------------
        # Elliptical annuli
        for i in range(sed_n_ann):

            filters = r[r['Atom selection scheme']==sch][r['lvl_sep']==lvl_sep][r['R_kpc']==R_kpc]['filter']

            # mean flux + corrections + conversion
            flux = r[r['Atom selection scheme']==sch][r['lvl_sep']==lvl_sep][r['R_kpc']==R_kpc]['reg_%d_m'%i].values
            mag = -2.5 * np.log10(flux) + ZP_AB
            mag_corr = mag + phot_corr
            flux = 10**( (ZP_AB - mag_corr) / 2.5 ) * pixar_sr * 1E9 # / areal[i] # 10^9 factor to convert from MJy to mJy (cigale)

            up_flux = r[r['Atom selection scheme']==sch][r['lvl_sep']==lvl_sep][r['R_kpc']==R_kpc]['reg_%d_up'%i].values
            mag = -2.5 * np.log10(up_flux) + ZP_AB
            mag_corr = mag + phot_corr
            up_flux = 10**( (ZP_AB - mag_corr) / 2.5 ) * pixar_sr * 1E9 # / areal[i] # 10^9 factor to convert from MJy to mJy (cigale)
            errup = up_flux - flux

            low_flux = r[r['Atom selection scheme']==sch][r['lvl_sep']==lvl_sep][r['R_kpc']==R_kpc]['reg_%d_low'%i].values
            mag = -2.5 * np.log10(low_flux) + ZP_AB
            mag_corr = mag + phot_corr
            low_flux = 10**( (ZP_AB - mag_corr) / 2.5 ) * pixar_sr * 1E9 #/ areal[i]# 10^9 factor to convert from MJy to mJy (cigale)
            errlow = flux - low_flux

            print(flux)

            outline = '\n%d %f'%(i, z)
            for (wflux, werrlow, werrup) in zip(flux, errlow, errup):
                outline = outline + ' %1.4e'%wflux
                outline = outline + ' %1.4e'%((werrlow + werrup)/2)
            of.write(outline)

            plt.errorbar( x = filters, \
                          y = flux, \
                          yerr = [errup, errlow],\
                          marker = 'o', \
                          markersize = 10, \
                          barsabove = True, \
                          color = 'blue', \
                          alpha = 1 - ( i / sed_n_ann), \
                          elinewidth = 2, \
                          linewidth = 2, \
                          capsize = 4, \
                          linestyle = '-', \
                          label = 'Region %d'%i)

        #---------------------------------------------------------------------------
        # Tidal streams
        k = 0
        for i in range(sed_n_ann, sed_n_ann + sed_n_str):
            k += 1

            filters = r[r['Atom selection scheme']==sch][r['lvl_sep']==lvl_sep][r['R_kpc']==R_kpc]['filter']

            # mean flux + corrections + conversion
            flux = r[r['Atom selection scheme']==sch][r['lvl_sep']==lvl_sep][r['R_kpc']==R_kpc]['reg_%d_m'%i].values
            mag = -2.5 * np.log10(flux) + ZP_AB
            mag_corr = mag + phot_corr
            flux = 10**( (ZP_AB - mag_corr) / 2.5 ) * pixar_sr * 1E9 #/ areal[i] # 10^9 factor to convert from MJy to mJy

            up_flux = r[r['Atom selection scheme']==sch][r['lvl_sep']==lvl_sep][r['R_kpc']==R_kpc]['reg_%d_up'%i].values
            mag = -2.5 * np.log10(up_flux) + ZP_AB
            mag_corr = mag + phot_corr
            up_flux = 10**( (ZP_AB - mag_corr) / 2.5 ) * pixar_sr * 1E9 #/ areal[i] # 10^9 factor to convert from MJy to mJy
            errup = up_flux - flux

            low_flux = r[r['Atom selection scheme']==sch][r['lvl_sep']==lvl_sep][r['R_kpc']==R_kpc]['reg_%d_low'%i].values
            mag = -2.5 * np.log10(low_flux) + ZP_AB
            mag_corr = mag + phot_corr
            low_flux = 10**( (ZP_AB - mag_corr) / 2.5 ) * pixar_sr * 1E9 #/ areal[i] # 10^9 factor to convert from MJy to mJy
            errlow = flux - low_flux

            print(flux)

            outline = '\n%d %f'%(i, z)
            for (wflux, werrlow, werrup) in zip(flux, errlow, errup):
                outline = outline + ' %1.4e'%wflux
                outline = outline + ' %1.4e'%((werrlow + werrup)/2)
            of.write(outline)

            plt.errorbar( x = filters, \
                          y = flux, \
                          yerr = [errup, errlow],\
                          marker = 'o', \
                          markersize = 10, \
                          barsabove = True, \
                          color = 'red', \
                          alpha = 1 - ( k / (1+sed_n_str)), \
                          elinewidth = 2, \
                          linewidth = 2, \
                          capsize = 4, \
                          linestyle = '-', \
                          label = 'Region %d'%i)

        #---------------------------------------------------------------------------
        # ICL Global SED
        filters = r[r['Atom selection scheme']==sch][r['lvl_sep']==lvl_sep][r['R_kpc']==R_kpc]['filter']

        # mean flux + corrections + conversion
        flux = r[r['Atom selection scheme']==sch][r['lvl_sep']==lvl_sep][r['R_kpc']==R_kpc]['F_ICL_m'].values
        mag = -2.5 * np.log10(flux) + ZP_AB
        mag_corr = mag + phot_corr
        flux = 10**( (ZP_AB - mag_corr) / 2.5 ) * pixar_sr * 1E9 #/ areal[i] # 10^9 factor to convert from MJy to mJy

        up_flux = r[r['Atom selection scheme']==sch][r['lvl_sep']==lvl_sep][r['R_kpc']==R_kpc]['F_ICL_up'].values
        mag = -2.5 * np.log10(up_flux) + ZP_AB
        mag_corr = mag + phot_corr
        up_flux = 10**( (ZP_AB - mag_corr) / 2.5 ) * pixar_sr * 1E9 #/ areal[i] # 10^9 factor to convert from MJy to mJy
        errup = up_flux - flux

        low_flux = r[r['Atom selection scheme']==sch][r['lvl_sep']==lvl_sep][r['R_kpc']==R_kpc]['F_ICL_low'].values
        mag = -2.5 * np.log10(low_flux) + ZP_AB
        mag_corr = mag + phot_corr
        low_flux = 10**( (ZP_AB - mag_corr) / 2.5 ) * pixar_sr * 1E9 #/ areal[i] # 10^9 factor to convert from MJy to mJy
        errlow = flux - low_flux

        print(flux)

        outline = '\n%s %f'%('ICL', z)
        for (wflux, werrlow, werrup) in zip(flux, errlow, errup):
            outline = outline + ' %1.4e'%wflux
            outline = outline + ' %1.4e'%((werrlow + werrup)/2)
        of.write(outline)

        plt.errorbar( x = filters, \
                      y = flux, \
                      yerr = [errup, errlow],\
                      marker = 'o', \
                      markersize = 10, \
                      barsabove = True, \
                      color = 'orange', \
                      alpha = 1, \
                      elinewidth = 2, \
                      linewidth = 2, \
                      capsize = 4, \
                      linestyle = '-', \
                      label = 'Total ICL')

        #---------------------------------------------------------------------------
        # Satellites Global SED
        filters = r[r['Atom selection scheme']==sch][r['lvl_sep']==lvl_sep][r['R_kpc']==R_kpc]['filter']

        # mean flux + corrections + conversion
        flux = r[r['Atom selection scheme']==sch][r['lvl_sep']==lvl_sep][r['R_kpc']==R_kpc]['F_gal_m'].values
        mag = -2.5 * np.log10(flux) + ZP_AB
        mag_corr = mag + phot_corr
        flux = 10**( (ZP_AB - mag_corr) / 2.5 ) * pixar_sr * 1E9 #/ areal[i] # 10^9 factor to convert from MJy to mJy

        up_flux = r[r['Atom selection scheme']==sch][r['lvl_sep']==lvl_sep][r['R_kpc']==R_kpc]['F_gal_up'].values
        mag = -2.5 * np.log10(up_flux) + ZP_AB
        mag_corr = mag + phot_corr
        up_flux = 10**( (ZP_AB - mag_corr) / 2.5 ) * pixar_sr * 1E9 #/ areal[i] # 10^9 factor to convert from MJy to mJy
        errup = up_flux - flux

        low_flux = r[r['Atom selection scheme']==sch][r['lvl_sep']==lvl_sep][r['R_kpc']==R_kpc]['F_gal_low'].values
        mag = -2.5 * np.log10(low_flux) + ZP_AB
        mag_corr = mag + phot_corr
        low_flux = 10**( (ZP_AB - mag_corr) / 2.5 ) * pixar_sr * 1E9 #/ areal[i] # 10^9 factor to convert from MJy to mJy
        errlow = flux - low_flux

        print(flux)

        outline = '\n%s %f'%('GAL', z)
        for (wflux, werrlow, werrup) in zip(flux, errlow, errup):
            outline = outline + ' %1.4e'%wflux
            outline = outline + ' %1.4e'%((werrlow + werrup)/2)
        of.write(outline)

        plt.errorbar( x = filters, \
                      y = flux, \
                      yerr = [errup, errlow],\
                      marker = 'o', \
                      markersize = 10, \
                      barsabove = True, \
                      color = 'green', \
                      alpha = 1, \
                      elinewidth = 2, \
                      linewidth = 2, \
                      capsize = 4, \
                      linestyle = '-', \
                      label = 'Total satellites')

        plt.legend()
        plt.ylabel('Flux [mJy.pix$^{-1}$]', fontsize = 15)
        plt.yscale('log')
        plt.xticks(fontsize = 13)
        plt.yticks(fontsize = 13)
        plt.tight_layout()
        plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def plot_PR():

    # Paths, lists & variables
    path_data = '/home/ellien/JWST/data/'
    path_scripts = '/home/ellien/JWST/JWST_scripts'
    path_wavelets = '/home/ellien/JWST/wavelets/out15/'
    path_plots = '/home/ellien/JWST/plots'

    filterl = [ 'f277w', 'f356w', 'f444w']
    schl = ['WS+SF', 'WS+BCGSF+SS']
    size_sepl = [80]
    R_kpcl = [ 128, 200, 400 ]
    lvl_sepl = [ 5, 6 ]
    upliml = [ 5000, 1000, 50, 20 ]

    colors = [ 'dodgerblue', 'mediumaquamarine', 'paleturquoise' , 'white' ]

    r = pd.read_excel('/home/ellien/JWST/analysis/results_out1.xlsx')
    r = r.sort_values(by = 'filter') # Sort values so filters goes from low wavelength to high wavelength


    for lvl_sep in lvl_sepl:

        fig, ax = plt.subplots(4, figsize=(7,9))
        for i, R_kpc in enumerate(R_kpcl):

            for j in range(4):

                filters = r[r['Atom selection scheme']=='WS+SF+SS'][r['lvl_sep']==lvl_sep][r['R_kpc']==R_kpc][r['size_sep']==100]['filter']
                fractions = r[r['Atom selection scheme']=='WS+SF+SS'][r['lvl_sep']==lvl_sep][r['R_kpc']==R_kpc][r['size_sep']==100]['PR_%d_m'%(j+1)] * 1E7
                up_fractions = r[r['Atom selection scheme']=='WS+SF+SS'][r['lvl_sep']==lvl_sep][r['R_kpc']==R_kpc][r['size_sep']==100]['PR_%d_up'%(j+1)] * 1E7
                low_fractions = r[r['Atom selection scheme']=='WS+SF+SS'][r['lvl_sep']==lvl_sep][r['R_kpc']==R_kpc][r['size_sep']==100]['PR_%d_low'%(j+1)] * 1E7
                errup = (up_fractions - fractions).to_numpy()
                errlow = (fractions - low_fractions).to_numpy()
                print('PR_%d_m\n'%(j+1), fractions, filters)
                ax[j].errorbar( x = filters, \
                              y = fractions, \
                              yerr = [errlow, errup],\
                              marker = 'o', markerfacecolor = colors[i], \
                              markersize = 10, \
                              barsabove = True, \
                              markeredgecolor = colors[i], \
                              ecolor = colors[i], \
                              elinewidth = 2, \
                              linewidth = 2, \
                              capsize = 4, \
                              color = colors[i], \
                              label = '$R_{ap}=$%d kpc'%(R_kpc))

                #ax[j].set_ylim(top = upliml[j], bottom = 0)
                ax[j].set_ylabel('$P_{%d}/P_{0}$'%(j+1), fontsize = 15)
                ax[j].tick_params(axis = 'both', labelsize = 13)
                if j == 3:
                    ax[j].set_xticklabels(filters, fontsize = 13)
                else:
                    ax[j].set_xticks([])

        ax[0].legend(loc = 'upper left', bbox_to_anchor=(0, 1.75), fontsize = 13)

        plt.suptitle('Power ratios lvl_sep = %d'%lvl_sep)
        plt.tight_layout()
        plt.subplots_adjust(hspace = 0.1)
        plt.savefig(os.path.join(path_plots, 'PR_vs_filters_%d_with_bcg.pdf'%lvl_sep), format = 'pdf')
    plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def fICL_vs_filters():
    '''
    22/04/2024
    UP TO DATE.
    '''

    # Paths, lists & variables
    path_data = '/home/aellien/JWST/data/'
    path_scripts = '/home/aellien/JWST/JWST_scripts'
    path_wavelets = '/home/aellien/JWST/wavelets/out15/'
    path_plots = '/home/aellien/JWST/plots'

    filterl = [ 'f090w', 'f150w', 'f200w', 'f277w', 'f356w', 'f444w']
    schl = [ ('WS+BCGSF+SS', 'o', 'dodgerblue'), ('WS+SF+SS', '^', 'mediumaquamarine')] #('WS+SF', '--', 'o', 'mediumaquamarine'), ('WS+SF+SS', '--', '*',  'dodgerblue')
    size_sepl = [ (60, '--'), (80, '-'), (100, 'dotted'), (140, '-.' )]
    R_kpcl = [ 400 ]
    colors = [ 'paleturquoise', 'mediumaquamarine', 'dodgerblue', 'white' ]

    r = pd.read_excel('/home/aellien/JWST/analysis/results_out5.xlsx')
    r = r.sort_values(by = 'filter') # Sort values so filters goes from low wavelength to high wavelength

    plt.figure(figsize=(8,5))
    for i, R_kpc in enumerate(R_kpcl):

        for ( sch, ms, cs ) in schl:
            if (sch == 'WS+SF+SS') or (sch == 'WS+BCGSF+SS'):
                for (size_sep, ls) in size_sepl:
                    print(sch, size_sep)
                    filters = r[r['Atom selection scheme']==sch][r['lvl_sep']==5][r['R_kpc']==R_kpc][r['size_sep']==size_sep]['filter']
                    fractions = r[r['Atom selection scheme']==sch][r['lvl_sep']==5][r['R_kpc']==R_kpc][r['size_sep']==size_sep]['f_ICL_m']
                    print(filters, fractions)
                    print(np.mean(fractions))
                    up_fractions = r[r['Atom selection scheme']==sch][r['lvl_sep']==5][r['R_kpc']==R_kpc][r['size_sep']==size_sep]['f_ICL_up']
                    low_fractions = r[r['Atom selection scheme']==sch][r['lvl_sep']==5][r['R_kpc']==R_kpc][r['size_sep']==size_sep]['f_ICL_low']
                    errup = (up_fractions - fractions).to_numpy()
                    errlow = (fractions - low_fractions).to_numpy()

                    alpha = 0.2
                    if size_sep != 80:ms = None
                    if size_sep == 80:ms = 'o'
                    if size_sep == 80:alpha = 1.0

                    plt.errorbar( x = filters, \
                                  y = fractions, \
                                  yerr = [errup, errlow],\
                                  marker = ms, markerfacecolor = cs, \
                                  markersize = 10, \
                                  barsabove = True, \
                                  markeredgecolor = cs, \
                                  ecolor = cs, \
                                  elinewidth = 2, \
                                  linewidth = 2, \
                                  capsize = 2, \
                                  alpha = alpha, \
                                  linestyle = ls, \
                                  color = cs, label = sch + ' %d kpc, R_ap = %d kpc'%(size_sep, R_kpc))

            else:
                filters = r[r['Atom selection scheme']==sch][r['lvl_sep']==6][r['R_kpc']==R_kpc]['filter']
                fractions = r[r['Atom selection scheme']==sch][r['lvl_sep']==6][r['R_kpc']==R_kpc]['f_ICL_m']
                up_fractions = r[r['Atom selection scheme']==sch][r['lvl_sep']==6][r['R_kpc']==R_kpc]['f_ICL_up']
                low_fractions = r[r['Atom selection scheme']==sch][r['lvl_sep']==6][r['R_kpc']==R_kpc]['f_ICL_low']
                errup = (up_fractions - fractions).to_numpy()
                errlow = (fractions - low_fractions).to_numpy()

                plt.errorbar(x = filters, \
                              y = fractions, \
                              yerr = [errup, errlow],\
                              marker = ms, markerfacecolor = cs, \
                              markersize = 10, \
                              barsabove = True, \
                              markeredgecolor = cs, \
                              ecolor = cs, \
                              elinewidth = 2, \
                              linewidth = 2, \
                              capsize = 4, \
                              color = cs)

    custom_lines = [ Line2D([0], [0], color = 'mediumaquamarine', marker = None, lw = 4, markersize = 0 ), \
                     Line2D([0], [0], color = 'dodgerblue', marker = None, lw = 4, markersize = 0 ) ]

    custom_lines2 = [ Line2D([0], [0], color = 'black', marker = None, lw = 2, markersize = 0, ls = '--' ), \
                      Line2D([0], [0], color = 'black', marker = None, lw = 2, markersize = 0 , ls = '-'), \
                      Line2D([0], [0], color = 'black', marker = None, lw = 2, markersize = 0, ls = 'dotted' ), \
                      Line2D([0], [0], color = 'black', marker = None, lw = 2, markersize = 0 , ls = '-.') ]

    leg1 = plt.legend(handles = custom_lines, labels = ['ICL', 'ICL+BCG'], loc = 'upper left', fontsize = 15)
    leg2 = plt.legend(handles = custom_lines2, labels = ['SS = 60 kpc', 'SS = 80 kpc','SS = 100 kpc', 'SS = 140 kpc'], loc = 'upper right', fontsize = 15)
    plt.gca().add_artist(leg1)
    plt.gca().add_artist(leg2)

    plt.ylim(top = 0.6, bottom = 0.1)
    plt.ylabel('ICL fractions', fontsize = 18)
    plt.tight_layout()
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    plt.gca().tick_params(axis='x', labelsize=15)
    plt.savefig(os.path.join(path_plots, 'fICL_vs_filters.pdf'), format = 'pdf')
    plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def Fgal_vs_filters():

    # Paths, lists & variables
    path_data = '/home/aellien/JWST/data/'
    path_scripts = '/home/aellien/JWST/JWST_scripts'
    path_wavelets = '/home/aellien/JWST/wavelets/out15/'
    path_plots = '/home/aellien/JWST/plots'

    filterl = [ 'f090w', 'f150w', 'f200w', 'f277w', 'f356w', 'f444w']
    schl = [ ('WS+BCGSF+SS', '-', 's'), ('WS+SF+SS', '--', 'o')]
    size_sepl = [ 80 ]
    R_kpcl = [ 128, 200, 400 ]
    colors = [ 'paleturquoise', 'mediumaquamarine', 'dodgerblue', 'white' ]
    pixar_sr = np.array([ 2.29E-14, 2.31E-14, 2.29E-14, 9.31E-14, 9.31E-14, 9.31E-14 ])
    ZP_AB = -6.10 - 2.5 * np.log10(pixar_sr)
    phot_corr = np.array([-0.174, -0.047, -0.114, 0, 0, 0])

    r = pd.read_excel('/home/aellien/JWST/analysis/results_out5.xlsx')
    r = r.sort_values(by = 'filter') # Sort values so filters goes from low wavelength to high wavelength

    plt.figure(figsize=(8,8))
    for i, R_kpc in enumerate(R_kpcl):

        for ( sch, ls, ms ) in schl:

            if (sch == 'WS+SF+SS') or (sch == 'WS+BCGSF+SS'):

                for size_sep in size_sepl:

                    filters = r[r['Atom selection scheme']==sch][r['lvl_sep']==5][r['R_kpc']==R_kpc][r['size_sep']==size_sep]['filter']
                    #import pdb; pdb.set_trace()
                    # mean flux + corrections + conversion
                    flux = r[r['Atom selection scheme']==sch][r['lvl_sep']==5][r['R_kpc']==R_kpc][r['size_sep']==size_sep]['F_gal_m'].values
                    mag = -2.5 * np.log10(flux) + ZP_AB
                    mag_corr = mag + phot_corr
                    flux = 10**( (ZP_AB - mag_corr) / 2.5 ) * pixar_sr

                    up_flux = r[r['Atom selection scheme']==sch][r['lvl_sep']==5][r['R_kpc']==R_kpc][r['size_sep']==size_sep]['F_gal_up'].values
                    mag = -2.5 * np.log10(up_flux) + ZP_AB
                    mag_corr = mag + phot_corr
                    up_flux = 10**( (ZP_AB - mag_corr) / 2.5 ) * pixar_sr
                    errup = up_flux - flux

                    low_flux = r[r['Atom selection scheme']==sch][r['lvl_sep']==5][r['R_kpc']==R_kpc][r['size_sep']==size_sep]['F_gal_low'].values
                    mag = -2.5 * np.log10(low_flux) + ZP_AB
                    mag_corr = mag + phot_corr
                    low_flux = 10**( (ZP_AB - mag_corr) / 2.5 ) * pixar_sr
                    errlow = flux - low_flux

                    print('GAL', filters)
                    print('GAL', flux)

                    plt.errorbar( x = filters, \
                                  y = flux, \
                                  yerr = [errup, errlow],\
                                  marker = ms, markerfacecolor = colors[i], \
                                  markersize = 10, \
                                  barsabove = True, \
                                  markeredgecolor = colors[i], \
                                  ecolor = colors[i], \
                                  elinewidth = 2, \
                                  linewidth = 2, \
                                  capsize = 4, \
                                  linestyle = ls, \
                                  color = colors[i] )# label = sch + ' %d kpc, R_ap = %d kpc'%(size_sep, R_kpc))

            else:
                filters = r[r['Atom selection scheme']==sch][r['lvl_sep']==5][r['R_kpc']==R_kpc]['filter']
                flux = r[r['Atom selection scheme']==sch][r['lvl_sep']==5][r['R_kpc']==R_kpc]['F_gal_m']# / (4 * np.pi**2 * R_kpc )
                up_flux = r[r['Atom selection scheme']==sch][r['lvl_sep']==5][r['R_kpc']==R_kpc]['F_gal_up']# / (4 * np.pi**2 * R_kpc )
                low_flux = r[r['Atom selection scheme']==sch][r['lvl_sep']==5][r['R_kpc']==R_kpc]['F_gal_low']# / (4 * np.pi**2 * R_kpc )
                errup = (up_flux - flux).to_numpy()
                errlow = (flux - low_flux).to_numpy()

                plt.errorbar(x = filters, \
                              y = flux, \
                              yerr = [errup, errlow],\
                              marker = 'o', markerfacecolor = colors[i], \
                              markersize = 10, \
                              barsabove = True, \
                              markeredgecolor = colors[i], \
                              ecolor = colors[i], \
                              elinewidth = 2, \
                              linewidth = 2, \
                              capsize = 4, \
                              color = colors[i])

    custom_lines = [ Line2D([0], [0], color = 'paleturquoise', marker = None, lw = 4, markersize = 0 ), \
                     Line2D([0], [0], color = 'mediumaquamarine', marker = None, lw = 4, markersize = 0 ), \
                     Line2D([0], [0], color = 'dodgerblue', marker = None, lw = 4, markersize = 0 ) ]
    plt.legend(handles = custom_lines, labels = ['$R_{ap} = %d$ kpc'%128, '$R_{ap} = %d$ kpc'%200, '$R_{ap} = %d$ kpc'%400], loc = 'upper left', bbox_to_anchor=(0, 1.25), fontsize = 15)
    #plt.ylim(top = 1., bottom = 0.)
    plt.ylabel('Galaxy flux [MJy.pix$^{-1}$]', fontsize = 15)
    plt.tight_layout()
    plt.xticks(fontsize = 13)
    plt.yticks(fontsize = 13)
    plt.savefig(os.path.join(path_plots, 'Fgal_vs_filters.pdf'), format = 'pdf')
    plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def FICL_vs_filters():
    '''
    NO PHOTOMETRIC CORRECTION BTW SHORT & LONG HERE
    '''
    # Paths, lists & variables
    path_data = '/home/aellien/JWST/data/'
    path_scripts = '/home/aellien/JWST/JWST_scripts'
    path_wavelets = '/home/aellien/JWST/wavelets/out15/'
    path_plots = '/home/aellien/JWST/plots'

    filterl = [ 'f277w', 'f356w', 'f444w']
    schl = [ ('WS+BCGSF+SS', '-', 's'), ('WS+SF+SS', '--', 'o')]
    size_sepl = [ 80 ]
    R_kpcl = [ 128, 200, 400 ]
    colors = [ 'paleturquoise', 'mediumaquamarine', 'dodgerblue', 'white' ]
    pixar_sr = np.array([ 2.29E-14, 2.31E-14, 2.29E-14, 9.31E-14, 9.31E-14, 9.31E-14 ])
    print(pixar_sr)

    r = pd.read_excel('/home/aellien/JWST/analysis/results_out5.xlsx')
    r = r.sort_values(by = 'filter') # Sort values so filters goes from low wavelength to high wavelength

    plt.figure(figsize=(8,8))
    for i, R_kpc in enumerate(R_kpcl):

        for ( sch, ls, ms ) in schl:
            if (sch == 'WS+SF+SS') or (sch == 'WS+BCGSF+SS'):
                for size_sep in size_sepl:
                    filters = r[r['Atom selection scheme']==sch][r['lvl_sep']==5][r['R_kpc']==R_kpc][r['size_sep']==size_sep]['filter']
                    flux = r[r['Atom selection scheme']==sch][r['lvl_sep']==5][r['R_kpc']==R_kpc][r['size_sep']==size_sep]['F_ICL_m'].values * pixar_sr
                    up_flux = r[r['Atom selection scheme']==sch][r['lvl_sep']==5][r['R_kpc']==R_kpc][r['size_sep']==size_sep]['F_ICL_up'].values * pixar_sr #(4 * np.pi**2 * R_kpc )
                    low_flux = r[r['Atom selection scheme']==sch][r['lvl_sep']==5][r['R_kpc']==R_kpc][r['size_sep']==size_sep]['F_ICL_low'].values * pixar_sr #(4 * np.pi**2 * R_kpc )
                    errup = up_flux - flux
                    errlow = flux - low_flux

                    print('ICL\n', filters, flux, pixar_sr)

                    plt.errorbar( x = filters, \
                                  y = flux, \
                                  yerr = [errup, errlow],\
                                  marker = ms, markerfacecolor = colors[i], \
                                  markersize = 10, \
                                  barsabove = True, \
                                  markeredgecolor = colors[i], \
                                  ecolor = colors[i], \
                                  elinewidth = 2, \
                                  linewidth = 2, \
                                  capsize = 4, \
                                  linestyle = ls, \
                                  color = colors[i] )# label = sch + ' %d kpc, R_ap = %d kpc'%(size_sep, R_kpc))

            else:
                filters = r[r['Atom selection scheme']==sch][r['lvl_sep']==5][r['R_kpc']==R_kpc]['filter']
                flux = r[r['Atom selection scheme']==sch][r['lvl_sep']==5][r['R_kpc']==R_kpc]['F_ICL_m']# / (4 * np.pi**2 * R_kpc )
                up_flux = r[r['Atom selection scheme']==sch][r['lvl_sep']==5][r['R_kpc']==R_kpc]['F_ICL_up']# / (4 * np.pi**2 * R_kpc )
                low_flux = r[r['Atom selection scheme']==sch][r['lvl_sep']==5][r['R_kpc']==R_kpc]['F_ICL_low']# / (4 * np.pi**2 * R_kpc )
                errup = (up_flux - flux).to_numpy()
                errlow = (flux - low_flux).to_numpy()

                plt.errorbar(x = filters, \
                              y = flux, \
                              yerr = [errup, errlow],\
                              marker = 'o', markerfacecolor = colors[i], \
                              markersize = 10, \
                              barsabove = True, \
                              markeredgecolor = colors[i], \
                              ecolor = colors[i], \
                              elinewidth = 2, \
                              linewidth = 2, \
                              capsize = 4, \
                              color = colors[i])

    custom_lines = [ Line2D([0], [0], color = 'paleturquoise', marker = None, lw = 4, markersize = 0 ), \
                     Line2D([0], [0], color = 'mediumaquamarine', marker = None, lw = 4, markersize = 0 ), \
                     Line2D([0], [0], color = 'dodgerblue', marker = None, lw = 4, markersize = 0 ) ]
    plt.legend(handles = custom_lines, labels = ['$R_{ap} = %d$ kpc'%128, '$R_{ap} = %d$ kpc'%200, '$R_{ap} = %d$ kpc'%400], loc = 'upper left', bbox_to_anchor=(0, 1.25), fontsize = 15)
    #plt.ylim(top = 1., bottom = 0.)
    plt.ylabel('ICL flux [MJy.pix$^{-1}$]', fontsize = 15)
    plt.tight_layout()
    plt.xticks(fontsize = 13)
    plt.yticks(fontsize = 13)
    plt.savefig(os.path.join(path_plots, 'FICL_vs_filters.pdf'), format = 'pdf')
    plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def fICL_vs_lvlsep():

    # Paths, lists & variables
    path_data = '/home/aellien/JWST/data/'
    path_scripts = '/home/aellien/JWST/JWST_scripts'
    path_wavelets = '/home/aellien/JWST/wavelets/out15/'
    path_plots = '/home/aellien/JWST/plots'

    #filterl = [ 'f090w', 'f150w', 'f200w']
    filterl = [ 'f277w', 'f356w', 'f444w']
    lvl_sepl = [ 3, 4, 5, 6, 7 ]
    schl = ['WS+SF','WS+SF+SS']
    size_sepl = [ 80 ]
    R_kpcl = [ 128, 200, 400 ]
    colors = [ 'dodgerblue', 'mediumaquamarine', 'paleturquoise' , 'white' ]
    markerl = ['s', '^', 'o']
    lsl = ['-', '--']

    r = pd.read_excel('/home/aellien/JWST/analysis/results_out5.xlsx')
    r = r.sort_values(by = 'filter') # Sort values so filters goes from low wavelength to high wavelength

    for j, R_kpc in enumerate(R_kpcl):

        plt.figure(figsize=(8,8))
        for sch in schl:

            for i, filter in enumerate(filterl):

                if sch == 'WS+SF+SS':

                    for k, size_sep in enumerate(size_sepl):

                        fractions = r[r['Atom selection scheme']==sch][r['filter']==filter][r['size_sep']==size_sep][r['R_kpc']==R_kpc].sort_values(by = 'lvl_sep')['f_ICL_m']
                        up_fractions = r[r['Atom selection scheme']==sch][r['filter']==filter][r['size_sep']==size_sep][r['R_kpc']==R_kpc].sort_values(by = 'lvl_sep')['f_ICL_up']
                        low_fractions = r[r['Atom selection scheme']==sch][r['filter']==filter][r['size_sep']==size_sep][r['R_kpc']==R_kpc].sort_values(by = 'lvl_sep')['f_ICL_low']
                        errup = (up_fractions - fractions).to_numpy()
                        errlow = (fractions - low_fractions).to_numpy()

                        plt.errorbar( x = lvl_sepl, \
                                      y = fractions, \
                                      yerr = [errup, errlow],\
                                      marker = markerl[k], markerfacecolor = colors[i], \
                                      markersize = 10, \
                                      barsabove = True, \
                                      markeredgecolor = colors[i], \
                                      ecolor = colors[i], \
                                      elinewidth = 2, \
                                      linewidth = 2, \
                                      capsize = 4, \
                                      linestyle = '-', \
                                      color = colors[i], \
                                      label = '%s %s %d kpc'%(filter, sch, size_sep))

                else:

                    fractions = r[r['Atom selection scheme']==sch][r['filter']==filter][r['R_kpc']==R_kpc].sort_values(by = 'lvl_sep')['f_ICL_m']
                    up_fractions = r[r['Atom selection scheme']==sch][r['filter']==filter][r['R_kpc']==R_kpc].sort_values(by = 'lvl_sep')['f_ICL_up']
                    low_fractions = r[r['Atom selection scheme']==sch][r['filter']==filter][r['R_kpc']==R_kpc].sort_values(by = 'lvl_sep')['f_ICL_low']
                    errup = (up_fractions - fractions).to_numpy()
                    errlow = (fractions - low_fractions).to_numpy()
                    print(sch, lvl_sepl, fractions)

                    plt.errorbar( x = lvl_sepl, \
                                  y = fractions, \
                                  yerr = [errup, errlow],\
                                  marker = 'o', markerfacecolor = colors[i], \
                                  markersize = 10, \
                                  barsabove = True, \
                                  markeredgecolor = colors[i], \
                                  ecolor = colors[i], \
                                  elinewidth = 2, \
                                  linewidth = 2, \
                                  capsize = 4, \
                                  linestyle = '--', \
                                  color = colors[i], \
                                  label = '%s %s'%(filter, sch))


            plt.legend(loc='upper right', fontsize = 15)
            #plt.ylim(top = 0.4)
            plt.ylabel('ICL+BCG fractions', fontsize = 15)
            plt.xlabel('Wavelet scale', fontsize = 15)
            plt.suptitle('$R_{ap}=%d$ kpc'%R_kpc)
            plt.tight_layout()
            plt.xticks(lvl_sepl, fontsize = 13)
            plt.yticks(fontsize = 13)
            plt.savefig(os.path.join(path_plots, 'fICL_vs_lvlsep_%d_bcg.pdf'%R_kpc), format = 'pdf')
    plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def rebin(im, xbin = 2, ybin = 2, type = 'SUM'):

    xedge = np.shape(im)[0]%xbin
    yedge = np.shape(im)[1]%ybin
    im = im[xedge:,yedge:]
    binim = np.reshape(im,(int(np.shape(im)[0]/xbin),xbin,int(np.shape(im)[1]/ybin),ybin))

    if type == 'MEAN':
        binim = np.mean(binim,axis=3)
        binim = np.mean(binim,axis=1)
    elif type == 'SUM':
        binim = np.sum(binim,axis=3)
        binim = np.sum(binim,axis=1)

    return binim

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def plot_array_recim_long():
    '''
    OUTDATED
    '''

    # Paths, lists & variables
    path_data = '/home/aellien/JWST/data/'
    path_wavelets = '/home/aellien/JWST/wavelets/out15/'
    path_plots = '/home/aellien/JWST/plots'

    filterl = [ ('f150w', 'short'), ('f356w', 'long')]
    oiml = []
    recl = []
    resl = []
    binf = 2
    for (filt, chan) in filterl:

        if chan == 'long':
            nf = 'jw02736001001_%s_bkg_rot_crop_input.fits'%filt
            nfp = os.path.join(path_data, nf)
            oiml.append( fits.getdata(nfp))

            nf = 'jw02736001001_%s_bkg_rot_crop_input.synth.restored.fits'%filt
            nfp = os.path.join(path_wavelets, nf)
            recl.append(fits.getdata(nfp))

            nf = 'jw02736001001_%s_bkg_rot_crop_input.synth.residuals.fits'%filt
            nfp = os.path.join(path_wavelets, nf)
            resl.append(fits.getdata(nfp))
        else:
            nf = 'jw02736001001_%s_bkg_rot_crop_warp_nobkg2.fits'%filt
            nfp = os.path.join(path_data, nf)
            oiml.append( fits.getdata(nfp))

            nf = 'jw02736001001_%s_bkg_rot_crop_warp_nobkg2.synth.restored.fits'%filt
            nfp = os.path.join(path_wavelets, nf)
            recl.append(fits.getdata(nfp))

            nf = 'jw02736001001_%s_bkg_rot_crop_warp_nobkg2.synth.residuals.fits'%filt
            nfp = os.path.join(path_wavelets, nf)
            resl.append(fits.getdata(nfp))

    fig, ax = plt.subplots(2, 3, figsize = (11,12))
    sls = sls_cmap()
    cmap = cmasher.seasons
    resl = np.array(resl)

    for i, (filt, chan) in enumerate(filterl):

        oim = rebin(oiml[i], binf, binf, 'SUM')
        poim = ax[i][0].imshow( oim, norm = ImageNormalize( oim, interval = ZScaleInterval(), \
                                stretch = LinearStretch()), cmap = cmap, origin = 'lower')
        ax[i][0].get_xaxis().set_ticks([])
        ax[i][0].get_yaxis().set_ticks([])
        ax[i][0].set_ylabel('%s'%filt, fontsize = 15)
        divider = make_axes_locatable(ax[i][0])
        cax = divider.append_axes("bottom", size="5%", pad=0.05)
        caxre = fig.colorbar( poim, cax = cax, \
                                    cmap = sls, \
                                    orientation = 'horizontal', \
                                    format = '%2.1f',\
                                    pad = 0,\
                                    shrink = 1.0,\
                                    ticklocation = 'bottom' )
        caxre.ax.tick_params(labelsize=15)
        if i == 1:
            cax.set_xlabel('MJy/sr', fontsize = 15)

        rec = rebin(recl[i], binf, binf, 'SUM')
        poim = ax[i][1].imshow( rec, norm = ImageNormalize( oim, interval = ZScaleInterval(), \
                                stretch = LinearStretch()), cmap = cmap, origin = 'lower')
        ax[i][1].get_xaxis().set_ticks([])
        ax[i][1].get_yaxis().set_ticks([])
        divider = make_axes_locatable(ax[i][1])
        cax = divider.append_axes("bottom", size="5%", pad=0.05)
        caxre = fig.colorbar( poim, cax = cax, \
                                    orientation = 'horizontal', \
                                    format = '%2.1f',\
                                    pad = 0,\
                                    shrink = 1.0,\
                                    ticklocation = 'bottom' )
        caxre.ax.tick_params(labelsize=15)
        if i == 1:
            cax.set_xlabel('MJy/sr', fontsize = 15)

        res = rebin(resl[i], binf, binf, 'SUM')
        print(filt, np.nanmean(res))
        poim = ax[i][2].imshow( (res - np.nanmean(res) ) / np.nanmean(res), vmax = 10, vmin = -10, cmap = cmap, origin = 'lower') #PuOr
        ax[i][2].get_xaxis().set_ticks([])
        ax[i][2].get_yaxis().set_ticks([])
        divider = make_axes_locatable(ax[i][2])
        cax = divider.append_axes("bottom", size="5%", pad=0.05)
        caxre = fig.colorbar( poim, cax = cax, \
                                    orientation = 'horizontal', \
                                    format = '%2.1f',\
                                    pad = 0,\
                                    shrink = 1.0,\
                                    ticklocation = 'bottom' )
        caxre.ax.tick_params(labelsize=15)
        at = AnchoredText("<r>=%1.3f MJy/sr"%np.nanmean(res), loc='upper right', \
                                                prop = dict(size=12), frameon=True)
        ax[i][2].add_artist(at)
        if i == 1:
            cax.set_xlabel(r'(r - <r>)/<r>[%]', fontsize = 15)

    ax[0][0].set_title('Original image', fontsize = 15)
    ax[0][1].set_title('Restored image', fontsize = 15)
    ax[0][2].set_title('Residuals', fontsize = 15)

    plt.tight_layout()
    #plt.subplots_adjust( left=0.05, bottom=0.05, right=0.98, top=0.98, wspace=0.03, hspace=0.1)
    plt.savefig(os.path.join(path_plots, 'plot_array_recim_long_f356w.pdf'), format = 'pdf', dpi = 500)
    plt.show()

    return

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def plot_array_scattered_recim_short():

    # Paths, lists & variables
    path_data = '/home/aellien/JWST/data/'
    path_wavelets = '/home/aellien/JWST/wavelets/out20/'
    path_plots = '/home/aellien/JWST/plots'

    filterl = [ 'f090w', 'f150w', 'f200w']
    oiml = []
    combbkgl = []
    nobkgl = []
    recl = []
    resl = []

    binf = 2
    for filt in filterl:

        nfp = os.path.join(path_data, 'jw02736001001_%s_bkg_rot_crop_warp.fits'%filt)
        hdu = fits.open(nfp)
        oim = hdu[1].data
        oiml.append(oim[:2045, :2045])

        nfp = os.path.join(path_data, 'jw02736001001_%s_bkg_rot_crop_warp_nobkg2.fits'%filt)
        nobkg = fits.getdata(nfp)
        nobkgl.append(nobkg[:2045, :2045])

        nfp = os.path.join(path_data, 'jw02736001001_%s_bkg_rot_crop_warp_combbkg.fits'%filt)
        hdu = fits.open(nfp)
        combbkg = hdu[0].data
        combbkgl.append(combbkg[:2045, :2045])

        nf = 'jw02736001001_%s_bkg_rot_crop_warp_nobkg2.synth.restored.fits'%filt
        nfp = os.path.join(path_wavelets, nf)
        recl.append(fits.getdata(nfp)[:2045, :2045])

        nf = 'jw02736001001_%s_bkg_rot_crop_warp_nobkg2.synth.residuals.fits'%filt
        nfp = os.path.join(path_wavelets, nf)
        resl.append(fits.getdata(nfp)[:2045, :2045])

    fig, ax = plt.subplots(3, 5, figsize = (12,8))
    sls = sls_cmap()
    cmap = cmasher.seasons

    for i, filt in enumerate(filterl):

        oim = rebin(oiml[i], binf, binf, 'SUM')
        poim = ax[i][0].imshow( oim, norm = ImageNormalize( oim, interval = ZScaleInterval(), \
                                stretch = LinearStretch()), cmap = cmap, origin = 'lower')
        ax[i][0].get_xaxis().set_ticks([])
        ax[i][0].get_yaxis().set_ticks([])
        ax[i][0].set_ylabel('%s'%filterl[i], fontsize = 15)
        divider = make_axes_locatable(ax[i][0])
        cax = divider.append_axes("bottom", size="5%", pad=0.05)
        caxre = fig.colorbar( poim, cax = cax, \
                                    cmap = sls, \
                                    orientation = 'horizontal', \
                                    format = '%2.1f',\
                                    pad = 0,\
                                    shrink = 1.0,\
                                    ticklocation = 'bottom' )
        caxre.ax.tick_params(labelsize = 15)
        if i == 2:
            cax.set_xlabel('MJy/sr', fontsize = 15)

        combbkg = rebin(combbkgl[i], binf, binf, 'SUM')
        poim = ax[i][1].imshow( combbkg, norm = ImageNormalize( oim, interval = ZScaleInterval(), \
                                stretch = LinearStretch()), cmap = cmap, origin = 'lower')
        ax[i][1].get_xaxis().set_ticks([])
        ax[i][1].get_yaxis().set_ticks([])
        divider = make_axes_locatable(ax[i][1])
        cax = divider.append_axes("bottom", size="5%", pad=0.05)
        caxre = fig.colorbar( poim, cax = cax, \
                                    orientation = 'horizontal', \
                                    format = '%2.1f',\
                                    pad = 0,\
                                    shrink = 1.0,\
                                    ticklocation = 'bottom' )
        caxre.ax.tick_params(labelsize=15)
        if i == 2:
            cax.set_xlabel('MJy/sr', fontsize = 15)

        nobkg = rebin(nobkgl[i], binf, binf, 'SUM')
        poim = ax[i][2].imshow( nobkg, norm = ImageNormalize( oim, interval = ZScaleInterval(), \
                        stretch = LinearStretch()), cmap = cmap, origin = 'lower')
        ax[i][2].get_xaxis().set_ticks([])
        ax[i][2].get_yaxis().set_ticks([])
        divider = make_axes_locatable(ax[i][2])
        cax = divider.append_axes("bottom", size="5%", pad=0.05)
        caxre = fig.colorbar( poim, cax = cax, \
                                    orientation = 'horizontal', \
                                    format = '%2.1f',\
                                    pad = 0,\
                                    shrink = 1.0,\
                                    ticklocation = 'bottom' )
        caxre.ax.tick_params(labelsize=15)
        if i == 2:
            cax.set_xlabel('MJy/sr', fontsize = 15)

        rec = rebin(recl[i], binf, binf, 'SUM')
        poim = ax[i][3].imshow( rec, norm = ImageNormalize( oim, interval = ZScaleInterval(), \
                                stretch = LinearStretch()), cmap = cmap, origin = 'lower')
        ax[i][3].get_xaxis().set_ticks([])
        ax[i][3].get_yaxis().set_ticks([])
        divider = make_axes_locatable(ax[i][3])
        cax = divider.append_axes("bottom", size="5%", pad=0.05)
        caxre = fig.colorbar( poim, cax = cax, \
                                    orientation = 'horizontal', \
                                    format = '%2.1f',\
                                    pad = 0,\
                                    shrink = 1.0,\
                                    ticklocation = 'bottom' )
        caxre.ax.tick_params(labelsize=15)
        if i == 2:
            cax.set_xlabel('MJy/sr', fontsize = 15)

        res = rebin(resl[i], binf, binf, 'SUM')
        res[res < -1000] = -1000
        print(filt, np.nanmean(res))
        print(np.nanmax(res), np.nanmin(res))
        poim = ax[i][4].imshow( (res - np.nanmean(res) ) / np.nanmean(res), vmax = 10, vmin = -10, cmap = cmap, origin = 'lower') #PuOr
        ax[i][4].get_xaxis().set_ticks([])
        ax[i][4].get_yaxis().set_ticks([])
        divider = make_axes_locatable(ax[i][4])
        cax = divider.append_axes("bottom", size="5%", pad=0.05)
        caxre = fig.colorbar( poim, cax = cax, \
                                    orientation = 'horizontal', \
                                    format = '%2.1f',\
                                    pad = 0,\
                                    shrink = 1.0,\
                                    ticklocation = 'bottom' )
        caxre.ax.tick_params(labelsize=15)
        at = AnchoredText("<r>=%1.3f MJy/sr"%np.nanmean(res), loc='upper right', \
                                                prop = dict(size=12), frameon=True)
        ax[i][4].add_artist(at)
        if i == 2:
            cax.set_xlabel(r'(r - <r>)/<r>[%]', fontsize = 15)

    ax[0][0].set_title('Original image', fontsize = 15)
    ax[0][1].set_title('Background', fontsize = 15)
    ax[0][2].set_title('Corrected image', fontsize = 15)
    ax[0][3].set_title('Restored image', fontsize = 15)
    ax[0][4].set_title('Residuals', fontsize = 15)

    plt.tight_layout()
    #plt.subplots_adjust( left=0.05, bottom=0.05, right=0.98, top=0.98, wspace=0.03, hspace=0.1)
    plt.savefig(os.path.join(path_plots, 'plot_array_recim_short.pdf'), format = 'pdf', dpi = 500)
    plt.show()

    return

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def plot_array_icl_maps_all_filters():

    # Paths, lists & variables
    path_data = '/home/aellien/JWST/data/'
    path_scripts = '/home/aellien/JWST/JWST_scripts'
    path_wavelets = '/home/aellien/JWST/wavelets/out20/'
    path_plots = '/home/aellien/JWST/plots'

    nfl = [ {'nf':'jw02736001001_f356w_bkg_rot_crop_input.fits', 'filt':'f356w', 'chan':'long', 'pix_scale':0.063, 'pixar_sr':9.31E-14, 'n_levels':10, 'lvl_sep_max':999 }, \
            {'nf':'jw02736001001_f444w_bkg_rot_crop_input.fits', 'filt':'f444w', 'chan':'long', 'pix_scale':0.063, 'pixar_sr':9.31E-14, 'n_levels':10, 'lvl_sep_max':999 }, \
            {'nf':'jw02736001001_f277w_bkg_rot_crop_input.fits', 'filt':'f277w', 'chan':'long', 'pix_scale':0.063, 'pixar_sr':9.31E-14, 'n_levels':10, 'lvl_sep_max':999 }, \
            {'nf':'jw02736001001_f090w_bkg_rot_crop_warp_nobkg2.fits', 'filt':'f090w', 'chan':'short', 'pix_scale':0.031, 'pixar_sr':2.29E-14, 'n_levels':10, 'lvl_sep_max':999 }, \
            {'nf':'jw02736001001_f150w_bkg_rot_crop_warp_nobkg2.fits', 'filt':'f150w', 'chan':'short', 'pix_scale':0.031, 'pixar_sr':2.31E-14, 'n_levels':10, 'lvl_sep_max':999 }, \
            {'nf':'jw02736001001_f200w_bkg_rot_crop_warp_nobkg2.fits', 'filt':'f200w', 'chan':'short', 'pix_scale':0.031, 'pixar_sr':2.29E-14, 'n_levels':10, 'lvl_sep_max':999 } ]
            #{'nf':'jw02736001001_f090w_bkg_rot_crop_input.fits', 'filt':'f090w', 'chan':'short', 'pix_scale':0.031, 'pixar_sr':2.29E-14, 'n_levels':11, 'lvl_sep_max':8 }, \
            #{'nf':'jw02736001001_f150w_bkg_rot_crop_input.fits', 'filt':'f150w', 'chan':'short', 'pix_scale':0.031, 'pixar_sr':2.31E-14, 'n_levels':11, 'lvl_sep_max':8 }, \
            #{'nf':'jw02736001001_f200w_bkg_rot_crop_input.fits', 'filt':'f200w', 'chan':'short', 'pix_scale':0.031, 'pixar_sr':2.29E-14, 'n_levels':11, 'lvl_sep_max':8 } ]

    filterl = [ 'f090w', 'f150w', 'f200w', 'f277w', 'f356w', 'f444w' ]
    n_rebin = 2.02983 # 4150 / 2045
    mu_lim = 30
    lvl_sepl = [ 3, 4, 5, 6, 7 ]
    size_sepl = [ 60, 80, 100, 140, 200 ] # kpc
    sls = sls_cmap()
    sls_r = ListedColormap(sls.colors[::-1])


    for filt in filterl:
        print('\n%s - making ICL synthesis map grid - mu_lim = %d mag/arcsec2'%(filt, mu_lim))

        for nf in nfl:

            if filt == nf['filt']:

                # Photometry
                if nf['chan'] == 'short':
                    arcsecar_sr = nf['pixar_sr'] / (n_rebin * nf['pix_scale']**2) # short channel images are rebinned
                else:
                    arcsecar_sr = nf['pixar_sr'] / (nf['pix_scale']**2)

                # Read all files
                recml = []
                recl = []
                recmsl = []

                for lvl_sep in lvl_sepl:

                    n = nf['nf'][:-4] + 'synth.icl.wavsepmask_%03d.fits' %(lvl_sep)
                    nfp = os.path.join(path_wavelets, n)
                    hdu = fits.open(nfp)
                    im_MJy = hdu[1].data
                    #im_MJy = rebin(im_MJy, 2, 2)
                    im_mu = MJy_to_mu(im_MJy, arcsecar_sr, mu_lim)
                    recml.append(im_mu)

                    n = nf['nf'][:-4] + 'synth.icl.wavsep_%03d.fits'%(lvl_sep)
                    nfp = os.path.join(path_wavelets, n)
                    hdu = fits.open(nfp)
                    im_MJy = hdu[1].data
                    #im_MJy = rebin(im_MJy, 2, 2)
                    im_mu = MJy_to_mu(im_MJy, arcsecar_sr, mu_lim)
                    recl.append(im_mu)

                    l = []
                    for size_sep in size_sepl:
                        n = nf['nf'][:-4] + 'synth.icl.wavsizesepmask_%03d_%03d.fits'%(lvl_sep, size_sep)
                        nfp = os.path.join(path_wavelets, n)
                        hdu = fits.open(nfp)
                        im_MJy = hdu[1].data
                        #im_MJy = rebin(im_MJy, 2, 2)
                        im_mu = MJy_to_mu(im_MJy, arcsecar_sr, mu_lim)
                        l.append(im_mu)
                    recmsl.append(l)

        n_grid = 10
        n_cols = 5
        n_rows = 7
        fig = plt.figure(1, figsize = (7., 9.5))
        grid = GridSpec(nrows = n_rows * n_grid, ncols = n_cols * n_grid + 1, figure=1, left=0.05, bottom=0.01, right=0.9, top=0.92, wspace=0.01, hspace=0.01, width_ratios=None, height_ratios=None)

        fig.suptitle('ICL synthesis maps %s'%filt)
        cmap = cmasher.seasons
        bounds = [ 0., 24.5, 25.5, 26.5, 27.5, 28.5, 29.5, 40., ]
        norm = colors.BoundaryNorm(bounds, cmap.N, clip = True)

        for i, lvl_sep in enumerate(lvl_sepl):

            #poim = ax[0][i].imshow( recl[i], norm = norm, cmap = cmap, origin = 'lower' )
            ax = fig.add_subplot(grid[:n_grid,(i*n_grid):(i*n_grid+n_grid)])
            poim = ax.imshow( recl[i], norm = ImageNormalize( recl[0], interval = MinMaxInterval(), \
                                    stretch = LinearStretch()), cmap = cmap, origin = 'lower')
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            ax.set_title('$w_s$ = %d'%lvl_sep)
            if i == 0:ax.set_ylabel('WS')

            #poim = ax[1][i].imshow( recml[i], norm = norm, cmap = cmap, origin = 'lower' )
            ax = fig.add_subplot(grid[n_grid:2 * n_grid,(i*n_grid):(i*n_grid+n_grid)])
            poim = ax.imshow( recml[i], norm = ImageNormalize( recl[0], interval = MinMaxInterval(), \
                                    stretch = LinearStretch()), cmap = cmap, origin = 'lower')
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            if i == 0:ax.set_ylabel('WS + SF')

            for j, size_sep in enumerate(size_sepl):

                #poim = ax[2][i].imshow( recmsl[i], norm = norm, cmap = cmap, origin = 'lower' )
                ax = fig.add_subplot(grid[ (2 + j) * n_grid: (3 + j) * n_grid,(i*n_grid):(i*n_grid+n_grid)])
                poim = ax.imshow( recmsl[i][j], norm = ImageNormalize( recl[0], interval = MinMaxInterval(), \
                                        stretch = LinearStretch()), cmap = cmap, origin = 'lower')
                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])
                if i == 0:ax.set_ylabel('WS + SF \n+ SS %03d kpc'%size_sep)

        for i in range(n_rows):
            cax = fig.add_subplot(grid[(i*n_grid):(i*n_grid+n_grid), n_cols * n_grid])
            caxre = fig.colorbar( poim, cax = cax, \
                                        orientation = 'vertical', \
                                        format = '%2.1f',\
                                        pad = 0.05,\
                                        ticklocation = 'right',\
                                        label = '$\mu$ [mag.arcsec$^{-2}$]' )

        plt.tight_layout()
        #plt.subplots_adjust( left=0.1, bottom=0.05, right=0.9, top=0.9, wspace=0.05, hspace=0.05)
        plt.savefig(os.path.join(path_plots, 'plot_array_icl_maps_%s_mulim_%2d_out20.pdf'%(filt, mu_lim)), format = 'pdf', dpi = 500)
        # Clear the current axes.
        plt.cla()
        # Clear the current figure.
        plt.clf()
        # Closes all the figure windows.
        plt.close('all')
        plt.close(fig)
        #plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def plot_array_bcgicl_maps_all_filters():

    # Paths, lists & variables
    path_data = '/home/ellien/JWST/data/'
    path_scripts = '/home/ellien/JWST/JWST_scripts'
    path_wavelets = '/home/ellien/JWST/wavelets/out15/'
    path_plots = '/home/ellien/JWST/plots'

    nfl = [ {'nf':'jw02736001001_f356w_bkg_rot_crop_input.fits', 'filt':'f356w', 'chan':'long', 'pix_scale':0.063, 'pixar_sr':9.31E-14, 'n_levels':10, 'lvl_sep_max':999 }, \
            {'nf':'jw02736001001_f444w_bkg_rot_crop_input.fits', 'filt':'f444w', 'chan':'long', 'pix_scale':0.063, 'pixar_sr':9.31E-14, 'n_levels':10, 'lvl_sep_max':999 }, \
            {'nf':'jw02736001001_f277w_bkg_rot_crop_input.fits', 'filt':'f277w', 'chan':'long', 'pix_scale':0.063, 'pixar_sr':9.31E-14, 'n_levels':10, 'lvl_sep_max':999 }, \
            {'nf':'jw02736001001_f090w_bkg_rot_crop_warp_nobkg2.fits', 'filt':'f090w', 'chan':'short', 'pix_scale':0.031, 'pixar_sr':2.29E-14, 'n_levels':10, 'lvl_sep_max':999 }, \
            {'nf':'jw02736001001_f150w_bkg_rot_crop_warp_nobkg2.fits', 'filt':'f150w', 'chan':'short', 'pix_scale':0.031, 'pixar_sr':2.31E-14, 'n_levels':10, 'lvl_sep_max':999 }, \
            {'nf':'jw02736001001_f200w_bkg_rot_crop_warp_nobkg2.fits', 'filt':'f200w', 'chan':'short', 'pix_scale':0.031, 'pixar_sr':2.29E-14, 'n_levels':10, 'lvl_sep_max':999 } ]
            #{'nf':'jw02736001001_f090w_bkg_rot_crop_input.fits', 'filt':'f090w', 'chan':'short', 'pix_scale':0.031, 'pixar_sr':2.29E-14, 'n_levels':11, 'lvl_sep_max':8 }, \
            #{'nf':'jw02736001001_f150w_bkg_rot_crop_input.fits', 'filt':'f150w', 'chan':'short', 'pix_scale':0.031, 'pixar_sr':2.31E-14, 'n_levels':11, 'lvl_sep_max':8 }, \
            #{'nf':'jw02736001001_f200w_bkg_rot_crop_input.fits', 'filt':'f200w', 'chan':'short', 'pix_scale':0.031, 'pixar_sr':2.29E-14, 'n_levels':11, 'lvl_sep_max':8 } ]

    filterl = [ 'f090w', 'f150w', 'f200w', 'f277w', 'f356w', 'f444w' ]
    n_rebin = 2
    mu_lim = 30
    lvl_sepl = [ 3, 4, 5, 6, 7 ]
    size_sepl = [ 60, 80, 100, 140, 200 ] # kpc
    sls = sls_cmap()
    sls_r = ListedColormap(sls.colors[::-1])

    for filt in filterl:
        print('\n%s - making ICL+BCG synthesis map grid - mu_lim = %d mag/arcsec2'%(filt, mu_lim))

        for nf in nfl:

            if filt == nf['filt']:

                # Photometry
                if nf['chan'] == 'short':
                    arcsecar_sr = nf['pixar_sr'] / (n_rebin * nf['pix_scale']**2) # short channel images are rebinned
                else:
                    arcsecar_sr = nf['pixar_sr'] / (nf['pix_scale']**2)

                # Read all files
                recml = []
                recl = []
                recmsl = []

                for lvl_sep in lvl_sepl:

                    n = nf['nf'][:-4] + 'synth.icl.bcgwavsepmask_%03d.fits'%(lvl_sep)
                    nfp = os.path.join(path_wavelets, n)
                    im_MJy = fits.getdata(nfp)
                    im_MJy = rebin(im_MJy, 2, 2)
                    im_mu = MJy_to_mu(im_MJy, arcsecar_sr, mu_lim)
                    recml.append(im_mu)

                    n = nf['nf'][:-4] + 'synth.icl.wavsep_%03d.fits'%(lvl_sep)
                    nfp = os.path.join(path_wavelets, n)
                    im_MJy = fits.getdata(nfp)
                    im_MJy = rebin(im_MJy, 2, 2)
                    im_mu = MJy_to_mu(im_MJy, arcsecar_sr, mu_lim)
                    recl.append(im_mu)

                    l = []
                    for size_sep in size_sepl:
                        n = nf['nf'][:-4] + 'synth.icl.bcgwavsizesepmask_%03d_%03d.fits'%(lvl_sep, size_sep)
                        nfp = os.path.join(path_wavelets, n)
                        im_MJy = fits.getdata(nfp)
                        im_MJy = rebin(im_MJy, 2, 2)
                        im_mu = MJy_to_mu(im_MJy, arcsecar_sr, mu_lim)
                        l.append(im_mu)
                    recmsl.append(l)

        n_grid = 10
        n_cols = 5
        n_rows = 7
        fig = plt.figure(1, figsize = (7., 9.5))
        grid = GridSpec(nrows = n_rows * n_grid, ncols = n_cols * n_grid + 1, figure=1, left=0.05, bottom=0.01, right=0.9, top=0.92, wspace=0.01, hspace=0.01, width_ratios=None, height_ratios=None)

        fig.suptitle('ICL+BCG synthesis maps %s'%filt)
        cmap = cmasher.seasons
        bounds = [ 0., 24.5, 25.5, 26.5, 27.5, 28.5, 29.5, 40., ]
        #norm = colors.BoundaryNorm(bounds, cmap.N, clip = True)

        for i, lvl_sep in enumerate(lvl_sepl):

            #poim = ax[0][i].imshow( recl[i], norm = norm, cmap = cmap, origin = 'lower' )
            ax = fig.add_subplot(grid[:n_grid,(i*n_grid):(i*n_grid+n_grid)])
            poim = ax.imshow( recl[i], norm = ImageNormalize( recl[0], interval = MinMaxInterval(), \
                                    stretch = LinearStretch()), cmap = cmap, origin = 'lower')
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            ax.set_title('$w_s$ = %d'%lvl_sep)
            if i == 0:ax.set_ylabel('WS')

            #poim = ax[1][i].imshow( recml[i], norm = norm, cmap = cmap, origin = 'lower' )
            ax = fig.add_subplot(grid[n_grid:2 * n_grid,(i*n_grid):(i*n_grid+n_grid)])
            poim = ax.imshow( recml[i], norm = ImageNormalize( recl[0], interval = MinMaxInterval(), \
                                    stretch = LinearStretch()), cmap = cmap, origin = 'lower')
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            if i == 0:ax.set_ylabel('WS + SF')

            for j, size_sep in enumerate(size_sepl):

                #poim = ax[2][i].imshow( recmsl[i], norm = norm, cmap = cmap, origin = 'lower' )
                ax = fig.add_subplot(grid[ (2 + j) * n_grid: (3 + j) * n_grid,(i*n_grid):(i*n_grid+n_grid)])
                poim = ax.imshow( recmsl[i][j], norm = ImageNormalize( recl[0], interval = MinMaxInterval(), \
                                        stretch = LinearStretch()), cmap = cmap, origin = 'lower')
                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])
                if i == 0:ax.set_ylabel('WS + SF \n+ SS %03d kpc'%size_sep)

        for i in range(n_rows):
            cax = fig.add_subplot(grid[(i*n_grid):(i*n_grid+n_grid), n_cols * n_grid])
            caxre = fig.colorbar( poim, cax = cax, \
                                        orientation = 'vertical', \
                                        format = '%2.1f',\
                                        pad = 0.05,\
                                        ticklocation = 'right',\
                                        label = '$\mu$ [mag.arcsec$^{-2}$]' )

        plt.tight_layout()
        #plt.subplots_adjust( left=0.1, bottom=0.05, right=0.9, top=0.9, wspace=0.05, hspace=0.05)
        plt.savefig(os.path.join(path_plots, 'plot_array_bcgicl_maps_%s_mulim_%2d.pdf'%(filt, mu_lim)), format = 'pdf', dpi = 500)
        # Clear the current axes.
        plt.cla()
        # Clear the current figure.
        plt.clf()
        # Closes all the figure windows.
        plt.close('all')
        plt.close(fig)
        #plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def plot_rgb_mask_wcs():

    # Paths, lists & variables
    path_data = '/home/ellien/JWST/data/'
    path_wavelets = '/home/ellien/JWST/wavelets/out13/'
    path_plots = '/home/ellien/JWST/plots'

    # RGB image
    image = mpimg.imread(os.path.join(path_plots, 'RGB_f277w_f356w_f444w.png'))
    image = np.array(image)
    image = np.flip(image, axis = 0)

    # WCS from one of the filters
    nf = 'jw02736001001_f444w_bkg_rot_crop_input.fits'
    nfp = os.path.join(path_data, nf)
    hdu = fits.open(os.path.join(path_data, nf))[0]
    wcs = WCS(hdu.header)

    # Star region files
    nf = 'mahler_noirot_merged_member_gal_ra_dec_pix_long.reg'
    r = pyr.open(os.path.join(path_data, nf))
    patch_list, artist_list = r.get_mpl_patches_texts()

    # Member galaxies region files
    nf = 'star_flags_polygon_pix_long.reg'
    r2 = pyr.open(os.path.join(path_data, nf))
    patch_list2, artist_list2 = r2.get_mpl_patches_texts()

    # Circles
    nf = 'circles_128_200_400kpc_pix.reg'
    r3 = pyr.open(os.path.join(path_data, nf))
    patch_list3, artist_list3 = r3.get_mpl_patches_texts()

    plt.figure(figsize=(9,9))
    plt.subplot(projection = wcs)
    plt.gca().tick_params(axis='x', labelsize=15)
    plt.gca().tick_params(axis='y', labelsize=15)
    plt.imshow(image)
    plt.grid(color='white', ls='dashed', alpha = 0.8)
    plt.xlabel('R.a.', fontsize = 15)
    plt.ylabel('Dec.', fontsize = 15)

    for p in patch_list:
        plt.gca().add_patch(p)
    for a in artist_list:
        plt.gca().add_artist(a)

    for p in patch_list2:
        plt.gca().add_patch(p)
    for a in artist_list2:
        plt.gca().add_artist(a)

    for p in patch_list3:
        plt.gca().add_patch(p)
    for a in artist_list3:
        plt.gca().add_artist(a)

    plt.tight_layout()
    plt.subplots_adjust( left=0.2, bottom=0.03, right=0.99, top=0.99, wspace=0, hspace=0.3)
    plt.savefig(os.path.join(path_plots, 'RGB_masks_wcs.pdf'), format = 'pdf', dpi = 500)
    plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def plot_recim_rgb_icl():

    # Paths, lists & variables
    path_data = '/home/aellien/JWST/data/'
    path_scripts = '/home/aellien/JWST/JWST_scripts'
    path_wavelets = '/home/aellien/JWST/wavelets/out20/'
    path_plots = '/home/aellien/JWST/plots'

    filterl = [ 'f277w', 'f356w', 'f444w']
    #filterl = [ 'f090w', 'f150w', 'f200w' ]
    schl = ['WS+SF+SS']
    size_sepl = [100, 200, 300]
    lvl_sepl = [ 5, 6 ]
    colorl = [ 'dodgerblue', 'mediumaquamarine', 'paleturquoise' , 'white' ]
    cmap = cmasher.seasons
    #cmap = 'gray_r'
    n_grid = 10
    n_cols = 3
    n_rows = 5
    std = 'WS+SF', 
    n_bin = 1
    fig = plt.figure(1, figsize = (5., 8.3))
    grid = GridSpec(nrows = n_rows * n_grid, ncols = n_cols * n_grid + 1, figure = 1, left = 0.1, bottom = 0.01, right = 0.99, top = 0.9, wspace = 0.1, hspace = 0.1, width_ratios = None, height_ratios = None)

    # Member galaxies region files
    nf = 'mahler_noirot_merged_member_gal_ra_dec_pix_long.reg'
    r = pyr.open(os.path.join(path_data, nf))
    patch_list, artist_list = r.get_mpl_patches_texts()

    tot_cl_l = []
    tot_cl_mu_l = []

    for i, filt in enumerate(filterl):

        nf = 'jw02736001001_%s_bkg_rot_crop_input.fits' %filt
        nfp = os.path.join(path_data, nf)
        oim = fits.getdata(nfp)
        oim = gaussian_filter(oim, std)

        oim[ np.where(oim == 0.) ] = 1E-10
        oim_mu = - 2.5 * np.log10( oim / 4.25 * 1E-4 ) + 8.906
        oim_mu[oim_mu > 31.1] = 31.1 # SB limit from Montes 2022

        nf = 'jw02736001001_%s_bkg_rot_crop_input.synth.bcgwavsizesepmask_005_080.fits'%(filt)
        #nf = 'jw02736001001_%s_bkg_rot_crop_warp_nobkg2.synth.icl.bcgwavsizesepmask_005_080.fits'%(filt)

        nfp = os.path.join(path_wavelets, nf)
        hdu = fits.open(nfp)
        iclbcg = hdu[1].data
        #iclbcg = gaussian_filter(iclbcg, std)
        #iclbcg = rebin(iclbcg, n_bin, n_bin)
        iclbcg[ np.where(iclbcg == 0.) ] = 1E-10
        iclbcg_mu = - 2.5 * np.log10( iclbcg / 4.25 * 1E-4 ) + 8.906
        #iclbcg_mu[iclbcg_mu > 28.] = 28. # SB limit from Montes 2022
        tot_cl_l.append(iclbcg)

        nf = 'jw02736001001_%s_bkg_rot_crop_input.synth.bcgwavsizesepmask_005_080.fits'%(filt)
        #nf = 'jw02736001001_%s_bkg_rot_crop_warp_nobkg2.synth.gal.bcgwavsizesepmask_005_200.fits'%(filt)

        nfp = os.path.join(path_wavelets, nf)
        hdu = fits.open(nfp)
        sat = hdu[2].data
        #sat = rebin(sat, n_bin, n_bin)
        sat[ np.where(iclbcg == 0.) ] = 1E-10
        sat_mu = - 2.5 * np.log10( sat / 4.25 * 1E-4 ) + 8.906
        sat_mu[sat_mu > 31.1] = 31.1 # SB limit from Montes 2022

        tot_cl = iclbcg + sat
        hduo = fits.PrimaryHDU( tot_cl )
        hduo.writeto(os.path.join( path_plots, '%s_tot_cl.fits'%filt ), overwrite = True)
        tot_cl[ np.where(iclbcg == 0.) ] = 1E-10
        tot_cl_mu = - 2.5 * np.log10( tot_cl / 4.25 * 1E-4) + 8.906
        tot_cl_mu[tot_cl_mu > 28.] = 28. # SB limit from Montes 2022
        #tot_cl_l.append(tot_cl)
        tot_cl_mu_l.append(tot_cl_mu)

        #ax = fig.add_subplot(grid[0, i])
        #poim = ax.imshow( sat_mu, norm = ImageNormalize( oim_mu, interval = ZScaleInterval(), \
        #                        stretch = LinearStretch()), cmap = cmap, origin = 'lower')
        #ax.get_xaxis().set_ticks([])
        #ax.get_yaxis().set_ticks([])

        ax = fig.add_subplot(grid[0:n_grid, i * n_grid: (i+1)*n_grid])
        poim = ax.imshow( iclbcg_mu, norm = ImageNormalize( oim_mu, interval = ZScaleInterval(), \
                                stretch = LinearStretch()), cmap = cmap, origin = 'lower')
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        if i == 0:
            ax.set_ylabel('ICL+BCG')


        ax = fig.add_subplot(grid[n_grid: 2 * n_grid, i * n_grid: (i+1)*n_grid])
        poim = ax.imshow( tot_cl_mu, norm = ImageNormalize( oim_mu, interval = ZScaleInterval(), \
                                stretch = LinearStretch()), cmap = cmap, origin = 'lower')
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        if i == 0:
            ax.set_ylabel('ICL+BCG+satellites')

        cax = fig.add_subplot(grid[0, (i*n_grid):(i*n_grid+n_grid)])
        caxre = fig.colorbar( poim, cax = cax, \
                                    orientation = 'horizontal', \
                                    format = '%2.0f',\
                                    pad = 0.05,\
                                    ticklocation = 'top',\
                                    label = '$\mu$ [mag.arcsec$^{-2}$]' )
        caxre.ax.tick_params( labelsize = 10)
        cax.set_title('%s'%filt)

    #fig = plt.figure()
    rgb = make_lupton_rgb(rebin(tot_cl_l[0], n_bin, n_bin), rebin(tot_cl_l[1], n_bin, n_bin), rebin(tot_cl_l[2], n_bin, n_bin), Q = 10, stretch=0.1)
    ax = fig.add_subplot(grid[2 * n_grid:, 0:-1])
    ax.set_ylabel('Composite ICL+BCG map')
    poim = ax.imshow( rgb, origin = 'lower')

    #for p in patch_list:
    #    plt.gca().add_patch(p)
    #for a in artist_list:
    #    plt.gca().add_artist(a)

    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    plt.tight_layout()
    plt.subplots_adjust( left = 0.01, right = 0.99, hspace = 0.1, wspace = 0.1 )

    plt.savefig(os.path.join(path_plots, 'RGB_iclbcg_out20.pdf'), format = 'pdf')


    plt.figure()
    plt.imshow(tot_cl_mu_l[1] - tot_cl_mu_l[2], vmax = 0.5, vmin = -2, cmap = 'hot', origin = 'lower')
    plt.colorbar()

    plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def plot_recim_rgb_all_filters():

    # Paths, lists & variables
    path_data = '/home/aellien/JWST/data/'
    path_scripts = '/home/aellien/JWST/JWST_scripts'
    path_wavelets = '/home/aellien/JWST/wavelets/out15/'
    path_plots = '/home/aellien/JWST/plots'

    nfl = [ {'nf':'jw02736001001_f356w_bkg_rot_crop_input.fits', 'filt':'f356w', 'chan':'long', 'pix_scale':0.063, 'pixar_sr':9.31E-14, 'n_levels':10, 'lvl_sep_max':999 }, \
            {'nf':'jw02736001001_f444w_bkg_rot_crop_input.fits', 'filt':'f444w', 'chan':'long', 'pix_scale':0.063, 'pixar_sr':9.31E-14, 'n_levels':10, 'lvl_sep_max':999 }, \
            {'nf':'jw02736001001_f277w_bkg_rot_crop_input.fits', 'filt':'f277w', 'chan':'long', 'pix_scale':0.063, 'pixar_sr':9.31E-14, 'n_levels':10, 'lvl_sep_max':999 }, \
            {'nf':'jw02736001001_f090w_bkg_rot_crop_warp_nobkg2.fits', 'filt':'f090w', 'chan':'short', 'pix_scale':0.063, 'pixar_sr':9.31E-14, 'n_levels':10, 'lvl_sep_max':8 }, \
            {'nf':'jw02736001001_f150w_bkg_rot_crop_warp_nobkg2.fits', 'filt':'f150w', 'chan':'short', 'pix_scale':0.063, 'pixar_sr':9.31E-14, 'n_levels':10, 'lvl_sep_max':8 }, \
            {'nf':'jw02736001001_f200w_bkg_rot_crop_warp_nobkg2.fits', 'filt':'f200w', 'chan':'short', 'pix_scale':0.063, 'pixar_sr':9.31E-14, 'n_levels':10, 'lvl_sep_max':8 } ]

    paired_filterl = [ ['f090w', 'f150w'], ['f200w', 'f277w'], ['f356w', 'f444w' ] ] # paired for RGB
    binning_factor = 4151 / 2045 # size short / size long
    combiml = []
    cmap = cmasher.seasons
    std = 5

    for i, (filt1, filt2) in enumerate(paired_filterl):

        combim = np.zeros((2045, 2045))
        combiml.append(combim)
        for nf in nfl:
            if (nf['filt'] == filt1) or (nf['filt'] == filt2):
                if nf['chan'] == 'long':
                    nfp = os.path.join(path_wavelets, 'jw02736001001_%s_bkg_rot_crop_input.synth.icl.bcgwavsizesepmask_005_080.fits'%(nf['filt']))
                    icl = fits.getdata(nfp)
                    icl = gaussian_filter(icl, std)

                    nfp = os.path.join(path_wavelets, 'jw02736001001_%s_bkg_rot_crop_input.synth.gal.bcgwavsizesepmask_005_080.fits'%(nf['filt']))
                    gal = fits.getdata(nfp)
                    #gal = gaussian_filter(gal, std)

                if nf['chan'] == 'short':
                    nfp = os.path.join(path_wavelets, 'jw02736001001_%s_bkg_rot_crop_warp_nobkg2.synth.icl.bcgwavsizesepmask_005_080.fits'%(nf['filt']))
                    icl = fits.getdata(nfp)
                    #icl = zoom( icl, 1 / binning_factor, order = 5 ) # resample to same size as long channel images
                    icl = gaussian_filter(icl, std)

                    nfp = os.path.join(path_wavelets, 'jw02736001001_%s_bkg_rot_crop_warp_nobkg2.synth.gal.bcgwavsizesepmask_005_080.fits'%(nf['filt']))
                    gal = fits.getdata(nfp)
                    #gal = zoom( gal, 1 / binning_factor, order = 5 ) # resample to same size as long channel images
                    #gal = gaussian_filter(gal, std)

                combiml[i] += icl
                combiml[i] += gal

    combiml = np.array(combiml)
    print(combiml.shape)

    fig, ax = plt.subplots(3, 1)
    for i in range(3):
        ax[i].imshow( combiml[i], norm = ImageNormalize( combiml[i], interval = MinMaxInterval(), \
                                stretch = LogStretch()), cmap = cmap, origin = 'lower')

    plt.show()

    fig = plt.figure()
    rgb = make_lupton_rgb(image_r = combiml[2], image_g = combiml[1], image_b = combiml[0], Q = 15, stretch=0.15)
    plt.imshow( rgb, origin = 'lower')
    plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def plot_scattered_light_maps():

    # Paths, lists & variables
    path_data = '/home/aellien/JWST/data/'
    path_scripts = '/home/aellien/JWST/JWST_scripts'
    path_plots = '/home/aellien/JWST/plots'

    filterl = [ 'f090w', 'f150w', 'f200w' ]
    binning_factor = 4151 / 2045 # size short / size long
    cmap = cmasher.seasons
    std = 5
    binf = 2

    oiml = []
    nobkgl = []
    combbkgl = []

    for i, filt in enumerate(filterl):

        nfp = os.path.join(path_data, 'jw02736001001_%s_bkg_rot_crop_warp.fits'%filt)
        hdu = fits.open(nfp)
        oim = hdu[1].data
        oiml.append(oim)
        nfp = os.path.join(path_data, 'jw02736001001_%s_bkg_rot_crop_warp_nobkg2.fits'%filt)
        nobkg = fits.getdata(nfp)[:2046, :2046]
        nobkgl.append(nobkg)
        nfp = os.path.join(path_data, 'jw02736001001_%s_bkg_rot_crop_input_nobkg2_warp.fits'%filt)
        hdu = fits.open(nfp)
        combbkg = hdu[1].data[:2046, :2046]
        combbkgl.append(combbkg)
        #print(- 2.5 * np.log10(np.nanmean(combbkg) / 4.25 * 1E-4 ) + 8.906, - 2.5 * np.log10(np.nanmax(combbkg) / 4.25 * 1E-4 ) + 8.906)

    fig, ax = plt.subplots(3, 3, figsize = (11,12))
    sls = sls_cmap()
    cmap = cmasher.seasons

    for i, filt in enumerate(filterl):

        oim = rebin(oiml[i], binf, binf, 'SUM')
        poim = ax[i][0].imshow( oim, norm = ImageNormalize( oim, interval = ZScaleInterval(), \
                                stretch = LinearStretch()), cmap = cmap, origin = 'lower')
        ax[i][0].get_xaxis().set_ticks([])
        ax[i][0].get_yaxis().set_ticks([])
        ax[i][0].set_ylabel('%s'%filterl[i], fontsize = 15)
        divider = make_axes_locatable(ax[i][0])
        cax = divider.append_axes("bottom", size="5%", pad=0.05)
        caxre = fig.colorbar( poim, cax = cax, \
                                    cmap = sls, \
                                    orientation = 'horizontal', \
                                    format = '%2.1f',\
                                    pad = 0,\
                                    shrink = 1.0,\
                                    ticklocation = 'bottom' )

        caxre.ax.tick_params(labelsize = 15)
        if i == 2:
            cax.set_xlabel('MJy/sr', fontsize = 15)

        combbkg = rebin(combbkgl[i], binf, binf, 'SUM')
        poim = ax[i][1].imshow( combbkg, norm = ImageNormalize( oim, interval = ZScaleInterval(), \
                                stretch = LinearStretch()), cmap = cmap, origin = 'lower')
        ax[i][1].get_xaxis().set_ticks([])
        ax[i][1].get_yaxis().set_ticks([])
        divider = make_axes_locatable(ax[i][1])
        cax = divider.append_axes("bottom", size="5%", pad=0.05)
        caxre = fig.colorbar( poim, cax = cax, \
                                    orientation = 'horizontal', \
                                    format = '%2.1f',\
                                    pad = 0,\
                                    shrink = 1.0,\
                                    ticklocation = 'bottom' )
        caxre.ax.tick_params(labelsize=15)
        if i == 2:
            cax.set_xlabel('MJy/sr', fontsize = 15)

        nobkg = rebin(nobkgl[i], binf, binf, 'SUM')
        poim = ax[i][2].imshow( nobkg, norm = ImageNormalize( oim, interval = ZScaleInterval(), \
                        stretch = LinearStretch()), cmap = cmap, origin = 'lower')
        ax[i][2].get_xaxis().set_ticks([])
        ax[i][2].get_yaxis().set_ticks([])
        divider = make_axes_locatable(ax[i][2])
        cax = divider.append_axes("bottom", size="5%", pad=0.05)
        caxre = fig.colorbar( poim, cax = cax, \
                                    orientation = 'horizontal', \
                                    format = '%2.1f',\
                                    pad = 0,\
                                    shrink = 1.0,\
                                    ticklocation = 'bottom' )
        caxre.ax.tick_params(labelsize=15)
        if i == 2:
            cax.set_xlabel('MJy/sr', fontsize = 15)

    ax[0][0].set_title('Original image', fontsize = 15)
    ax[0][1].set_title('Prebin correction', fontsize = 15)
    ax[0][2].set_title('Postbin correction', fontsize = 15)

    plt.tight_layout()
    #plt.subplots_adjust( left=0.05, bottom=0.05, right=0.98, top=0.98, wspace=0.03, hspace=0.1)
    plt.savefig(os.path.join(path_plots, 'plot_scattered_light_short.pdf'), format = 'pdf', dpi = 500)
    plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def plot_oim_rgb_all_filters():

    # Paths, lists & variables
    path_data = '/home/aellien/JWST/data/'
    path_scripts = '/home/aellien/JWST/JWST_scripts'
    path_wavelets = '/home/aellien/JWST/wavelets/out13/'
    path_plots = '/home/aellien/JWST/plots'

    nfdl = [ {'nf':'jw02736001001_f356w_bkg_rot_crop_input.fits', 'filt':'f356w', 'chan':'long', 'pix_scale':0.063, 'pixar_sr':9.31E-14, 'n_levels':10, 'lvl_sep_max':999 }, \
            {'nf':'jw02736001001_f444w_bkg_rot_crop_input.fits', 'filt':'f444w', 'chan':'long', 'pix_scale':0.063, 'pixar_sr':9.31E-14, 'n_levels':10, 'lvl_sep_max':999 }, \
            {'nf':'jw02736001001_f277w_bkg_rot_crop_input.fits', 'filt':'f277w', 'chan':'long', 'pix_scale':0.063, 'pixar_sr':9.31E-14, 'n_levels':10, 'lvl_sep_max':999 }, \
            {'nf':'jw02736001001_f090w_bkg_rot_crop_warp_nobkg1_det_nosky_input.fits', 'filt':'f090w', 'chan':'short', 'pix_scale':0.063, 'pixar_sr':9.31E-14, 'n_levels':10, 'lvl_sep_max':8 }, \
            {'nf':'jw02736001001_f150w_bkg_rot_crop_warp_nobkg1_det_nosky_input.fits', 'filt':'f150w', 'chan':'short', 'pix_scale':0.063, 'pixar_sr':9.31E-14, 'n_levels':10, 'lvl_sep_max':8 }, \
            {'nf':'jw02736001001_f200w_bkg_rot_crop_warp_nobkg1_det_nosky_input.fits', 'filt':'f200w', 'chan':'short', 'pix_scale':0.063, 'pixar_sr':9.31E-14, 'n_levels':10, 'lvl_sep_max':8 } ]

    #paired_filterl = [ ['f090w', 'f150w'], ['f200w', 'f277w'], ['f356w', 'f444w' ] ] # paired for RGB
    paired_filterl = [ ['f090w', 'f090w'], ['f277w', 'f277w'], ['f356w', 'f356w' ] ] # paired for RGB
    combiml = []
    cmap = cmasher.seasons
    std = 5
    xs, ys = (2045, 2045)

    for i, (filt1, filt2) in enumerate(paired_filterl):

        combim = np.zeros((xs, ys))
        combiml.append(combim)
        for nfd in nfdl:

            if (nfd['filt'] == filt1) or (nfd['filt'] == filt2):

                nfp = os.path.join(path_data, nfd['nf'])
                oim = fits.getdata(nfp)
                #oim = gaussian_filter(icl, std)
                #oim[oim < 0.] = 0.
                combiml[i] += oim[:xs, :ys]

    combiml = np.array(combiml)
    print(combiml.shape)

    fig, ax = plt.subplots(1, 3)
    for i in range(3):
        ax[i].imshow( combiml[i], cmap = cmap, origin = 'lower', norm = ImageNormalize( combiml[i], interval = ZScaleInterval(), \
                                stretch = LinearStretch()))

    plt.show()

    fig = plt.figure()
    rgb = make_lupton_rgb(image_r = combiml[2], image_g = combiml[1], image_b = combiml[0], Q = 15, stretch=0.15)
    plt.imshow( rgb, origin = 'lower')
    plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def plot_example_recim_all_filter_maps():

    # Paths, lists & variables
    path_data = '/home/aellien/JWST/data/'
    path_scripts = '/home/aellien/JWST/JWST_scripts'
    path_plots = '/home/aellien/JWST/plots'
    path_wavelets = '/home/aellien/JWST/wavelets/out21'

    nfdl = [ {'nf':'jw02736001001_f356w_bkg_rot_crop_input.fits', 'filt':'f356w', 'chan':'long', 'pix_scale':0.063, 'pixar_sr':9.31E-14, 'phot_corr':0.163, 'n_levels':10 }, \
            {'nf':'jw02736001001_f444w_bkg_rot_crop_input.fits', 'filt':'f444w', 'chan':'long', 'pix_scale':0.063, 'pixar_sr':9.31E-14, 'phot_corr':0.162, 'n_levels':10 }, \
            {'nf':'jw02736001001_f277w_bkg_rot_crop_input.fits', 'filt':'f277w', 'chan':'long', 'pix_scale':0.063, 'pixar_sr':9.31E-14, 'phot_corr':0.223, 'n_levels':10 }, \
            {'nf':'jw02736001001_f090w_bkg_rot_crop_warp_nobkg2.fits', 'filt':'f090w', 'chan':'short', 'pix_scale':0.031, 'phot_corr':-0.174, 'pixar_sr':2.29E-14, 'n_levels':10 }, \
            {'nf':'jw02736001001_f150w_bkg_rot_crop_warp_nobkg2.fits', 'filt':'f150w', 'chan':'short', 'pix_scale':0.031, 'phot_corr':-0.047, 'pixar_sr':2.31E-14, 'n_levels':10 }, \
            {'nf':'jw02736001001_f200w_bkg_rot_crop_warp_nobkg2.fits', 'filt':'f200w', 'chan':'short', 'pix_scale':0.031, 'phot_corr':-0.114, 'pixar_sr':2.29E-14, 'n_levels':10 } ]


    filterl = [ 'f090w', 'f150w', 'f200w', 'f277w', 'f356w', 'f444w' ]
    binning_factor = 4151 / 2045 # size short / size long
    cmap = cmasher.seasons
    binf = 2

    oiml = []
    reciml = []

    for i, filt in enumerate(filterl):
        for nfd in nfdl:
            if nfd['filt'] == filt:
                
                nfp = os.path.join(path_data, nfd['nf'])
                hdu = fits.open(nfp)
                oim = hdu[0].data
                oiml.append(oim)
                
                nfp = os.path.join(path_wavelets, nfd['nf'][:-5] + '_synth.bcgwavsizesepmask_005_080.fits')
                print(nfp)
                hdu = fits.open(nfp)
                recim = hdu[1].data[:2046, :2046]

                #r = pyr.open(os.path.join(path_data, "mask_display_bad_atoms_out21.reg")).as_imagecoord(hdu[1].header)
                #m = r.get_mask(hdu = hdu[1])[:2046, :2046]

                #recim[m] = np.min(recim)
                recim = rebin(recim, 8, 8)
                #recim = gaussian_filter(recim, sigma = 2)
                
                reciml.append(recim)

    fig, ax = plt.subplots(2, 3, figsize = (12,8))
    cmap = 'gray_r'
    #print(np.shape(reciml))
    k = 0
    l = -1
    for i, filt in enumerate(filterl):

        l = l + 1
        if l > 2:
            l = 0
            k += 1

        norm = ImageNormalize( rebin(oiml[i], 8, 8), vmin = 0, interval = ZScaleInterval(), stretch = LinearStretch())
        poim = ax[k][l].imshow( reciml[i], norm = norm, cmap = cmap, origin = 'lower')
        ax[k][l].get_xaxis().set_ticks([])
        ax[k][l].get_yaxis().set_ticks([])
        divider = make_axes_locatable(ax[k][l])
        cax = divider.append_axes("bottom", size="5%", pad=0.05)
        caxre = fig.colorbar( poim, cax = cax, \
                                    cmap = cmap, \
                                    orientation = 'horizontal', \
                                    format = '%1.2f',\
                                    pad = 0,\
                                    shrink = 1.0,\
                                    ticklocation = 'bottom' )

        caxre.ax.tick_params(labelsize = 15)
        if k == 1:
            cax.set_xlabel('MJy/sr', fontsize = 15)


        ax[k][l].set_title('%s'%filt, fontsize = 15)

    plt.tight_layout()
    #plt.subplots_adjust( left=0.05, bottom=0.05, right=0.98, top=0.98, wspace=0.03, hspace=0.1)
    plt.savefig(os.path.join(path_plots, 'plot_example_recim_all_filter_maps_out21.pdf'), format = 'pdf', dpi = 500)
    plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def plot_test():

    # Paths, lists & variables
    path_data = '/home/ellien/JWST/data/'
    path_scripts = '/home/ellien/JWST/JWST_scripts'
    path_plots = '/home/ellien/JWST/plots'
    path_wavelets = '/home/ellien/JWST/wavelets/out15'

    nfdl = [ {'nf':'jw02736001001_f356w_bkg_rot_crop_input.fits', 'filt':'f356w', 'chan':'long', 'pix_scale':0.063, 'pixar_sr':9.31E-14, 'phot_corr':0.163, 'n_levels':10 }, \
            {'nf':'jw02736001001_f444w_bkg_rot_crop_input.fits', 'filt':'f444w', 'chan':'long', 'pix_scale':0.063, 'pixar_sr':9.31E-14, 'phot_corr':0.162, 'n_levels':10 }, \
            {'nf':'jw02736001001_f277w_bkg_rot_crop_input.fits', 'filt':'f277w', 'chan':'long', 'pix_scale':0.063, 'pixar_sr':9.31E-14, 'phot_corr':0.223, 'n_levels':10 }, \
            {'nf':'jw02736001001_f090w_bkg_rot_crop_warp_nobkg2.fits', 'filt':'f090w', 'chan':'short', 'pix_scale':0.031, 'phot_corr':-0.174, 'pixar_sr':2.29E-14, 'n_levels':10 }, \
            {'nf':'jw02736001001_f150w_bkg_rot_crop_warp_nobkg2.fits', 'filt':'f150w', 'chan':'short', 'pix_scale':0.031, 'phot_corr':-0.047, 'pixar_sr':2.31E-14, 'n_levels':10 }, \
            {'nf':'jw02736001001_f200w_bkg_rot_crop_warp_nobkg2.fits', 'filt':'f200w', 'chan':'short', 'pix_scale':0.031, 'phot_corr':-0.114, 'pixar_sr':2.29E-14, 'n_levels':10 } ]

    filterl = [ 'f090w', 'f150w', 'f200w', 'f277w', 'f356w', 'f444w' ]
    binning_factor = 4151 / 2045 # size short / size long
    binf = 2

    sumim = np.zeros((2045, 2045))
    for i, filt in enumerate(filterl):
        for nfd in nfdl:
            if nfd['filt'] == filt:

                nfp = os.path.join(path_wavelets, nfd['nf'][:-4] + 'synth.icl.bcgwavsizesepmask_005_080.fits')
                recim = fits.getdata(nfp)[:2045, :2045]
                sumim += ( recim / recim.max())

    sumim = gaussian_filter(sumim, 15)

    # Sat region files
    nf = 'mahler_noirot_merged_member_gal_ra_dec_pix_long.reg'
    r = pyr.open(os.path.join(path_data, nf))
    patch_list, artist_list = r.get_mpl_patches_texts()

    # Stream region files
    nf = 'streams_flags_pix_long.reg'
    r = pyr.open(os.path.join(path_data, nf))
    patch_list2, artist_list2 = r.get_mpl_patches_texts()

    plt.ion()
    kwargs = {'origin':'lower'}
    cdc, wdc = d.bspl_atrous(sumim, 6)
    wp = wdc.array[:,:,4]
    wp[wp < 0] = 0
    #wdc.waveplot(ncol = 3, origin = 'lower')
    fig, ax = plt.subplots(2, 1)
    ax[0].imshow(sumim, norm = ImageNormalize( sumim, interval = AsymmetricPercentileInterval(40, 99.5), stretch = LogStretch()), origin = "lower", cmap = cmasher.sunburst)
    ax[1].imshow(wp, norm = ImageNormalize( wp, interval = AsymmetricPercentileInterval(0, 99.9), stretch = LogStretch()), origin = "lower", cmap = cmasher.ember)
    ax[0].get_xaxis().set_ticks([])
    ax[0].get_yaxis().set_ticks([])
    ax[1].get_xaxis().set_ticks([])
    ax[1].get_yaxis().set_ticks([])
    for a in artist_list:
        a.set_marker('x')
        a.set_markeredgecolor('white')
        a.set_markersize('4')
        plt.gca().add_artist(a)
    for p in patch_list2:
        p.set_edgecolor('yellow')
        p.set_facecolor(None)
        p.set_fill(False)
        plt.gca().add_artist(p)
    plt.tight_layout()
    plt.show(block = True)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if __name__ == '__main__':

    #plot_array_icl_maps_all_filters()
    #plot_array_bcgicl_maps_all_filters()
    #plot_rgb_mask_wcs()
    #plot_recim_rgb_icl()
    #plot_array_scattered_recim_short()
    #plot_array_recim_long()
    #fICL_vs_filters()
    #FICL_vs_filters()
    #Fgal_vs_filters()
    #fICL_vs_lvlsep()
    #plot_PR()
    #plot_oim_rgb_all_filters()
    #plot_scattered_light_maps()
    #SED_tidal_streams()
    plot_example_recim_all_filter_maps()
    #plot_and_make_sed()
    #plot_test()
