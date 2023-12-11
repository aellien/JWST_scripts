import numpy as np
import os
import dawis as d
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.io import fits
from astropy.visualization import *
from matplotlib.colors import SymLogNorm


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def check_noirot_cat():

    # Paths, lists & variables
    path_data = '/home/ellien/JWST/data/'
    path_scripts = '/home/ellien/JWST/JWST_scripts'
    path_plots = '/home/ellien/JWST/plots'

    cat = 'noirot_redshifts.cat'
    outcat = 'noirot_grism_gal_members_ra_dec_z.cat'
    z = 0.3877 # redshift MACS~J0723
    dz = 0.01
    z_sup = z + dz
    z_inf = z - dz
    ra_bcg = 110.82712
    dec_bcg = -73.45472

    n_grism = 0
    n_spec = 0
    n_member = 0
    c = []

    with open(os.path.join(path_data, cat),'r') as f:
        f.readline()
        for l in f:
            s = l.split()
            ra = float(s[1])
            dec = float(s[2])
            try:
                z_grism = float(s[21])
                n_grism += 1
            except:
                z_grism = -999.
            try:
                z_spec = float(s[24])
                n_spec += 1
            except:
                z_spec = -999.

            if (z_grism >= z_inf) & (z_grism <= z_sup):
                n_member += 1
                c.append([ra, dec, z_grism])


    print('n_grism = %d\n_spec = %d\nn_member = %d'%(n_grism, n_spec, n_member))

    with open(os.path.join(path_data, outcat),'w+') as f:
        f.write('#RA DEC z_grism\n')
        for l in c:
            print(l)
            f.write('%f %f %f\n'%(l[0], l[1], l[2]))

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def make_gal_cat_radii():

    # Paths, lists & variables
    path_data = '/home/ellien/JWST/data/'
    path_scripts = '/home/ellien/JWST/JWST_scripts'
    path_plots = '/home/ellien/JWST/plots'

    cat = 'mahler_noirot_merged_member_gal_ra_dec.cat'
    radii = [ 128, 200, 400 ] # kpc, same as in Mahler 2023
    z = 0.3877 # redshift MACS~J0723
    dz = 0.01
    pix_scale = 0.063 # "/pix
    physcale = 5.3 # kpc/"
    z_sup = z + dz
    z_inf = z - dz
    ra_bcg = 110.82734 # deg
    dec_bcg = -73.454683 # deg
    coo_bcg = SkyCoord( '110.82734', '-73.454683', unit = u.deg, frame = 'fk5')
    # read catalog
    coo_gal_list = []
    with open(os.path.join(path_data, cat),'r') as f:
        f.readline()
        for l in f:
            s = l.split()
            ra = s[0] # deg
            dec = s[1] # deg
            coo_gal = SkyCoord( ra, dec, unit = u.deg, frame = 'fk5')
            coo_gal_list.append(coo_gal)

    # make catalogs of galaxies within different radii
    for rad in radii:
        outcat = 'mahler_RS_ra_dec_z_%dkpc.cat'%rad
        with open(os.path.join(path_data, outcat), 'w+') as of:
            n_member = 0
            for coo_gal in coo_gal_list:
                sep = coo_bcg.separation(coo_gal)
                if sep.arcsecond * physcale <= rad :
                    of.write('%f %f\n'%(coo_gal.ra.deg,coo_gal.dec.deg))
                    n_member += 1


        print('r = %d kpc ~ %f" --> %d galaxies'%(rad, rad / physcale, n_member))

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def make_source_catalog( oim, n_levels, n_sig_gal = 50, level_gal = 3, display = True ):

    # path, list & variables
    xs, ys = oim.shape
    gal = np.zeros( (xs, ys) )

    sigma, mean, gain = d.pg_noise_bissection( oim, max_err = 1E-3, n_sigmas = 3 )
    aim = d.anscombe_transform( oim, sigma, mean, gain )
    acdc, awdc = d.bspl_atrous( aim, n_levels )
    sdc = d.hard_threshold( awdc, n_sig_gal )
    sup = sdc.array[:,:,level_gal]
    lab = d.label( sup )
    reg = d.regionprops( lab )

    cat = []
    for r in reg:
     cat.append( (r.centroid[1], r.centroid[0] ) )

    if display == True:
     logthresh = 1E-3
     norm = SymLogNorm(logthresh)
     fig, ax = plt.subplots( 1, 3 )
     ax[0].imshow( oim, norm = ImageNormalize( oim, \
                                       interval = ZScaleInterval(), \
                                       stretch = LinearStretch()), \
                        cmap = 'gray_r' )

     ax[1].imshow( sdc.array[:,:,level_gal], norm = norm )
     ax[2].imshow( sup )

     for r in reg:
         ax[0].plot( r.centroid[1], r.centroid[0], 'b+' )
         ax[1].plot( r.centroid[1], r.centroid[0], 'b+' )

     plt.show()

    return np.array(cat)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if __name__ == '__main__':

    # Paths, lists & variables
    path_data = '/home/ellien/JWST/data/'
    path_scripts = '/home/ellien/JWST/JWST_scripts'
    path_plots = '/home/ellien/JWST/plots'

    cat = 'mahler_RS_ra_dec_z.cat'
    radii = [ 128, 200, 400 ] # kpc, same as in Mahler 2023
    z = 0.3877 # redshift MACS~J0723
    dz = 0.01
    pixscale = 0.063 # "/pix
    physcale = 5.3 # kpc/"
    z_sup = z + dz
    z_inf = z - dz
    ra_bcg = 110.82734 # deg
    dec_bcg = -73.454683 # deg
    coo_bcg = SkyCoord( '110.82734', '-73.454683', unit = u.deg, frame = 'fk5')
    rc = 10 # kpc
    rc_pix = rc / physcale / pixscale # pixel


    filterl = [ 'f277w', 'f356w', 'f444w']
    for filt in filterl:
        hdu = fits.open(os.path.join( path_data, 'jw02736001001_%s_bkg_rot_crop_input.fits'%filt ))
        oim, header = hdu[0].data, hdu[0].header
        wcs = WCS(header)

        # read mahler catalog
        coo_gal_list_pix = []
        with open(os.path.join(path_data, cat),'r') as f:
            f.readline()
            for l in f:
                s = l.split()
                ra = s[0] # deg
                dec = s[1] # deg
                coo_gal = SkyCoord( ra, dec, unit = u.deg, frame = 'fk5')
                coo_gal = wcs.world_to_pixel(coo_gal)
                coo_gal_list_pix.append(coo_gal)

        coo_source_list_pix = make_source_catalog( oim, n_levels = 10, n_sig_gal = 5, level_gal = 4, display = False)

        coo_source_list_wcs = []
        for coo in coo_source_list_pix:
            coo_wcs = wcs.pixel_to_world(coo[0], coo[1])
            coo_source_list_wcs.append([ coo_wcs.ra.deg, coo_wcs.dec.deg ])

        nocl = []
        with open( os.path.join(path_data, 'not_cluster_src.cat'), 'w+' ) as of:
            for xco, yco in coo_source_list_pix:
                flag = False
                for xgal, ygal in coo_gal_list_pix:
                    dr = np.sqrt( (xgal - xco)**2 + (ygal - yco)**2 )
                    if dr <= rc_pix:
                        flag = True
                        break
                if flag == False:
                    of.write('%f %f\n'%(xco, yco))
                    nocl.append([xco, yco])
        nocl = np.array(nocl)

        plt.figure()
        plt.imshow(oim, cmap = 'gray_r', norm =  ImageNormalize( oim, \
                                          interval = ZScaleInterval(), \
                                          stretch = LinearStretch()))
        plt.plot(np.swapaxes( nocl, 0, 1)[0], np.swapaxes( nocl, 0, 1)[1], 'b+' )
        plt.plot(np.swapaxes( coo_gal_list_pix, 0, 1)[0], np.swapaxes( coo_gal_list_pix, 0, 1)[1], 'r+' )

    plt.show()

    #make_gal_cat_radii()
