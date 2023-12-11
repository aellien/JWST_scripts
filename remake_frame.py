import os
import dawis as d
import sys
import glob as glob
import numpy as np
from astropy.io import fits

rm_gamma_for_big = True
lvl_sep_big = 6
gamma = 0.5

for flt in ['f277', 'f356', 'f444']:

    oim_file = '/home/ellien/JWST/data/%s_rot.fits'%flt
    oim = fits.getdata(oim_file)
    res = np.copy(oim)
    rec = np.zeros(oim.shape)
    olnl = glob.glob( '/home/ellien/JWST/wavelets/out3/%s_rot.*ol*pkl'%flt )
    olnl.sort()
    outpath = '/home/ellien/JWST/wavelets/out3/%s_rot'%flt

    for oln in olnl:

        it = int( oln[-7:-4] )
        ol = d.read_objects_from_pickle(oln)

        # Atom
        atom = np.zeros(res.shape)
        for object in ol:
            x_min, y_min, x_max, y_max = object.bbox
            if (rm_gamma_for_big == True) & (object.level >= lvl_sep_big):
                atom[ x_min : x_max, y_min : y_max ] += object.image
            else:
                atom[ x_min : x_max, y_min : y_max ] += object.image * gamma

        # Update Residuals
        res -= atom
        rec += atom


        d.plot_frame( level = ol[0].level, it = it, nobj = len(ol), \
                                           original_image = oim, \
                                           restored_image = rec, \
                                           residuals = res, \
                                           atom = atom, \
                                           outpath = outpath )
