#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 10:29:05 2024

@author: aellien
"""
import h5py
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
import pickle
from astropy.io import fits
from astropy.visualization import *
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.morphology import binary_dilation
from scipy.stats import kurtosis
import gc

                

if __name__ == '__main__':

    path_wavelets = '/home/aellien/JWST/wavelets/out20/'
    
    #for pickle_file in *ol*.pkl; do python /home/aellien/JWST/JWST_scripts/pkl_tests.py  "$pickle_file";done
    '''nfp = sys.argv[1]
    print(nfp)
    pkl_to_hdf5_conversion(nfp)'''
    
    #ol1 = d.read_objects_from_pickle(nfp)
    #ol2 = read_objects_from_hdf5(nfp[:-4]+'.hdf5')
    '''for o1, o2 in zip(ol1, ol2):
        for attr1, attr2 in zip(list(o1.__dict__.values()), list(o2.__dict__.values())):
            print(attr1, attr2)'''
    
    #for filt in [ 'f277w', 'f356w', 'f444w' ]:
    #    concatenate_hdf5_files(nfp = '/home/aellien/JWST/wavelets/out20/jw02736001001_%s_bkg_rot_crop_input'%filt)

    '''
    itl1 = d.read_interscale_trees_from_pickle('/home/aellien/JWST/wavelets/out15/jw02736001001_f200w_bkg_rot_crop_warp_nobkg2.itl.it050.pkl')
    save_itl_to_hdf5(itl1, 'test_itl.hdf5')
    itl2 = read_itl_from_hdf5('test_itl.hdf5')
    print(pickle.dumps(itl1) == pickle.dumps(itl2))
    
    ol1 = d.read_objects_from_pickle('/home/aellien/JWST/wavelets/out15/jw02736001001_f200w_bkg_rot_crop_warp_nobkg2.ol.it050.pkl')
    save_ol_to_hdf5(ol1, 'test_ol.hdf5')
    ol2 = read_ol_from_hdf5('test_ol.hdf5')
    print(pickle.dumps(ol1) == pickle.dumps(ol2))
    '''
    
    for filt in [ 'f090w', 'f150w', 'f200w' ]:
        
        nf = 'jw02736001001_f090w_bkg_rot_crop_warp_nobkg2'
        nfp = os.path.join(path_wavelets, nf)
        
        print(nfp)
        
        nfopl = glob.glob(nfp + '.ol.it???pkl')
        nfopl.sort()
        nfitpl = glob.glob(nfp + '.itl.it???pkl')
        nfitpl.sort()
        
        for nfop, nfitp in zip( nfoptl, nfitpl ):
            pkl_to_hdf5(nfop, nfitp)
        
    for filt in ['f277w', 'f356w', 'f444w' ]:
        
        nf = 'jw02736001001_f444w_bkg_rot_crop_input'
        nfp = os.path.join(path_wavelets, nf)
        
        print(nfp)
        
        nfopl = glob.glob(nfp + '.ol.it???pkl')
        nfopl.sort()
        nfitpl = glob.glob(nfp + '.itl.it???pkl')
        nfitpl.sort()
        
        for nfop, nfitp in zip( nfoptl, nfitpl ):
            pkl_to_hdf5(nfop, nfitp)