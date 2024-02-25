#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 14:13:55 2023

@author: aellien
"""
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Modules
import ultranest as u
import pcigale as pc
import os as os
import pandas as pd
import numpy as np

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Ultranest related
def create_uniform_prior_for(model, pname, index, pval, pdelta, pmin, pmax):
    """
    Use for location variables (position)
    The uniform prior gives equal weight in non-logarithmic scale.
    Modified from BXA doc: https://github.com/JohannesBuchner/BXA
    """

    print('  uniform prior for %s between %f and %f ' % (pname, pmin, pmax))

    low = float(pmin)
    spread = float(pmax - pmin)
    if pmin > 0 and pmax / pmin > 100:
        print('   note: this parameter spans several dex. Should it be log-uniform (create_jeffreys_prior_for)?')
	
    def uniform_transform(x):
        return x * spread + low
	
    #return dict(model=model, index=par._Parameter__index, name=par.name, transform=uniform_transform, aftertransform=lambda x: x)
    return dict(model=model, index=index, name=pname, transform=uniform_transform, aftertransform=lambda x: x)

def simple_prior(cube):
    params = cube.copy()

    params[0] = cube[0] * 12000 + 500 # Tau_main
    params[1] = cube[1] * 1000 + 50 # Tau_starburst
    params[2] = cube[2] * 0.5 + 0 # fraction_starbust
    params[3] = cube[3] * 12000 + 1000 # Age, resolution 100 Myr
    params[4] = cube[4] * 200 + 10 # age starbust
    params[5] = cube[5] * 100 + 1 # age starbust
    
    return params

def likelihood(params):
    
    os.chdir(working_dir)
    if os.path.isfile('out/results.txt'):
        print('Removing old output directory')
        os.system('rm -r out')
    
    config = pc.Configuration()
    pc.init(config)
    
    # Some global variables, see main /!\
    config.pcigaleini_exists = True
    config.config['data_file'] = data_file
    config.config['parameters_file'] = ''
    config.config['sed_modules'] = sed_modules
    config.config['analysis_method'] = 'pdf_analysis'
    config.config['cores'] = cores
    
    pc.genconf(config)
    config.config['bands'] = bands
    config.config['additionalerror'] = additionalerror
    
    smp_dic = config.config['sed_modules_params']
    i = 0
    k = 0
    l = 0
    for mod_key in smp_dic:
        for par_key in smp_dic[mod_key]:
            if i in frozen_params:
                smp_dic[mod_key][par_key] = str(frozen_param_values[l])
                #print(par_key, 'frozen', k, i, smp_dic[mod_key][par_key])
                l += 1
            else:
                smp_dic[mod_key][par_key] = str(int(params[k]))
                #print(par_key, 'free', k, i, smp_dic[mod_key][par_key])
                k += 1
            i += 1
            
    pc.run(config)
    results = pd.read_fwf('out/results.txt')
    os.chdir(scripting_dir)
    chi2 = results['best.chi_square'].values[0]
    if not np.isfinite(chi2):
        chi2 = -1e+100
    
    return chi2

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if __name__ == '__main__':

    # Path list & variables
    working_dir = '/home/aellien/JWST/analysis/sed_out5/test3'
    scripting_dir = '/home/aellien/JWST/JWST_scripts'
    data_file = os.path.join(working_dir, 'sed_test.txt')
    cores = 1
    additionalerror = 0.
    sed_modules = ['sfh2exp', 'bc03', 'redshifting']
    bands = ['jwst.nircam.F090W', 'jwst.nircam.F150W', 'jwst.nircam.F200W', \
             'jwst.nircam.F277W', 'jwst.nircam.F356W', 'jwst.nircam.F444W' ]
    
    frozen_params = [ 5, 6, 7, 8, 10 ]
    frozen_param_values = [ 1.0, True, 0, 0.02, 0.3877 ]
    
    free_params = [ 0, 1, 2, 3, 4, 9 ]
    sampler = u.ReactiveNestedSampler(['tau_main', 'tau_starburst', 'f_starburst', 'Age_stars', 'Age_starburst', 'sep_age'], 
                                      likelihood, simple_prior, 
                                      log_dir = '/home/aellien/JWST/analysis/sed_out5/test4')
    sampler.run()
    sampler.print_results()
    sampler.plot_corner()