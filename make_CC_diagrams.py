#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute all CC diagrams for the 6 NIRCam filters.
 
Created on Mon Dec 11 21:16:12 2023

@author: aellien
"""
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Modules
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def init_cigale(working_dir, scripting_dir):
    os.chdir(working_dir)
    
    if os.path.isdir('out/'):
        os.system('rm -r out')
    
    if os.path.isfile('pcigale.ini'):
        os.remove('pcigale.ini')
    os.system('pcigale init')

    os.remove('pcigale.ini')
    with open('pcigale.ini', 'w+') as f:
        f.write('''data_file = 
parameters_file = 
sed_modules = sfhdelayedbq, bc03, redshifting
analysis_method = savefluxes
cores = 10''')

    os.chdir(scripting_dir)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def cigale_genconf(working_dir, scripting_dir):
    os.chdir(working_dir)
    os.system('pcigale genconf')
    os.chdir(scripting_dir)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def write_input_script(file_path, metallicity):
    '''
    Should be modified a soon as the SED modules are changed in CIGALE.
    '''
    filename = os.path.join(file_path, 'pcigale.ini')
    if os.path.isfile(filename):
        os.remove(filename)
    with open(filename, 'w+') as f:

        f.write('''data_file = 
parameters_file = 
sed_modules = sfhdelayedbq, bc03, redshifting
analysis_method = savefluxes
cores = 10
bands = jwst.nircam.F090W, jwst.nircam.F090W_err, jwst.nircam.F150W, jwst.nircam.F150W_err, jwst.nircam.F200W, jwst.nircam.F200W_err, jwst.nircam.F277W, jwst.nircam.F277W_err, jwst.nircam.F356W, jwst.nircam.F356W_err, jwst.nircam.F444W, jwst.nircam.F444W_err
properties = 
additionalerror = 0.0

[sed_modules_params]
  
  [[sfhdelayedbq]]
    # e-folding time of the main stellar population model in Myr.
    tau_main = 9000
    # Age of the main stellar population in the galaxy in Myr. The precision
    # is 1 Myr.
    age_main = 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000
    # Age of the burst/quench episode. The precision is 1 Myr.
    age_bq = 500
    # Ratio of the SFR after/before age_bq.
    r_sfr = 0.1
    # Multiplicative factor controlling the SFR if normalise is False. For
    # instance without any burst/quench: SFR(t)=sfr_A×t×exp(-t/τ)/τ²
    sfr_A = 1.0
    # Normalise the SFH to produce one solar mass.
    normalise = True
  
  [[bc03]]
    imf = 0
    metallicity = %s
    separation_age = 1
  
  [[redshifting]]
    redshift = 0.3877


# Configuration of the statistical analysis method.
[analysis_params]
  variables = 
  save_sed = True
  blocks = 1
 '''%metallicity)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def run_cigale(working_dir, scripting_dir):
    os.chdir(working_dir)
    # os.system('pcigale genconf')
    os.system('pcigale run')
    os.chdir(scripting_dir)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def compute_stellar_pop(working_dir, scripting_dir, metallicityl):
    
    for met in metallicityl:
        
        output_dir = os.path.join(working_dir, 'out_Z_%s'%met)
        os.makedirs(output_dir, exist_ok=True)
        init_cigale(working_dir=output_dir, scripting_dir=scripting_dir)
        cigale_genconf(working_dir=output_dir, scripting_dir=scripting_dir)
        write_input_script(file_path=output_dir, metallicity = met)
        run_cigale(working_dir=output_dir, scripting_dir=scripting_dir)
        
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def make_colors(file_path, filterl):
    
    try:
        df = pd.read_csv(file_path, delimiter = ' ')
    except:
        df = pd.read_fwf(file_path)
    color_df = pd.DataFrame()
    filterlc = filterl.copy()    
    for c in df:
        if c in filterlc:
             i = filterlc.index(c)
             filterlc.pop(i)
             n1 = c.split('.')[-1]
             
             for filt in filterlc:
                 n2 = filt.split('.')[-1]
                 color_df['%s-%s'%(n1,n2)] = -2.5 * np.log10(df[c]) + 2.5 * np.log10(df[filt])
    
    return color_df

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def unify_data(working_dir, metallicityl, filterl):
    
    data = {}
    for met in metallicityl:
        output_dir = os.path.join(working_dir, 'out_Z_%s'%met, 'out', 'models-block-0.txt')
        df = make_colors(output_dir, filterl)
        data['Z_%s'%met] = df
        
    return data, list(df.columns)
    
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def plot_CC_diagram(sim_df, obs_df, metallicityl, c1, c2):
    
    plt.figure()
    
    for met in metallicityl:
        
        x = sim_df['Z_%s'%met][c1]
        y = sim_df['Z_%s'%met][c2]

        
        # sim
        plt.plot(x, y, label = 'Z_%s'%met)
        plt.plot(x.iloc[0], y.iloc[0], 'ks', markersize = 3)
        plt.plot(x.iloc[-1], y.iloc[-1], 'ko', markersize = 3)
        
        plt.xlabel(c1)
        plt.ylabel(c2)
    
    # obs
    x2 = obs_df[c1]
    y2 = obs_df[c2]
    s = x2.size + 1
    for i, (x2p, y2p) in enumerate(zip(x2, y2)):
        plt.plot(x2p, y2p, 'o', alpha = 1 - (i / s ), label = 'index %d'%i )
    
    plt.legend()
        
    
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def plot_all_diagrams(sim_df, obs_df, colorl, metallicityl):
    
    colorlc = colorl.copy()
    for c1 in colorlc:
        colorlc.pop(colorlc.index(c1))
        for c2 in colorlc:
            plot_CC_diagram(sim_df, obs_df, metallicityl, c1, c2)
            
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def read_sed(file_path):
    try:
        df = pd.read_csv(file_path, delimiter = ' ')
    except:
        df = pd.read_fwf(file_path)
    return df

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def plot_all_seds(working_dir, obs_sed_file, filterl, metallicityl, n_bin_ages):
    
    plt.figure()
    mpl_colorl = ['blue', 'cyan', 'green', 'yellow', 'orange', 'red']
    
    sed_obs = read_sed(obs_sed_file)

    icl_fluxl = []
    icl_fluxerrl = []
    gal_fluxl = []
    gal_fluxerrl = []
    for filt in filterl:
        icl_fluxl.append(sed_obs[filt][15])
        icl_fluxerrl.append(sed_obs[filt+'_err'][15])
        gal_fluxl.append(sed_obs[filt][16])
        gal_fluxerrl.append(sed_obs[filt+'_err'][16])
        
    icl_fluxl = np.array(icl_fluxl) 
    icl_fluxerrl = np.array(icl_fluxerrl)
    gal_fluxl = np.array(gal_fluxl) 
    gal_fluxerrl = np.array(gal_fluxerrl)
    
    #y_icl = (icl_fluxl - icl_fluxl.min()) / (icl_fluxl.max()- icl_fluxl.min())
    #y_gal = (gal_fluxl - gal_fluxl.min()) / (gal_fluxl.max()- gal_fluxl.min())
    y_icl = icl_fluxl
    y_gal = gal_fluxl

    plt.errorbar(filterl, y_icl, yerr = 0, marker = 'o', color = 'black', linestyle = '', alpha = 1)
    plt.errorbar(filterl, y_gal, yerr = 0, marker = 'o', color = 'black', linestyle = '', alpha = 0.5)


    for k, met in enumerate(metallicityl):
        output_dir = os.path.join(working_dir, 'out_Z_%s'%met, 'out', 'models-block-0.txt')
        sed_df = read_sed(output_dir)
        
        
        for i in range(n_bin_ages):
            y = sed_df.iloc[i][1:7].sort_index().values # /!\ to change as soon as 
            x = sed_df.iloc[0][1:7].sort_index().index  # 'write_input_script()' is modified /!\
            
            #y = ( y - y.min() ) / ( y.max() - y.min() )
            y *= 7E8
            plt.plot(x, y, alpha =  1 - ( i / n_bin_ages), color = mpl_colorl[k] )
            
    plt.yscale('log')
    plt.tight_layout()
    plt.show()
    return sed_df

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if __name__ == '__main__':
    # Path list & variables
    working_dir = '/home/aellien/JWST/analysis/CC_diagrams'
    scripting_dir = '/home/aellien/JWST/JWST_scripts'
    
    obs_sed_file = '/home/aellien/JWST/analysis/sed_all_regions.txt'
    
    metallicityl = [ '0.004' ]
    n_bin_ages = 10
    filterl = ['jwst.nircam.F090W', 'jwst.nircam.F150W', 'jwst.nircam.F200W', 'jwst.nircam.F277W', 'jwst.nircam.F356W', 'jwst.nircam.F444W' ]
   
    compute_stellar_pop(working_dir, scripting_dir, metallicityl)

    # CC diagrams
    #sim_df, colorl = unify_data(working_dir, metallicityl, filterl)
    #obs_df = make_colors(obs_sed_file, filterl)[-2:]
    #plot_all_diagrams(sim_df, obs_df, colorl, metallicityl)

    # SEDs
    sed_df = plot_all_seds(working_dir, obs_sed_file, filterl, metallicityl, n_bin_ages)

