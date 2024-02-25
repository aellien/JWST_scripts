import sys
import glob as glob
import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def moments(image, order, radius, display = False):
    '''Compute radial moments of an image.
    - Arguments:
    image       # image of which the moment is computed
    order       # order of the moments (0, 1, 2,...)
    radius      # radius within which the moments are computed [pixels]

    - Output
    A, B        # Moments

    Source: Buote & Tsai 1996.
    '''

    xs, ys = image.shape                # image shape, ex: 1000x1000 pixels
    x = np.linspace( -xs/2., xs/2, xs ) # 1d coordinates with image center as origin, ex: -500, .., 0, ..., 500
    y = np.linspace( -ys/2., ys/2, ys ) # 1d coordinates with image center as origin, ex: -500, .., 0, ..., 500
    xx, yy = np.meshgrid(x, y)       # 2d cartesian coordinate images with image center as origin - pixels
    rr = np.sqrt(xx**2 + yy**2)         # 2d radial coordinate image with image center as origin - pixels
    theta = np.arctan2(yy, xx)          # 2d polar coordinate image with image center as origin - rads

    if order == 0:
        A = np.sum(image[rr <= radius]) # order 0 is just the area within given radius
        return A

    else:
        a = image * np.power(rr, order) * np.cos(order * theta) # weighted polar image
        b = image * np.power(rr, order) * np.sin(order * theta) # weighted polar image
        A = np.sum( a[rr <= radius] ) # moment = integration within given radius
        B = np.sum( b[rr <= radius] ) # moment = integration within given radius

        if display == True:
            fig, ax = plt.subplots(2)
            ax[0].imshow(a)
            ax[1].imshow(b)
            plt.show()

        return A, B

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def power_ratio(image, order, radius):
    '''Compute power ratio of an image.
    - Arguments:
    image       # image of which the moment is computed
    order       # order of the power ratio (0, 1, 2,...)
    radius      # radius within which the moments are computed [pixels]

    - Output
    P           # Power ratio - P_order / P_0

    Source: Buote & Tsai 1996.
    '''

    A0 = moments(image = image, order = 0, radius = radius) # First order moment
    A, B = moments(image = image, order = order, radius = radius)

    P0 = (A0 * np.log(radius))**2 # First order power
    P    = ( A**2 + B**2 ) / ( 2 * order**2 * radius**(2*order) ) # power of order m
    PR = P / P0

    return PR

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def test_PR_gallery():

    # Paths, lists & variables
    path_data = '/home/ellien/JWST/power_ratio/'
    path_scripts = '/home/ellien/JWST/JWST_scripts'
    path_wavelets = '/home/ellien/JWST/wavelets/out12/'
    path_plots = '/home/ellien/JWST/plots'

    figure_l = glob.glob( os.path.join( path_data, '*.fits' ) )

    for figure in figure_l:

        image = fits.getdata(figure)

        R_l = np.linspace( 50, 500, 20 ) # pix
        order_l = np.arange(1, 5) # order

        plt.figure()
        for order in order_l:
            PR_l = []
            for R in R_l:
                PR = power_ratio( image = image, order = order, radius = R )
                PR_l.append(PR)

            plt.plot(R_l, PR_l, linewidth = 2, label = 'm = %d'%order)

        plt.yscale('log')
        plt.gca().tick_params(axis='x', labelsize = 15)
        plt.gca().tick_params(axis='y', labelsize = 15)
        plt.legend(fontsize = 15)
        plt.xlabel('Radius [pix]')
        plt.ylabel('Power Ratio')
        plt.suptitle('%s'%figure.split('/')[-1][:-5])

    plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def plot_PR_vs_R():

    # Paths, lists & variables
    path_data = '/home/ellien/JWST/power_ratio/'
    path_scripts = '/home/ellien/JWST/JWST_scripts'
    path_wavelets = '/home/ellien/JWST/wavelets/out12/'
    path_plots = '/home/ellien/JWST/plots'

    figure_l = glob.glob( os.path.join( path_data, '*.fits' ) )

    for figure in figure_l:

        image = fits.getdata(figure)

        R_l = np.linspace( 50, 500, 20 ) # pix
        order_l = np.arange(1, 5) # order

        plt.figure()
        for order in order_l:
            PR_l = []
            for R in R_l:
                PR = power_ratio( image = image, order = order, radius = R ) * 10**7
                PR_l.append(PR)

            plt.plot(R_l, PR_l, linewidth = 2, label = 'm = %d'%order)

        plt.yscale('log')
        plt.gca().tick_params(axis='x', labelsize = 15)
        plt.gca().tick_params(axis='y', labelsize = 15)
        plt.legend(fontsize = 15)
        plt.xlabel('Radius [pix]')
        plt.ylabel('Power Ratio')
        plt.suptitle('%s'%figure.split('/')[-1][:-5])
        plt.savefig(os.path.join(path_plots, figure[:-5] + '.pdf'), format = 'pdf')

    plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def plot_PR_vs_fig():

    # Paths, lists & variables
    path_data = '/home/ellien/JWST/power_ratio/'
    path_scripts = '/home/ellien/JWST/JWST_scripts'
    path_wavelets = '/home/ellien/JWST/wavelets/out12/'
    path_plots = '/home/ellien/JWST/plots'

    figure_l = glob.glob( os.path.join( path_data, '*.fits' ) )
    R = 500
    order_l = np.arange(1, 5) # order

    for order in order_l:

        plt.figure()
        figl = []
        for i, figure in enumerate(figure_l):
            image = fits.getdata(figure)
            fig = figure.split('/')[-1][:-5]
            figl.append(fig)
            PR = power_ratio( image = image, order = order, radius = R ) * 10**7

            plt.plot(i, PR, '+', markersize = 15, label = '%s'%fig)

        print(figl)
        plt.yscale('log')
        plt.gca().tick_params(axis='x', labelsize = 15)
        plt.gca().tick_params(axis='y', labelsize = 15)
        plt.gca().set_xticklabels(['']+figl, rotation = 45, ha = 'right')
        plt.ylabel('Power Ratio', fontsize = 15)
        plt.suptitle('PR %d'%order)
        plt.tight_layout()
        plt.savefig(os.path.join(path_plots, 'order_%d.pdf'%order), format = 'pdf')

    plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if __name__ == '__main__':

    # Paths, lists & variables
    path_data = '/home/aellien/JWST/power_ratio/'
    path_scripts = '/home/aellien/JWST/JWST_scripts'
    path_wavelets = '/home/aellien/JWST/wavelets/out15/'
    path_plots = '/home/aellien/JWST/plots'

    figure_l = glob.glob( os.path.join( path_data, '*.fits' ) )
    R = 500
    order_l = np.arange(1, 5) # order

    for order in order_l:

        plt.figure()
        figl = []
        for i, figure in enumerate(figure_l):
            image = fits.getdata(figure)
            fig = figure.split('/')[-1][:-5]
            figl.append(fig)
            PR = power_ratio( image = image, order = order, radius = R ) * 10**7

            plt.plot(i, PR, '+', markersize = 15, label = '%s'%fig)

        print(figl)
        plt.yscale('log')
        plt.gca().tick_params(axis='x', labelsize = 15)
        plt.gca().tick_params(axis='y', labelsize = 15)
        plt.gca().set_xticklabels(['']+figl, rotation = 45, ha = 'right')
        plt.ylabel('Power Ratio', fontsize = 15)
        plt.suptitle('PR %d'%order)
        plt.tight_layout()
        plt.savefig(os.path.join(path_plots, 'demo_PR_order_%d.pdf'%order), format = 'pdf')

    plt.show()
