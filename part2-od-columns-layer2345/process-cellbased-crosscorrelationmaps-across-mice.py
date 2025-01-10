#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This script loads binned odi map data and calculates the cross correlations over the binned planes

python process-cellbased-crosscorrelationmaps-across-mice.py 

Created on Sunday 15 May 2022

@author: pgoltstein
"""


# Imports
import sys, os
import numpy as np
from skimage.transform import resize as imresize

# Local imports
sys.path.append('../xx_analysissupport')
import plottingtools
from tqdm import tqdm

# Module settings
plottingtools.font_size = { "title": 6, "label": 5, "tick": 5, "text": 5, "legend": 5 }

# Arguments
import argparse


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Arguments

parser = argparse.ArgumentParser( description = "This script loads binned odi map data and calculates the cross correlations over the binned planes.\n (written by Pieter Goltstein - May 2022)")
parser.add_argument('-sh', '--shufflename', type=str, default=None, help='Number to add to save shuffled odi maps (always loads shuffle 0)')
parser.add_argument('-pl', '--shufflemice', type=int, default=None, help='Flag enables shuffling of planes across mice')
parser.add_argument('-sw', '--swapname', type=str, default=None, help='Number to add to save swapped odi maps (always loads swap 0)')
parser.add_argument('-d', '--swapdistance', type=int, default=None, help='Flag sets the distance over which the local swap is done')
args = parser.parse_args()


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Settings

# Path settings
odimapsdatapath = os.path.join("../../data/part2-odimapsdata-layer2345")
ccmapsdatapath = os.path.join("../../data/part2-ccmapsdata-layer2345")
print(f"{odimapsdatapath=}")

# Select mice
mice = ["O02","O03","O06","O07","O09","O10","O11","O12","O13"]
n_mice = len(mice)

# Map settings
odi_cell_sigma = 16
n_depth_bins = 4
depth_bins = np.arange(170,531,90) # 4 bins
ref_map_nr = 2 #2=350-430 (inclusive)

# Minimum number of pixels that maps have to overlap (in percentage of all pixels)
min_n_pix_cc_perc = 0.2

# resize map
cc_map_npix = 64
band_avg = int(np.round(cc_map_npix/10)) # pixels left and right of middle of map to take into account, set to about 10% of the image


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Functions

def spatial_crosscorrelation( im, im2, min_n_pix_cc=19 ):
    """ Calculates the spatial cross-correlation similar as to the auto-correlation used in grid cell papers, such as Hafting, T., Fyhn, M., Molden, S. et al. Microstructure of a spatial map in the entorhinal cortex. Nature 436, 801–806 (2005). https://doi.org/10.1038/nature03721

        NOTE: the equation in the Moser papers (at least the early ones) is incorrect, the correct equation for the The Pearson Product-Moment Correlation equation can be found all over the internet
            e.g. here: https://www.statisticshowto.com/cross-correlation/
            or below:

        (n * sum(l1_xy * l1_xy_t)) - (sum(l1_xy) * sum(l1_xy_t))
        ---------------------------------------------------------
        sqrt( (n*sum(l1_xy^2))-(sum(l1_xy)^2) ) * sqrt( (n*sum(l1_xy_t^2))-(sum(l1_xy_t)^2) )

    """

    # Pad array with NaN's on all sides
    y_size,x_size = im.shape
    nan_im = np.full((y_size*3,x_size*3), np.NaN)
    nan_im[y_size:y_size*2,x_size:x_size*2] = im
    nan_im2 = np.full((y_size*3,x_size*3), np.NaN)
    nan_im2[y_size:y_size*2,x_size:x_size*2] = im2

    # define center, and range of spatial lags
    x_half = np.floor(im.shape[1]/2).astype(int)
    y_half = np.floor(im.shape[0]/2).astype(int)
    x_mid = np.floor(nan_im.shape[1]/2).astype(int)
    y_mid = np.floor(nan_im.shape[0]/2).astype(int)
    x_fov = np.arange(-x_half,x_half+1,1).astype(int)
    y_fov = np.arange(-y_half,y_half+1,1).astype(int)
    x_range = np.arange(x_size-x_half,(x_size*2)+x_half,1).astype(int)
    y_range = np.arange(y_size-y_half,(y_size*2)+y_half,1).astype(int)

    # Output variable
    output_im = np.zeros((len(y_range),len(x_range)))

    # non-changing equation elements
    l1_xy = nan_im[y_fov[0]+y_mid:y_fov[-1]+y_mid,x_fov[0]+x_mid:x_fov[-1]+x_mid]
    include_pixels1 = ~np.isnan(l1_xy)

    # loop spatial lags
    for t_y in y_range:
        for t_x in x_range:

            # Output matrix indices
            y_o, x_o = t_y-y_half, t_x-x_half

            # changing equation elements
            l1_xy_t = nan_im2[(y_fov[0]+t_y):(y_fov[-1]+t_y),(x_fov[0]+t_x):(x_fov[-1]+t_x)]
            include_pixels2 = ~np.isnan(l1_xy_t)
            include_pixels = np.logical_and(include_pixels1,include_pixels2)
            n_pix = np.sum(include_pixels)

            # According to moser, only calculate if enough pixels in map
            if n_pix > min_n_pix_cc:

                # Get part of image with valid pixels
                l1_xy_incl = l1_xy[include_pixels]
                l1_xy_t_incl = l1_xy_t[include_pixels]

                # equation groups
                eq_numerator = (n_pix * np.nansum(l1_xy_incl*l1_xy_t_incl)) - (np.nansum(l1_xy_incl)*np.nansum(l1_xy_t_incl))
                eq_denominator = np.sqrt( (n_pix*np.nansum(np.power(l1_xy_incl,2))) - np.power(np.nansum(l1_xy_incl),2)) * np.sqrt( (n_pix*np.nansum(np.power(l1_xy_t_incl,2))) - np.power(np.nansum(l1_xy_t_incl),2))

                # calculate spatial autocorrelation
                output_im[y_o,x_o] = eq_numerator/eq_denominator
            else:
                # Else, just put in a NaN
                output_im[y_o,x_o] = np.NaN

    # Return autocorr image
    return output_im


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Load data and calculate cross correlations

# Data containers
crosscorr_map_mice = []
cc_plot_x_mice = []
cc_plot_y_mice = []

# Loop mice
for m_nr,mouse in enumerate(mice):

    # Load maps
    datafilename = os.path.join(odimapsdatapath,"odimaps-sigma{}-{}bins-{}".format(odi_cell_sigma,n_depth_bins,mouse))
    if args.shufflename is not None:
        datafilename += "-shuffled_odi-{}".format(args.shufflename)
    if args.swapname is not None:
        datafilename += "-swapped_odi_{}-{}".format(args.swapdistance,args.swapname)
    datafilename += ".npz"
    print("Loading: {}".format(datafilename))
    datafile = np.load(datafilename)
    odi_maps = datafile["odi_maps_cells"]

    # Load and overwrite maps with random mice if requested (shuffle miceplanes)
    if args.shufflemice is not None:
        for d in range(n_depth_bins):
            if d != ref_map_nr:

                # select random mouse
                r_mouse = mouse
                while r_mouse == mouse:
                    r_mouse = np.random.choice(mice)
                print("Map nr {}, mouse = {} (instead of {})".format(d,r_mouse,mouse))

                # Load data of random mouse
                r_datafilename = os.path.join(odimapsdatapath,"odimaps-sigma{}-{}bins-{}.npz".format( odi_cell_sigma, n_depth_bins, r_mouse ))
                print("Loading: {}".format(r_datafilename))
                r_datafile = np.load(r_datafilename)
                r_odi_maps_cells = r_datafile["odi_maps_cells"]

                # Replace the maps
                odi_maps[:,:,d] = r_odi_maps_cells[:,:,d]

    # Get dimensions and resize dimensions
    y_res,x_res,n_maps = odi_maps.shape
    aspect_ratio = x_res/y_res
    print("aspect_ratio = {}".format(aspect_ratio))
    cc_map_y = int(cc_map_npix)
    cc_map_x = int(np.ceil(cc_map_y*aspect_ratio))
    if np.mod(cc_map_x,2) == 1:
        cc_map_x += 1
    min_n_pix_cc = int(np.round(min_n_pix_cc_perc * (cc_map_y*cc_map_x)))
    print("min_n_pix_cc = {}".format(min_n_pix_cc))

    # Get rid of Inf's
    print("Odimaps, converted {} infs to NaN".format(np.sum(np.isinf(odi_maps))))
    odi_maps[odi_maps>1.0] = np.NaN
    odi_maps[odi_maps<-1.0] = np.NaN

    # Get ref map
    ref_map = odi_maps[:,:,ref_map_nr]
    ref_map = imresize(ref_map, (cc_map_y, cc_map_x))

    # Data containers
    crosscorr_map_mice.append( np.zeros((cc_map_y*2,cc_map_x*2,n_maps)) )
    cc_plot_x_mice.append( np.zeros((cc_map_x*2,n_maps)) )
    cc_plot_y_mice.append( np.zeros((cc_map_y*2,n_maps)) )

    # Loop maps
    with tqdm(total=n_maps, initial=0, desc="Calculating cross-correlation maps", unit="maps") as bar:
        for nr in range(n_maps):

            # Get second map
            test_map = odi_maps[:,:,nr]
            test_map = imresize(test_map, (cc_map_y, cc_map_x))

            # Crosscorrelation
            crosscorr_map = spatial_crosscorrelation( ref_map, test_map, min_n_pix_cc )
            y_cc,x_cc = crosscorr_map.shape

            # Store map in data container
            crosscorr_map_mice[m_nr][:,:,nr] = crosscorr_map

            # Calculate projection orthogonal to this band
            map_mid_y = int(np.round(y_cc/2))
            cc_plot_x_mice[m_nr][:,nr] = np.nanmean(crosscorr_map[map_mid_y-band_avg:map_mid_y+band_avg],axis=0)

            # Calculate projection along this band
            map_mid_x = int(np.round(x_cc/2))
            cc_plot_y_mice[m_nr][:,nr] = np.nanmean(crosscorr_map[:,map_mid_x-band_avg:map_mid_x+band_avg],axis=1)

            # Update progress bar
            bar.update(1)

# Stack across mice into one matrix
crosscorr_map_mice = np.stack(crosscorr_map_mice,axis=3)
cc_plot_x_mice = np.stack(cc_plot_x_mice,axis=2)
cc_plot_y_mice = np.stack(cc_plot_y_mice,axis=2)

# Save data containers
savepath = os.path.join(ccmapsdatapath, "cc-maps-sigma{}-{}bins-size{}".format(odi_cell_sigma,n_depth_bins,cc_map_npix))
if args.shufflemice is not None:
    savepath += "-shuffled_planes-{}".format(args.shufflemice)
if args.shufflename is not None:
    savepath += "-shuffled_odi-{}".format(args.shufflename)
if args.swapname is not None:
    savepath += "-swapped_odi_{}-{}".format( args.swapdistance, args.swapname )
savepath += ".npz"
print("Saving: {}".format(savepath))
np.savez(savepath,crosscorr_map_mice=crosscorr_map_mice, cc_plot_x_mice=cc_plot_x_mice, cc_plot_y_mice=cc_plot_y_mice, n_maps=n_maps)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Done

print("\nDone.. that's all folks!")
# plt.show()
