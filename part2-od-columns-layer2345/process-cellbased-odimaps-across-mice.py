#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This script loads processed data of all L23-L4-L5 experiments, cell based ODI maps for each depth and saves them to a .npz file with all maps

python process-cellbased-odimaps-across-mice.py

Created on Sunday 15 May 2022

@author: pgoltstein
"""

# Imports
import sys, os
import numpy as np
import sklearn.metrics
from scipy.ndimage import gaussian_filter

# Detect operating system and add local import dir
sys.path.append('../xx_analysissupport')
import singlevolumeodfunctions

# Arguments
import argparse

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Arguments

parser = argparse.ArgumentParser( description = "This script loads processed data of all L23-L4-L5 experiments, cell based ODI maps for each depth and saves them to a .npz file with all maps.\n (written by Pieter Goltstein - May 2022)")
parser.add_argument('-sh', '--shuffleodi', type=int, default=None, help='Flag enables shuffling of odi, value indicates id of this shuffle')
parser.add_argument('-sw', '--localswapodi', type=int, default=None, help='Flag enables local swapping of odi, value indicates id of this shuffle')
parser.add_argument('-d', '--swapdistance', type=int, default=None, help='Flag sets the distance over which the local swap is done')
args = parser.parse_args()


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Settings

# Path settings
settingspath = "../settings"
datapath = os.path.join("../../data/part2-planedata-od-layer2345")
datapath_maps = os.path.join("../../data/part2-responsemaps-od-layer2345")
odimapsdatapath = os.path.join("../../data/part2-odimapsdata-layer2345")
print(f"{datapath=}")

# Select mice
mice = ["O02","O03","O06","O07","O09","O10","O11","O12","O13"]

# Data settings
n_mice = len(mice)

# Volume settings
start_depth = 170
depth_increment = 10
skip_first_plane=True
include_sign = 0.05
exclude_double_xy=3 # micron
exclude_double_z=depth_increment+5 # micron
n_stimuli = 2
convert_to_micron_x = 1192/1024
convert_to_micron_y = 1019/1024

# Depth settings
n_depth_bins = 4
depth_bins = np.arange(170,531,90) # 4 bins
depth_bins[-1] = depth_bins[-1]+1

# Map settings
odimap_cell_sigma = 16
blank_low_density_regions = False
coverage_thr_nstd = 1
coverage_thr_ncells = 5


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Functions

def calc_odi_map_cells( local_XY, local_ODI, odimap_cell_sigma, max_y=None, max_x=None ):
    # Prepare a matrix that represents the image
    if max_y is None:
        max_y = np.ceil(max(local_XY[:,1])).astype(int)+1
    if max_x is None:
        max_x = np.ceil(max(local_XY[:,0])).astype(int)+1
    print("Image dims: {},{}".format(max_y,max_x))
    odi_im = np.zeros((max_y,max_x))
    coverage_im = np.zeros((max_y,max_x))

    # Loop cells and add them to the image
    round_XY = np.round(local_XY).astype(int)
    n_neurons = local_XY.shape[0]
    for n in range(n_neurons):

        # Sum ODI and coverage
        odi_im[round_XY[n,1],round_XY[n,0]] = odi_im[round_XY[n,1],round_XY[n,0]] + local_ODI[n]
        coverage_im[round_XY[n,1],round_XY[n,0]] = 1.0

    # Smooth maps
    odi_im = gaussian_filter(odi_im, sigma=odimap_cell_sigma)
    coverage_im = gaussian_filter(coverage_im, sigma=odimap_cell_sigma)

    # odi_im = odi_im / n_neurons
    odi_im[np.isnan(coverage_im)] = np.NaN
    odi_im = odi_im / coverage_im

    # Get min/max for colormap
    vmax = max(abs(np.nanmin(odi_im)),abs(np.nanmax(odi_im)))

    return odi_im, vmax, coverage_im

def swap_coords(XY, swap_distance, max_dist=50):
    XY_sct = np.zeros_like(XY)
    swap_dist_list = []
    cell_list = np.arange(XY.shape[0])

    # Calculate the distance to all other remaining cells
    D = sklearn.metrics.pairwise_distances(XY, metric="euclidean")
    np.fill_diagonal(D, 100000.0)

    max_dist_exceeded = 0
    max_dist_underceeded = 0
    not_done = True
    while not_done:

        # take a random cell
        n1 = np.random.choice(cell_list)

        # find a cell exactly the swap distance away
        n2 = np.argmin(np.abs(D[n1,:]-swap_distance))
        swap_dist_list.append(D[n1,n2])
        if (D[n1,n2]-swap_distance) > max_dist:
            max_dist_exceeded += 1
        if (D[n1,n2]-swap_distance) < -max_dist:
            max_dist_underceeded += 1

        # swap these cells
        XY_sct[n1,:] = XY[n2,:]
        XY_sct[n2,:] = XY[n1,:]

        # remove both cells from list
        cell_list = cell_list[cell_list != n1]
        cell_list = cell_list[cell_list != n2]

        # set distances from used cells to large number
        D[n1,:] = 100000.0
        D[:,n1] = 100000.0
        D[n2,:] = 100000.0
        D[:,n2] = 100000.0

        if len(cell_list) < 2:
            not_done = False
    return XY_sct,swap_dist_list,max_dist_exceeded,max_dist_underceeded

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Main processing loop

# Loop mice
for m_nr,mouse in enumerate(mice):

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load complete volume

    # Now load the cell data
    print("Loading imaging volume of mouse {}".format(mouse))
    print("  << {} >>".format(datapath))
    volume,parameter_names,aspect_ratio,_,_ = singlevolumeodfunctions.load_volume( datapath, mouse, start_depth=start_depth, depth_increment=depth_increment, skip_first_plane=skip_first_plane, convert_to_micron_x=convert_to_micron_x, convert_to_micron_y=convert_to_micron_y, include_sign=include_sign, exclude_double_xy=exclude_double_xy, exclude_double_z=exclude_double_z, invert_odi_values=False )
    print("Loaded parameters:")
    for nr,name in enumerate(parameter_names):
        print("{:>2d}: {}".format(nr,name))

    # Actual image map data, mostly for image resolution
    imagemap_datafile = np.load(os.path.join(datapath_maps,"{}-responsemaps-volume.npz".format(mouse)))
    lefteye_response_maps = imagemap_datafile["lefteye_response_maps"]
    aspect_ratio = imagemap_datafile["aspect_ratio"]
    y_res, x_res, n_planes = lefteye_response_maps.shape

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Select data

    # All data
    XY = volume[:, [parameter_names.index("x"),parameter_names.index("y")]]
    z = volume[:, parameter_names.index("z")]
    ODI = volume[:,parameter_names.index("ODI")]

    # Output containers
    odi_maps_cells = np.zeros((y_res,int(np.ceil(x_res * aspect_ratio)),n_depth_bins))

    print("n_depth_bins = {}".format(n_depth_bins))
    
    # Loop depths
    for bin_nr,(depth1,depth2) in enumerate(zip(depth_bins[:-1],depth_bins[1:])):

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Do ODI map for cells

        # Select cells in depth range
        print("Selecting cells at {} <= z < {}".format(depth1,depth2))
        XY_d = XY[np.logical_and(z>=depth1,z<depth2),:]
        ODI_d = ODI[np.logical_and(z>=depth1,z<depth2)]

        # Shuffle ODI in place if requested
        if args.shuffleodi is not None:
            print("!!! Shuffling ODI in place")
            np.random.shuffle(ODI_d)

        # Local swap if requested
        if args.localswapodi is not None:
            print("!!! swaping neuron positions by {} pixels".format(args.swapdistance))
            XY_d, swap_dist_list, n_swap_dist_exceeds, n_swap_dist_underceeds = swap_coords(XY_d, swap_distance=args.swapdistance)
            print("     Exceeded distance threshold {}x".format(n_swap_dist_exceeds))
            print("  Underceeded distance threshold {}x".format(n_swap_dist_underceeds))

        # Get ODI map based on cells, and ODI contours
        odi_im, vmax, coverage_im = calc_odi_map_cells( XY_d, ODI_d, odimap_cell_sigma=odimap_cell_sigma, max_y=y_res, max_x=int(np.ceil(x_res * aspect_ratio)) )

        # Blank low density regions
        if blank_low_density_regions:
            max_est = np.zeros( (400,400) )
            max_est[200,200] = 1.0 * coverage_thr_ncells
            max_est = gaussian_filter(max_est, sigma=odimap_cell_sigma)
            cov_thresh = max_est[200-int(np.round(odimap_cell_sigma*coverage_thr_nstd)),200]
            print("Removing ODI data for areas with low density, defined by {}*sigma coverage_im for {} cells (value>{})".format( coverage_thr_nstd, coverage_thr_ncells, cov_thresh ))
            odi_im[coverage_im<cov_thresh] = np.NaN

        # store ODI im
        odi_maps_cells[:,:,bin_nr] = odi_im

    # Save data containers for this mouse
    savename = os.path.join(odimapsdatapath, "odimaps-sigma{}-{}bins-{}".format(odimap_cell_sigma,n_depth_bins,mouse))
    if args.shuffleodi is not None:
        savename += "-shuffled_odi-{}".format(args.shuffleodi)
    if args.localswapodi is not None:
        savename += "-swapped_odi_{}-{}".format(args.swapdistance,args.localswapodi)
    savename += ".npz"
    np.savez(savename,odi_maps_cells=odi_maps_cells)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Done

print("\nDone.. that's all folks!")
