#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This script loads data of the L23-L4-L5 imaging volumes of all mice, finds clusters, calculates ODI for clusters and stores data for further analysis.

Created on Tuesday 10 May 2022

python process-layer2345-columns-across-mice.py

@author: pgoltstein
"""

# Imports
import sys, os
import numpy as np
import sklearn.metrics
from tqdm import tqdm

# Local imports
sys.path.append('../xx_analysissupport')
import densityclustering
import plottingtools
import singlevolumeodfunctions

# Module settings
plottingtools.font_size = { "title": 6, "label": 5, "tick": 5, "text": 5, "legend": 5 }

# Probably shouldn't do this, but got tired of "mean of empty slice" warnings
import warnings
warnings.filterwarnings('ignore')

# Arguments
import argparse


#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Arguments

parser = argparse.ArgumentParser( description = "This script loads data of the L23-L4-L5 imaging volumes of all mice, finds clusters, calculates ODI for clusters and stores data for further analysis.\n (written by Pieter Goltstein - May 2022)")
parser.add_argument('-d', '--ndepths', type=int, default=9, help= 'number of depths, either 4 or 9 (default=9)')
parser.add_argument('-sh', '--shuffleodi',  action="store_true", default=False, help='Flag enables shuffling of odi')
parser.add_argument('-u', '--uniformxy',  action="store_true", default=False, help='Flag enables uniform, randomly sampled xy positions')
parser.add_argument('-r', '--randomclusters',  action="store_true", default=False, help='Flag enables random xy positions')
parser.add_argument('-n', '--nrandomizations', type=int, default=100, help='manually set the number of repeats for the local or global randomization control, i.e. the swapping-neuron-locations routine (default=100).')
args = parser.parse_args()


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Settings

# Path settings
settingspath = "../settings"
savepath = "../../figureout"
datapath = os.path.join("../../data/part2-planedata-od-layer2345")
processeddatapath = os.path.join("../../data/part2-processeddata-layer2345")
print(f"{datapath=}")

# Select mice
mice = ["O02","O03","O06","O07","O09","O10","O11","O12","O13"]

# Data settings
n_mice = len(mice)

# Only relevant when not using quickload
start_depth=170
depth_increment=10
skip_first_plane=True
include_sign = 0.05
exclude_double_xy=3 # micron
exclude_double_z=depth_increment+5 # micron
convert_to_micron_x = 1192/1024
convert_to_micron_y = 1019/1024
invert_odi_values = False

# Shuffling settings
if args.shuffleodi or args.uniformxy or args.randomclusters:
    shuffle_data = True
    n_shuffles = args.nrandomizations
    rand_clust_margin = 200
else:
    shuffle_data = False

# Cluster settings
z_L4_min = 370
z_L4_max = 431
rho_min = 0.2
delta_min = 0.2
rho_x_delta_min = None
max_n_clusters = 3

# Depth settings
if args.ndepths == 9:
    depth_bins = [170,210,250,290,330,370,410,450,490,531]
elif args.ndepths == 4:
    depth_bins = [170,260,350,440,531]
n_depths = len(depth_bins)-1

# Binning and local shuffle settings
bin_size = 25
distance_range=[0,410]
n_bins = len(np.arange(distance_range[0],distance_range[1],bin_size))-1
swap_dists = list(range(0,251,50))
n_swap_dists = len(swap_dists)
n_swap_dist_iter = args.nrandomizations
calc_swap_dist_for_controls = False # if False, skips swap_dist calculation for shuffle odi, uniform xy and random clusters

# Prepare output variable
if not shuffle_data:
    n_clusters_detected = np.zeros(( n_mice, ))
    swap_dist_bin_values = np.zeros(( n_swap_dists, n_bins, n_depths, n_mice ))
    n_swap_dist_exceeds = np.zeros(( n_swap_dists, n_swap_dist_iter, n_depths, n_mice ))
    n_swap_dist_underceeds = np.zeros(( n_swap_dists, n_swap_dist_iter, n_depths, n_mice ))
else:
    n_clusters_detected = np.zeros(( n_mice, n_shuffles ))
    swap_dist_bin_values = np.zeros(( n_swap_dists, n_bins, n_depths, n_mice, n_shuffles ))
    n_swap_dist_exceeds = np.zeros(( n_swap_dists, n_swap_dist_iter, n_depths, n_mice ))
    n_swap_dist_underceeds = np.zeros(( n_swap_dists, n_swap_dist_iter, n_depths, n_mice ))


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Functions

def swap_dist_swap_coords(XY, swap_dist_distance, max_dist=50):
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

        # find a cell exactly the swap_dist distance away
        n2 = np.argmin(np.abs(D[n1,:]-swap_dist_distance))
        swap_dist_list.append(D[n1,n2])
        if (D[n1,n2]-swap_dist_distance) > max_dist:
            max_dist_exceeded += 1
        if (D[n1,n2]-swap_dist_distance) < -max_dist:
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
# Load data

# Loop mice
for m_nr,mouse in enumerate(mice):

    # Now really load the data
    print("Loading imaging volume of mouse {}".format(mouse))
    print("  << {} >>".format(datapath))
    volume,parameter_names,aspect_ratio,_,_ = singlevolumeodfunctions.load_volume( datapath, mouse, start_depth=start_depth, depth_increment=depth_increment, skip_first_plane=skip_first_plane, convert_to_micron_x=convert_to_micron_x, convert_to_micron_y=convert_to_micron_y, include_sign=include_sign, exclude_double_xy=exclude_double_xy, exclude_double_z=exclude_double_z, invert_odi_values=invert_odi_values )
    print("Loaded parameters:")
    for nr,name in enumerate(parameter_names):
        print("{:>2d}: {}".format(nr,name))
    min_z = int(np.min(volume[:,parameter_names.index("z")]))
    max_z = int(np.max(volume[:,parameter_names.index("z")]))

    # Get data for entire volumns
    XY = volume[:, [parameter_names.index("x"),parameter_names.index("y")]]
    z = volume[:, parameter_names.index("z")]
    ODI = volume[:,parameter_names.index("ODI")]
    min_y,max_y = np.min(XY[:,0]),np.max(XY[:,0])
    min_x,max_x = np.min(XY[:,1]),np.max(XY[:,1])
    n_neurons = XY.shape[0]

    # Select layer 4 data
    L4_selector = np.logical_and(z>=z_L4_min, z<z_L4_max)
    volume_L4 = volume[L4_selector,:]
    XY_L4 = volume_L4[:, [parameter_names.index("x"),parameter_names.index("y")]]
    ODI_L4 = volume_L4[:,parameter_names.index("ODI")]

    # Standard analysis including swap_dist control
    if not shuffle_data:

        # Select layer 4 data that should be clustered
        XY_L4_ipsi = XY_L4[ODI_L4<=0,:]

        # Detect layer 4 ipsi clusters
        clusters = densityclustering.find_clusters(XY_L4_ipsi, fraction=0.05, rho_min=rho_min, delta_min=delta_min, rho_x_delta_min=rho_x_delta_min, show_rho_vs_delta=False)
        print("Detected {} ipsi clusters".format(len(clusters)))

        for nr,c in enumerate(clusters):
            print("{}) {}, {}: rho={:0.3f}, delta={:0.3f}".format(nr, c["X"], c["Y"], c["rho"], c["delta"]))

        # If more than max_n_clusters, take the best ones
        n_clusters_detected[m_nr] = len(clusters)
        if len(clusters) > max_n_clusters:
            clusters = clusters[:max_n_clusters]
            print("Selected {} best ipsi clusters".format(max_n_clusters))

        # Loop depths
        for d_nr,(depth1,depth2) in enumerate(zip(depth_bins[:-1],depth_bins[1:])):
            XY_d = XY[np.logical_and(z>=depth1,z<depth2),:]
            ODI_d = ODI[np.logical_and(z>=depth1,z<depth2)]

            # Loop swap_dists and get values in matrix
            print("Calculating ODI values per shell around cluster centers for mouse {}, depth {}-{}".format(mouse,depth1,depth2))
            for nr,swap_dist in enumerate(swap_dists):
                if swap_dist > 0:
                    bin_values = np.zeros(( n_swap_dist_iter, n_bins ))
                    for it in range(n_swap_dist_iter):

                        # swap_dist XY coordinates
                        XY_sct, _, n_swap_dist_exceeds[nr,it,d_nr,m_nr], n_swap_dist_underceeds[nr,it,d_nr,m_nr] = swap_dist_swap_coords(XY_d, swap_dist_distance=swap_dist)

                        print("{}, d={}, swap_dist={}".format(mouse,d_nr,swap_dist))
                        print("     Exceeded distance threshold {}x".format(n_swap_dist_exceeds[nr,it,d_nr,m_nr]))
                        print("  Underceeded distance threshold {}x".format(n_swap_dist_underceeds[nr,it,d_nr,m_nr]))

                        bin_v,_ = densityclustering.value_per_shell(XY_sct[:,0], XY_sct[:,1], ODI_d, clusters, bin_size=bin_size, start=distance_range[0], end=distance_range[1])
                        bin_values[it,:] = np.nanmean(bin_v,axis=0)

                    bin_values = np.nanmean(bin_values,axis=0)
                else:
                    bin_values, xvalues = densityclustering.value_per_shell(XY_d[:,0], XY_d[:,1], ODI_d, clusters, bin_size=bin_size, start=distance_range[0], end=distance_range[1])
                    bin_values = np.nanmean(bin_values,axis=0)
                swap_dist_bin_values[nr,:,d_nr,m_nr] = bin_values

    else:

        # Loop ODI shuffles
        if args.shuffleodi:
            shuffle_str = "Shuffling ODI"
        if args.uniformxy:
            shuffle_str = "Sampling uniform XY"
        if args.randomclusters:
            shuffle_str = "Randomizing clusters"

        with tqdm(total=n_shuffles, desc=shuffle_str, unit="iter") as bar:
            for sh in range(n_shuffles):

                # Use randomized XY positions
                if args.uniformxy:
                    XY[:,0] = np.random.uniform( low=min_y, high=max_y, size=n_neurons )
                    XY[:,1] = np.random.uniform( low=min_x, high=max_x, size=n_neurons )

                # Select layer 4 data that should be clustered
                XY_L4_ipsi = XY_L4[ODI_L4<=0,:]

                # Detect ipsi clusters
                clusters = densityclustering.find_clusters(XY_L4_ipsi, fraction=0.05, rho_min=rho_min, delta_min=delta_min, rho_x_delta_min=rho_x_delta_min, show_rho_vs_delta=False, quiet=True)

                # If more than max_n_clusters, take the best ones
                n_clusters_detected[m_nr,sh] = len(clusters)
                if len(clusters) > max_n_clusters:
                    clusters = clusters[:max_n_clusters]

                # Make random clusters if requested
                if args.randomclusters:
                    r_y = np.random.uniform( low=min_y+rand_clust_margin, high=max_y-rand_clust_margin, size=(len(clusters),) )
                    r_x = np.random.uniform( low=min_x+rand_clust_margin, high=max_x-rand_clust_margin, size=(len(clusters),) )
                    for nr,c in enumerate(clusters):
                        clusters[nr]["Y"] = r_y[nr]
                        clusters[nr]["X"] = r_x[nr]

                # Loop depths
                for d_nr,(depth1,depth2) in enumerate(zip(depth_bins[:-1],depth_bins[1:])):
                    XY_d = XY[np.logical_and(z>=depth1,z<depth2),:]
                    ODI_d = ODI[np.logical_and(z>=depth1,z<depth2)]

                    # Shuffle ODI in place
                    if args.shuffleodi:
                        np.random.shuffle(ODI_d)

                    # Loop swap_dists and get values in matrix
                    for nr,swap_dist in enumerate(swap_dists):
                        if swap_dist > 0:
                            if calc_swap_dist_for_controls:
                                print("this code got edited out because it is a double control. It can be updated to do the cluster detection after swap and to use the swap-swap function")
                            else:
                                bin_values = np.full(( n_bins, ), np.NaN)
                        else:
                            bin_values, xvalues = densityclustering.value_per_shell(XY_d[:,0], XY_d[:,1], ODI_d, clusters, bin_size=bin_size, start=distance_range[0], end=distance_range[1])
                            bin_values = np.nanmean(bin_values,axis=0)
                        swap_dist_bin_values[nr,:,d_nr,m_nr,sh] = bin_values
                bar.update(1)

# Function to fix
def add_suffix(basename, args, extension):
    basename += "-ndepths-{:1.0f}".format(args.ndepths)
    if args.shuffleodi:
        basename += "-shuffled_odi"
    if args.uniformxy:
        basename += "-uniform_xy"
    if args.randomclusters:
        basename += "-randomized_clusters"
    savefile = basename + extension
    return savefile

# Save data
savefile = os.path.join(processeddatapath, add_suffix("column-odi-mice", args, ".npz") )
np.savez(savefile, odi_bins=swap_dist_bin_values, n_clusters=n_clusters_detected, n_swap_dist_exceeds=n_swap_dist_exceeds, n_swap_dist_underceeds=n_swap_dist_underceeds)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Done

print("\nDone.. that's all folks!")
