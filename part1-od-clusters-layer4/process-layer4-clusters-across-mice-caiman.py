#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This script loads data of a single -CaImAn- L4 imaging volume of all mice, finds clusters and calculates ODI for clusters

python process-layer4-clusters-across-mice-caiman.py

Created on Tuesday 17 Dec 2024

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


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Arguments

parser = argparse.ArgumentParser( description = "This script loads data of a single -CaImAn- L4 imaging volume of all mice, finds clusters and calculates ODI for clusters.\n (written by Pieter Goltstein - Dec 2024)")
parser.add_argument('-sh', '--shuffleodi',  action="store_true", default=False, help='Flag enables shuffling of odi')
parser.add_argument('-u', '--uniformxy',  action="store_true", default=False, help='Flag enables uniform, randomly sampled xy positions')
parser.add_argument('-r', '--randomclusters',  action="store_true", default=False, help='Flag enables random xy positions')
parser.add_argument('-n', '--nrandomizations', type=int, default=200, help='manually set the number of repeats for the local or global randomization control, i.e. the swapping-neuron-locations routine (default=200).')
args = parser.parse_args()


#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Settings

# Path settings
settingspath = "../settings"
savepath = "../../figureout"
datapath = os.path.join("../../data/part1-planedata-od-layer4-caiman")
processeddatapath = os.path.join("../../data/part1-processeddata-layer4")
print(f"{datapath=}")

# Select mice
mice = ["O02","O03","O06","O07","O09","O10","O11","O12","O13"]

# Experiment type specific settings
convert_to_micron_x = 1192/1024
convert_to_micron_y = 1019/1024
invert_odi_values = False

# Data settings
n_mice = len(mice)
start_depth=370
depth_increment=20
skip_first_plane=False
include_sign = 0.05
exclude_double_xy=3 # micron
exclude_double_z=depth_increment+5 # micron
if args.shuffleodi or args.uniformxy or args.randomclusters:
    shuffle_data = True
    n_shuffles = args.nrandomizations # 200
    rand_clust_margin = 200
else:
    shuffle_data = False

# Cluster settings
fraction = 0.05
rho_min = 0.2
delta_min = 0.2
rho_x_delta_min = None
max_n_clusters = 3

# Binning and local shuffle settings
bin_size = 25
distance_range=[0,410]
n_bins = len(np.arange(distance_range[0],distance_range[1],bin_size))-1
swap_dists = list(range(0,251,50))
n_swap_iters = args.nrandomizations
n_swap_dists = len(swap_dists)
calc_swap_for_controls = False # if False, skips swap calculation for shuffle odi, uniform xy and random clusters, which are anyway not used

# Prepare output variable
if not shuffle_data:
    n_clusters_detected = np.zeros(( n_mice, ))
    swap_bin_values = np.zeros(( n_swap_dists, n_bins, n_mice ))
    n_swap_dist_exceeds = np.zeros(( n_swap_dists, n_swap_iters, n_mice ))
    n_swap_dist_underceeds = np.zeros(( n_swap_dists, n_swap_iters, n_mice ))
else:
    n_clusters_detected = np.zeros(( n_mice, n_shuffles ))
    swap_bin_values = np.zeros(( n_swap_dists, n_bins, n_mice, n_shuffles ))
    n_swap_dist_exceeds = np.zeros(( n_swap_dists, n_swap_iters, n_mice ))
    n_swap_dist_underceeds = np.zeros(( n_swap_dists, n_swap_iters, n_mice ))


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Functions

def swap_swap_coords(XY, swap_distance, max_dist=50):
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
        # print("n1={}".format(n1))

        # find a cell exactly the swap distance away
        n2 = np.argmin(np.abs(D[n1,:]-swap_distance))
        # print("n2={}".format(n2))
        # print("Distance n1-n2 = {}".format(D[n1,n2]))
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
# Main >> load data

# Loop mice
for m_nr,mouse in enumerate(mice):

    # Now really load the data
    print("Loading imaging volume of mouse {}".format(mouse))
    print("  << {} >>".format(datapath))
    volume,parameter_names,aspect_ratio,_,_ = singlevolumeodfunctions.load_volume( datapath, mouse, start_depth=start_depth, depth_increment=depth_increment, skip_first_plane=skip_first_plane, convert_to_micron_x=convert_to_micron_x, convert_to_micron_y=convert_to_micron_y, include_sign=include_sign, exclude_double_xy=exclude_double_xy, exclude_double_z=exclude_double_z, invert_odi_values=invert_odi_values )

    # Get data
    XY = volume[:, [parameter_names.index("x"),parameter_names.index("y")]]
    ODI = volume[:,parameter_names.index("ODI")]
    min_y,max_y = np.min(XY[:,0]),np.max(XY[:,0])
    min_x,max_x = np.min(XY[:,1]),np.max(XY[:,1])
    n_neurons = XY.shape[0]

    if not shuffle_data:

        # Loop swap_dists and get values in matrix
        print("Calculating ODI values per shell around cluster centers for mouse {}".format(mouse))
        for nr,swap_dist in enumerate(swap_dists):
            if swap_dist > 0:
                bin_values = np.zeros(( n_swap_iters, n_bins ))
                for it in range(n_swap_iters):

                    # swap XY coordinates
                    XY_sct, swap_dist_list, n_swap_dist_exceeds[nr,it,m_nr], n_swap_dist_underceeds[nr,it,m_nr] = swap_swap_coords(XY, swap_distance=swap_dist)

                    # Select data that should be clustered
                    XY_ipsi = XY_sct[ODI<=0,:]

                    # Detect ipsi clusters
                    sct_clusters = densityclustering.find_clusters(XY_ipsi, fraction=0.05, rho_min=rho_min, delta_min=delta_min, rho_x_delta_min=rho_x_delta_min, show_rho_vs_delta=False)
                    print("{}: Detected {} ipsi clusters (swap_dist={})".format(mouse,len(sct_clusters),swap_dist))
                    print("     Exceeded distance threshold {}x".format(n_swap_dist_exceeds[nr,it,m_nr]))
                    print("  Underceeded distance threshold {}x".format(n_swap_dist_underceeds[nr,it,m_nr]))

                    # If more than max_n_clusters, take the best ones
                    # n_clusters_detected[m_nr] = len(sct_clusters)
                    if len(sct_clusters) > max_n_clusters:
                        sct_clusters = sct_clusters[:max_n_clusters]
                        print("Selected {} best ipsi clusters (swaped)".format(max_n_clusters))

                    # Get swap_dist bins
                    bin_v,_ = densityclustering.value_per_shell(XY_sct[:,0], XY_sct[:,1], ODI, sct_clusters, bin_size=bin_size, start=distance_range[0], end=distance_range[1])
                    bin_values[it,:] = np.nanmean(bin_v,axis=0)

                bin_values = np.nanmean(bin_values,axis=0)
            else:

                # Select data that should be clustered
                XY_ipsi = XY[ODI<=0,:]

                # Detect ipsi clusters
                clusters = densityclustering.find_clusters(XY_ipsi, fraction=0.05, rho_min=rho_min, delta_min=delta_min, rho_x_delta_min=rho_x_delta_min, show_rho_vs_delta=False)
                print("{}: Detected {} ipsi clusters".format(mouse,len(clusters)))

                # If more than max_n_clusters, take the best ones
                n_clusters_detected[m_nr] = len(clusters)
                if len(clusters) > max_n_clusters:
                    clusters = clusters[:max_n_clusters]
                    print("Selected {} best ipsi clusters".format(max_n_clusters))

                # Get swap_dist bins
                bin_values, xvalues = densityclustering.value_per_shell(XY[:,0], XY[:,1], ODI, clusters, bin_size=bin_size, start=distance_range[0], end=distance_range[1])
                bin_values = np.nanmean(bin_values,axis=0)
            swap_bin_values[nr,:,m_nr] = bin_values

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

                # Shuffle ODI in place
                if args.shuffleodi:
                    np.random.shuffle(ODI)

                # Use randomized XY positions
                if args.uniformxy:
                    XY[:,0] = np.random.uniform( low=min_y, high=max_y, size=n_neurons )
                    XY[:,1] = np.random.uniform( low=min_x, high=max_x, size=n_neurons )

                # Select data that should be clustered
                XY_ipsi = XY[ODI<=0,:]

                # Detect ipsi clusters
                clusters = densityclustering.find_clusters(XY_ipsi, fraction=0.05, rho_min=rho_min, delta_min=delta_min, rho_x_delta_min=rho_x_delta_min, show_rho_vs_delta=False, quiet=True)

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

                # Loop swap_dists and get values in matrix
                for nr,swap_dist in enumerate(swap_dists):
                    if swap_dist > 0:
                        if calc_swap_for_controls:
                            print("this code got edited out because it is a double control. It can be updated to do the cluster detection after swap and to use the swap-swap function")
                        else:
                            bin_values = np.full(( n_bins, ), np.NaN)
                    else:
                        bin_values, xvalues = densityclustering.value_per_shell(XY[:,0], XY[:,1], ODI, clusters, bin_size=bin_size, start=distance_range[0], end=distance_range[1])
                        bin_values = np.nanmean(bin_values,axis=0)
                    swap_bin_values[nr,:,m_nr,sh] = bin_values
                bar.update(1)

# Function to fix
def add_suffix(basename, args, extension):
    if args.shuffleodi:
        basename += "-shuffled_odi"
    if args.uniformxy:
        basename += "-uniform_xy"
    if args.randomclusters:
        basename += "-randomized_clusters"
    savefile = basename + extension
    return savefile

# Save data
savefile = os.path.join(processeddatapath, add_suffix("cluster-odi-mice-caiman", args, ".npz") )
np.savez(savefile, odi_bins=swap_bin_values, n_clusters=n_clusters_detected, n_swap_dist_exceeds=n_swap_dist_exceeds, n_swap_dist_underceeds=n_swap_dist_underceeds)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# That's all folks !!
print("\nDone.\n")
