#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This script loads data of a single L4 imaging volume of all mice, finds clusters and calculates the distribution ODI as for groups of cells having  different distances to cluster centers

python show-layer4-clusters-across-mice-odihistograms.py

Created on Tuesday 17 Dec 2024

@author: pgoltstein
"""


# Imports
import sys, os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Local imports
sys.path.append('../xx_analysissupport')
import densityclustering
import plottingtools
import statstools
import singlevolumeodfunctions

# Module settings
plottingtools.font_size = { "title": 6, "label": 5, "tick": 5, "text": 5, "legend": 5 }


#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Settings

# Path settings
settingspath = "../settings"
savepath = "../../figureout"
datapath = os.path.join("../../data/part1-planedata-od-layer4")
print(f"{datapath=}")

# Select mice
mice = ["O02","O03","O06","O07","O09","O10","O11","O12","O13"]

# Data settings
n_mice = len(mice)
start_depth=370
depth_increment=20
skip_first_plane=False
include_sign = 0.05
exclude_double_xy=3 # micron
exclude_double_z=depth_increment+5 # micron

# Cluster settings
fraction = 0.05
rho_min = 0.2
delta_min = 0.2
rho_x_delta_min = None
max_n_clusters = 3

# Binning and local shuffle settings
bin_size = 25
distance_range=[0,410]
bins_cmap = matplotlib.cm.get_cmap("cool_r")
distance_bins = np.arange(distance_range[0],distance_range[1],bin_size)
n_bins = len(distance_bins)-1

# ODI binning
n_odibins = 10
odi_bins = np.linspace(-1, 1, n_odibins+1)

# Prepare output variable
odi_hist_values = np.zeros(( n_bins, n_odibins, n_mice ))


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Load data

# Loop mice
for m_nr,mouse in enumerate(mice):

    # Experiment type specific settings
    convert_to_micron_x = 1192/1024
    convert_to_micron_y = 1019/1024
    invert_odi_values = False

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

    # Select data that should be clustered
    XY_ipsi = XY[ODI<=0,:]

    # Detect ipsi clusters
    clusters = densityclustering.find_clusters(XY_ipsi, fraction=0.05, rho_min=rho_min, delta_min=delta_min, rho_x_delta_min=rho_x_delta_min, show_rho_vs_delta=False)
    print("{}: Detected {} ipsi clusters".format(mouse,len(clusters)))

    # If more than max_n_clusters, take the best ones
    if len(clusters) > max_n_clusters:
        clusters = clusters[:max_n_clusters]
        print("Selected {} best ipsi clusters".format(max_n_clusters))

    # Get per neuron a one-hot odi hist
    n_neurons = ODI.shape[0]
    onehot_odihist = np.zeros((n_neurons,n_odibins))
    for n in range(n_neurons):
        onehot_odihist[n,:], bin_edges = np.histogram(ODI[n], bins=odi_bins)

    # Loop bins in one-hot odi hists and get fraction per shell
    for odibin in range(n_odibins):
        response_per_bin, xvalues = densityclustering.value_per_shell(XY[:,0], XY[:,1], onehot_odihist[:,odibin], clusters, bin_size=bin_size, start=distance_range[0], end=distance_range[1])
        odi_hist_values[:,odibin,m_nr] = np.nanmean(response_per_bin,axis=0)

# Calculate mean across mice
mn, sem, n = statstools.mean_sem( odi_hist_values, axis=2 )

# Make figure
fig,ax = plottingtools.init_figure(fig_size=(4.5,4.0))
xvalues = (odi_bins[:-1] + odi_bins[1:]) / 2
for b in range(n_bins):
    plt.plot(xvalues, mn[b,:], ".-", markersize=1, color=bins_cmap(b/(n_bins+1)), linewidth=0.5, zorder=1)
plottingtools.finish_panel( ax, title="ODI hist", ylabel="Fraction", xlabel="ODI", legend="off", y_minmax=[0,0.245], y_step=[0.04,2], y_margin=0.005, y_axis_margin=0, x_minmax=[-1,1], x_ticks=np.linspace(-1,1,5), x_ticklabels=np.linspace(-1,1,5), x_margin=0.05, x_axis_margin=0.02 )

# Save the figure
savefile = os.path.join( savepath, "Fig-S4d-mean-odihist-from-cluster-center" )
plottingtools.finish_figure( filename=savefile, wspace=0.4, hspace=0.2 )


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# That's all folks !!
print("\nDone.\n")
