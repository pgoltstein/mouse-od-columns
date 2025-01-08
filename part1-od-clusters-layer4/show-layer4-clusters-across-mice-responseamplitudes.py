#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This script loads data of a single L4 imaging volume of all mice, finds clusters and calculates the tuning curves showing response amplitudes for groups of cells having different distances to cluster centers

python show-layer4-clusters-across-mice-responseamplitudes.py

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
datapath_ret = os.path.join("../../data/part1-planedata-retinotopy-layer4")
print(f"{datapath=}")
print(f"{datapath_ret=}")

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

# Prepare output variables
tc_bin_values = np.zeros(( n_bins, 2, 8, n_mice ))
tc_ret_bin_values = np.zeros(( n_bins, n_mice ))


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
    volume,parameter_names,aspect_ratio,_,tuningmatrix = singlevolumeodfunctions.load_volume( datapath, mouse, start_depth=start_depth, depth_increment=depth_increment, skip_first_plane=skip_first_plane, convert_to_micron_x=convert_to_micron_x, convert_to_micron_y=convert_to_micron_y, include_sign=include_sign, exclude_double_xy=exclude_double_xy, exclude_double_z=exclude_double_z, invert_odi_values=invert_odi_values )

    print("Loading imaging volume of mouse {}".format(mouse))
    print("  << {} >>".format(datapath_ret))
    volume_ret,parameter_names_ret,aspect_ratio_ret,_,tuningmatrix_ret = singlevolumeodfunctions.load_volume( datapath_ret, mouse, start_depth=start_depth, depth_increment=depth_increment, skip_first_plane=skip_first_plane, convert_to_micron_x=convert_to_micron_x, convert_to_micron_y=convert_to_micron_y, include_sign=include_sign, exclude_double_xy=exclude_double_xy, exclude_double_z=exclude_double_z, invert_odi_values=False )

    # Get data
    XY = volume[:, [parameter_names.index("x"),parameter_names.index("y")]]
    ODI = volume[:,parameter_names.index("ODI")]
    min_y,max_y = np.min(XY[:,0]),np.max(XY[:,0])
    min_x,max_x = np.min(XY[:,1]),np.max(XY[:,1])
    n_neurons = XY.shape[0]

    # Get retinotopy data
    XY_ret = volume_ret[:, [parameter_names_ret.index("x"),parameter_names_ret.index("y")]]
    AZI = volume_ret[:,parameter_names_ret.index("Pref azim")]
    ELE = volume_ret[:,parameter_names_ret.index("Pref elev")]

    # Select data that should be clustered
    XY_ipsi = XY[ODI<=0,:]

    # Detect ipsi clusters
    clusters = densityclustering.find_clusters(XY_ipsi, fraction=0.05, rho_min=rho_min, delta_min=delta_min, rho_x_delta_min=rho_x_delta_min, show_rho_vs_delta=False)
    print("{}: Detected {} ipsi clusters".format(mouse,len(clusters)))

    # If more than max_n_clusters, take the best ones
    if len(clusters) > max_n_clusters:
        clusters = clusters[:max_n_clusters]
        print("Selected {} best ipsi clusters".format(max_n_clusters))

    # Get PD aligned tuning curves
    tc = np.nanmean(tuningmatrix,axis=3)
    n_neurons = tc.shape[0]

    tc_aligned = np.zeros((n_neurons,2,8))
    for n in range(n_neurons):
        for eye in range(2):
            pref_ix = np.argmax(tc[n,eye,:]).ravel()
            shift_ix = np.mod(np.arange(8)-2+int(pref_ix),8)
            tc_aligned[n,eye,:] = tc[n,eye,shift_ix]

    # Loop direction and eye and get values per shell
    for eye in range(2):
        for stim in range(8):
            response_per_bin,xvalues = densityclustering.value_per_shell(XY[:,0], XY[:,1], tc_aligned[:,eye,stim], clusters, bin_size=bin_size, start=distance_range[0], end=distance_range[1])
            tc_bin_values[:,eye,stim,m_nr] = np.nanmean(response_per_bin,axis=0)

    # Get PD aligned tuning curves
    tc_ret = np.nanmean(tuningmatrix_ret,axis=3)
    tc_ret_max_azi = np.nanmax(tc_ret,axis=2)
    tc_ret_max_all = np.nanmax(tc_ret_max_azi,axis=1)

    # Loop direction and eye and get values per shell
    response_per_bin,xvalues_ret = densityclustering.value_per_shell(XY_ret[:,0], XY_ret[:,1], tc_ret_max_all, clusters, bin_size=bin_size, start=distance_range[0], end=distance_range[1])
    tc_ret_bin_values[:,m_nr] = np.nanmean(response_per_bin,axis=0)


# Calculate mean across mice
mn, sem, n = statstools.mean_sem( tc_bin_values, axis=3 )

# Make figure with mean tuning curve for OD data
fig = plottingtools.init_fig(fig_size=(7.5,4.0))
eyename = ["ipsi","contra"]
xvalues = np.arange(0,316,45)
for eye in range(2):
    ax = plt.subplot(1,2,eye+1)

    for b in range(n_bins):
        plt.plot(xvalues, mn[b,eye,:], ".-", markersize=1, color=bins_cmap(b/(n_bins+1)), linewidth=0.5, zorder=1)
    plottingtools.finish_panel( ax, title="Response to {}-eye".format(eyename[eye]), ylabel="Inf. sp. (a.u.)", xlabel="Orientation from P.O.", legend="off", y_minmax=[0,40], y_step=[10,0], y_margin=0.2, y_axis_margin=0.1, x_minmax=[0,315], x_ticks=xvalues, x_ticklabels=["-90","","0","","90","","180",""], x_margin=20, x_axis_margin=5 )

# Save the figure
savefile = os.path.join( savepath, "Fig-S4e-mean-tc-from-cluster-center" )
plottingtools.finish_figure( filename=savefile, wspace=0.4, hspace=0.2 )


# Make figure with mean tuning curve for Retinotopy data
fig,ax = plottingtools.init_figure(fig_size=(4.5,5))
for m_nr,mouse in enumerate(mice):
    plt.plot(xvalues_ret, tc_ret_bin_values[:,m_nr], ".-", color="#999999", markersize=3, zorder=1)
mn = np.nanmean(tc_ret_bin_values,axis=1)
for b in range(len(mn)-1):
    plt.plot(xvalues_ret[b:b+2], mn[b:b+2], ".-", color=bins_cmap(b/len(mn)), markersize=3, zorder=2+b)
plottingtools.finish_panel( ax, title="Response to best stimulus (ret-map, binoc)", ylabel="Inf. Sp. (a.u.)", xlabel="Distance from cluster center (micron)", legend="off", y_minmax=[0,40], y_step=[10,0], y_margin=0.2, y_axis_margin=0.1, x_minmax=[0.0,distance_range[1]+50.01], x_step=[100.0,0], x_margin=15, x_axis_margin=5 )

# Save the figure
savefile = os.path.join( savepath, "Fig-S4f-max-tc-ret-from-cluster-center-color" )
plottingtools.finish_figure( filename=savefile, wspace=0.4, hspace=0.2 )


# Statistical comparisons
eyename = ["ipsi","contra"]
test_dir = 2 # pref dir
for eye in range(2):

    print("\n\n*************************************\n\nDistance bin pref ori amplitude group means --{}--, OD experiment\n\n---------------------------\n".format(eyename[eye]))

    for b_nr in range(0,n_bins):
        print("  Bin {}-{}, Mean (SEM) = {:5.3f} ({:5.3f}) n={:1.0f}".format(distance_bins[b_nr],distance_bins[b_nr+1],*statstools.mean_sem( tc_bin_values[b_nr,eye,test_dir,:].ravel())))

    print("\nWithin bin amplitude, normalized across mice")
    samplelist = []
    for b_nr in range(0,n_bins):
        samplelist.append(tc_bin_values[b_nr,eye,test_dir,:] - np.nanmean(tc_bin_values[:,eye,test_dir,:],axis=0))
    statstools.report_kruskalwallis( samplelist, n_indents=2, alpha=0.05 )

    print("\nPost-hoc wmpsr tests:")
    for b_nr1 in range(0,n_bins):
        for b_nr2 in range(b_nr1+1,n_bins):
            statstools.report_wmpsr_test( tc_bin_values[b_nr1,eye,test_dir,:], tc_bin_values[b_nr2,eye,test_dir,:], n_indents=0, alpha=0.05, bonferroni=1, alternative="two-sided", preceding_text="* Scatter {}-{} vs {}-{}, ".format(distance_bins[b_nr1],distance_bins[b_nr1+1], distance_bins[b_nr2],distance_bins[b_nr2+1]))



print("\n\n*************************************\n\nDistance bin amplitude group means, --binocular-- retinotopy experiment\n\n---------------------------\n")
for b_nr in range(0,n_bins):
    print("  Bin {}-{}, Mean (SEM) = {:5.3f} ({:5.3f}) n={:1.0f}".format(distance_bins[b_nr],distance_bins[b_nr+1],*statstools.mean_sem(tc_ret_bin_values[b_nr,:].ravel())))

print("\nWithin bin amplitude, normalized across mice")
samplelist = []
for b_nr in range(0,n_bins):
    samplelist.append(tc_ret_bin_values[b_nr,:]-np.nanmean(tc_ret_bin_values,axis=0))
statstools.report_kruskalwallis( samplelist, n_indents=2, alpha=0.05 )

print("\nPost-hoc wmpsr tests:")
for b_nr1 in range(0,n_bins):
    for b_nr2 in range(b_nr1+1,n_bins):
        statstools.report_wmpsr_test( tc_ret_bin_values[b_nr1,:], tc_ret_bin_values[b_nr2,:], n_indents=0, alpha=0.05, bonferroni=1, alternative="two-sided", preceding_text="* Scatter {}-{} vs {}-{}, ".format(distance_bins[b_nr1],distance_bins[b_nr1+1], distance_bins[b_nr2],distance_bins[b_nr2+1]))









#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# That's all folks !!
print("\nDone.\n")
