#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This script loads data of a single L4 imaging volume of all mice, finds clusters and calculates ODI for clusters, based on half of the trials

python show-layer4-clusters-across-mice-halftrials.py

Created on Tuesday 17 Dec 2024

@author: pgoltstein
"""


# Imports
import sys, os
import numpy as np
import matplotlib.pyplot as plt

# Local imports
sys.path.append('../xx_analysissupport')
import densityclustering
import analysistools
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

# Test settings
test_bins = [0,4]
ctrl_bins = [4,8]

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
n_bins = len(np.arange(distance_range[0],distance_range[1],bin_size))-1


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Load data

# Loop blocks of first 5 and last 5 trials
for halftrialset in range(2):

    # Prepare output variable
    odi_bin_values = np.zeros(( n_bins, n_mice ))

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

        # Get data
        XY = volume[:, [parameter_names.index("x"),parameter_names.index("y")]]
        n_neurons,n_eyes,n_oris,n_trials = tuningmatrix.shape
        ODI = np.zeros((n_neurons,))
        halftrials = int(np.round(n_trials/2))

        # Get ODI based on first half of trials
        if halftrialset == 0:
            print("\n !! Calculating ODI based on trials 0 to {} !! \n".format(halftrials-1)) # -1 needed as python does not include last number in slice
            for nr in range(n_neurons):
                ipsi_tc = np.mean(tuningmatrix[nr,0,:,:halftrials], axis=1)
                contra_tc = np.mean(tuningmatrix[nr,1,:,:halftrials], axis=1)
                ODI[nr] = analysistools.odi(ipsi_tc, contra_tc, method=0)

        # Get ODI based on second half of trials
        else:
            print("\n !! Calculating ODI based on trials {} to {} !! \n".format(halftrials, n_trials)) # -1 needed as python does not include last number in slice
            for nr in range(n_neurons):
                ipsi_tc = np.mean(tuningmatrix[nr,0,:,halftrials:], axis=1)
                contra_tc = np.mean(tuningmatrix[nr,1,:,halftrials:], axis=1)
                ODI[nr] = analysistools.odi(ipsi_tc, contra_tc, method=0)

        # Get data
        min_y,max_y = np.min(XY[:,0]),np.max(XY[:,0])
        min_x,max_x = np.min(XY[:,1]),np.max(XY[:,1])

        # Select data that should be clustered
        print("Calculating ODI values per shell around cluster centers for mouse {}".format(mouse))
        XY_ipsi = XY[ODI<=0,:]

        # Detect ipsi clusters
        clusters = densityclustering.find_clusters(XY_ipsi, fraction=0.05, rho_min=rho_min, delta_min=delta_min, rho_x_delta_min=rho_x_delta_min, show_rho_vs_delta=False)
        print("{}: Detected {} ipsi clusters".format(mouse,len(clusters)))

        # If more than max_n_clusters, take the best ones
        if len(clusters) > max_n_clusters:
            clusters = clusters[:max_n_clusters]
            print("Selected {} best ipsi clusters".format(max_n_clusters))

        # Get swap_dist bins
        bin_values, xvalues = densityclustering.value_per_shell(XY[:,0], XY[:,1], ODI, clusters, bin_size=bin_size, start=distance_range[0], end=distance_range[1])
        bin_values = np.nanmean(bin_values,axis=0)
        odi_bin_values[:,m_nr] = bin_values

    # Calculate mean across mice
    odi_mn, odi_sem, odi_n = statstools.mean_sem( odi_bin_values, axis=1 )

    # Calculate test and control data
    odi_test = np.nanmean(odi_bin_values[test_bins[0]:test_bins[1],:],axis=0)
    odi_ctrl = np.nanmean(odi_bin_values[ctrl_bins[0]:ctrl_bins[1],:],axis=0)
    odi_test_mn, odi_test_sem, odi_test_n = statstools.mean_sem( odi_test )
    odi_ctrl_mn, odi_ctrl_sem, odi_ctrl_n = statstools.mean_sem( odi_ctrl )

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Display mean ODI as function of distance to cluster center in figure with individual mice

    if halftrialset == 0:
        title = "Trials {}-{}".format(1,halftrials)
        figname = "Fig-S3g-"
    else:
        title = "Trials {}-{}".format(halftrials+1, n_trials)
        figname = "Fig-S3h-"

    fig,ax = plottingtools.init_figure(fig_size=(3.5,4))
    for m_nr,mouse in enumerate(mice):
        plt.plot(xvalues, odi_bin_values[:,m_nr], ".-", color="#999999", linewidth=0.5, markersize=1, zorder=1)
    plottingtools.line( xvalues, odi_mn, e=odi_sem, line_color='#000000', line_width=1, sem_color=None, shaded=False )
    plt.plot( [xvalues[test_bins[0]]-5,xvalues[test_bins[1]-1]+5], [0.5, 0.5], color="#000000", linewidth=1 ) # -1 to correct for list range end
    plt.plot( [xvalues[ctrl_bins[0]]-5,xvalues[ctrl_bins[1]-1]+5], [0.5, 0.5], color="#000000", linewidth=1 ) # -1 to correct for list range end
    plottingtools.finish_panel( ax, title=title, ylabel="ODI", xlabel="Distance from cluster center (micron)", legend="off", y_minmax=[-0.4,0.6], y_step=[0.2,1], y_margin=0.0, y_axis_margin=0.0, x_minmax=[0.0,distance_range[1]+20.01], x_step=[100.0,0], x_margin=15, x_axis_margin=5 )
    savefile = os.path.join(savepath, figname+"odi-bins-indiv-mice")
    plottingtools.finish_figure( filename=savefile, wspace=0.5, hspace=0.5 )


    # Show figure with mean odi values of test and control range
    xvalues_2bin = 0,1
    fig,ax = plottingtools.init_figure(fig_size=(2.5,4))
    for m_nr,mouse in enumerate(mice):
        plt.plot(xvalues_2bin, [odi_test[m_nr],odi_ctrl[m_nr]], ".-", color="#AAAAAA", linewidth=0.5, markersize=1, zorder=1)
    plottingtools.line( xvalues_2bin, [odi_test_mn,odi_ctrl_mn], e=[odi_test_sem,odi_ctrl_sem], line_color='#000000', line_width=1, sem_color='#000000', shaded=False, top_bar_width=0.02 )
    plottingtools.finish_panel( ax, title=title, ylabel="ODI", xlabel="Position", legend="off", y_minmax=[-0.4,0.6], y_step=[0.2,1], y_margin=0.0, y_axis_margin=0.0, x_minmax=[0,1.01], x_margin=0.4, x_axis_margin=0.2, x_ticks=xvalues_2bin, x_ticklabels=["In","Out"] )
    savefile = os.path.join(savepath, figname+"odi-in-out-cluster")
    plottingtools.finish_figure( filename=savefile, wspace=0.5, hspace=0.5 )

    # Statistics
    print("\n\n--------\nTest versus control range:")
    statstools.report_mean( odi_test, odi_ctrl )
    statstools.report_wmpsr_test( odi_test, odi_ctrl , n_indents=2, alpha=0.05, bonferroni=1, alternative="two-sided", preceding_text="* ")
    print("\n--------\n")


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# That's all folks !!
print("\nDone.\n")
