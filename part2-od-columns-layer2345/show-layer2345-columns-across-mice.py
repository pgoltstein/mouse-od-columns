#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This script loads processed data of L2345 volumes and analyzes ODI as function of cluster center distance, e.g. including shuffles, for different depth ranges

python show-layer2345-columns-across-mice.py

Created on Tuesday 10 May 2022

@author: pgoltstein
"""

# Imports
import sys, os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Local imports
sys.path.append('../xx_analysissupport')
import plottingtools
import statstools

# Module settings
plottingtools.font_size = { "title": 6, "label": 5, "tick": 5, "text": 5, "legend": 5 }

# Probably shouldn't do this, but got tired of "mean of empty slice" warnings
import warnings
warnings.filterwarnings('ignore')

# Arguments
import argparse


#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Arguments

parser = argparse.ArgumentParser( description = "This script loads processed data of L2345 volumes and analyzes ODI as function of cluster center distance, e.g. including shuffles, for different depth ranges.\n (written by Pieter Goltstein - May 2022)")
parser.add_argument('-d', '--ndepths', type=int, default=9, help= 'number of depths, either 4 or 9 (default=9)')
args = parser.parse_args()


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Settings

# Path settings
settingspath = "../settings"
savepath = "../../figureout"
processeddatapath = os.path.join("../../data/part2-processeddata-layer2345")
print(f"{processeddatapath=}")

# Select mice
mice = ["O02","O03","O06","O07","O09","O10","O11","O12","O13"]
test_bins = [0,4]
ctrl_bins = [4,8]
n_mice = len(mice)

# Depth settings
if args.ndepths == 9:
    depth_bins = [170,210,250,290,330,370,410,450,490,531]
elif args.ndepths == 4:
    depth_bins = [170,260,350,440,531]
n_depths = len(depth_bins)-1
depth_tick_values = np.array(depth_bins)
depth_tick_values[-1] = depth_tick_values[-1] -1
depth_y_values = (depth_tick_values[1:] + depth_tick_values[:-1]) / 2

# Distance range settings
bin_size = 25
distance_range=[0,410]
distance_bins = np.arange(distance_range[0],distance_range[1],bin_size)
xvalues = distance_bins[:-1] + (0.5*bin_size)
n_bins = len(distance_bins)-1
swap_dists = list(range(0,251,50))
n_swap_dists = len(swap_dists)
swap_dist_cmap = matplotlib.cm.get_cmap("hot")

# Test settings
swap_dist_nr = 0


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Load data

# Actual data
datafile = np.load(os.path.join(processeddatapath,"column-odi-mice-ndepths-{}.npz".format(n_depths)))
odi_bins, n_clusters = datafile["odi_bins"], datafile["n_clusters"]

# Data with shuffled ODI
datafile_sh = np.load(os.path.join(processeddatapath,"column-odi-mice-ndepths-{}-shuffled_odi.npz".format(n_depths)))
odi_bins_sh, n_clusters_sh = datafile_sh["odi_bins"], datafile_sh["n_clusters"]

# Data with uniform XY
datafile_u = np.load(os.path.join(processeddatapath,"column-odi-mice-ndepths-{}-uniform_xy.npz".format(n_depths)))
odi_bins_u, n_clusters_u = datafile_u["odi_bins"], datafile_u["n_clusters"]

# Data with randomized clusters
datafile_r = np.load(os.path.join(processeddatapath,"column-odi-mice-ndepths-{}-randomized_clusters.npz".format(n_depths)))
odi_bins_r, n_clusters_r = datafile_r["odi_bins"], datafile_r["n_clusters"]

# Calculate mean across shuffles
odi_bins_sh = np.nanmean(odi_bins_sh,axis=4)
odi_bins_u = np.nanmean(odi_bins_u,axis=4)
odi_bins_r = np.nanmean(odi_bins_r,axis=4)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Plot real data, in and out of cluster, over depth along y-axis

# Calculate mean across mice
swap_dist_no = 0
odi_bin_z_test = np.nanmean(odi_bins[swap_dist_no,test_bins[0]:test_bins[1],:,:], axis=0)
odi_bin_z_ctrl = np.nanmean(odi_bins[swap_dist_no,ctrl_bins[0]:ctrl_bins[1],:,:], axis=0)
odi_bin_z_test_m,odi_bin_z_test_e,_ = statstools.mean_sem(odi_bin_z_test, axis=1)
odi_bin_z_ctrl_m,odi_bin_z_ctrl_e,_ = statstools.mean_sem(odi_bin_z_ctrl, axis=1)

# Plot the mean ODI per depth, vs control (for the no swap_dist group)
if args.ndepths == 9:
    fig,ax = plottingtools.init_figure(fig_size=(4.5,8))
    plottingtools.line_y( odi_bin_z_ctrl_m, depth_y_values, e=odi_bin_z_ctrl_e, line_color='#888888', line_width=1, sem_color="#888888", shaded=True )
    plottingtools.line_y( odi_bin_z_test_m, depth_y_values, e=odi_bin_z_test_e, line_color='#000000', line_width=1, sem_color="#000000", shaded=False )
    plottingtools.finish_panel( ax, title="", ylabel="Depth (micron)", xlabel="ODI", legend="off", y_minmax=[depth_tick_values[0],depth_tick_values[-1]+0.1], y_ticks=depth_tick_values, y_ticklabels=depth_tick_values, y_margin=10.0, y_axis_margin=5.0, x_minmax=[-0.1,0.3], x_step=[0.1,1], x_margin=0.1, x_axis_margin=0.05 )
    ax.invert_yaxis()

    # Save the figure
    savefile = os.path.join(savepath, "Fig-2c-Mean-odi-z-{}-depth-ranges".format(n_depths))
    plottingtools.finish_figure( filename=savefile, wspace=0.2, hspace=0.2 )


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Plot real data vs controls, over depth along y-axis

odi_bin_z_sh = np.nanmean(odi_bins_sh[swap_dist_no,test_bins[0]:test_bins[1],:,:], axis=0)
odi_bin_z_sh_m,odi_bin_z_sh_e,_ = statstools.mean_sem(odi_bin_z_sh, axis=1)

odi_bin_z_r = np.nanmean(odi_bins_r[swap_dist_no,test_bins[0]:test_bins[1],:,:], axis=0)
odi_bin_z_r_m,odi_bin_z_r_e,_ = statstools.mean_sem(odi_bin_z_r, axis=1)

odi_bin_z_u = np.nanmean(odi_bins_u[swap_dist_no,test_bins[0]:test_bins[1],:,:], axis=0)
odi_bin_z_u_m,odi_bin_z_u_e,_ = statstools.mean_sem(odi_bin_z_u, axis=1)

# Plot the mean ODI per depth, vs control (for the no swap_dist group)
if args.ndepths == 9:
    fig,ax = plottingtools.init_figure(fig_size=(4.5,8))
    plottingtools.line_y( odi_bin_z_sh_m, depth_y_values, e=odi_bin_z_sh_e, line_color='#000088', line_width=1, sem_color="#000088", shaded=True )
    plottingtools.line_y( odi_bin_z_u_m, depth_y_values, e=odi_bin_z_u_e, line_color='#008800', line_width=1, sem_color="#008800", shaded=True )
    plottingtools.line_y( odi_bin_z_r_m, depth_y_values, e=odi_bin_z_r_e, line_color='#880000', line_width=1, sem_color="#880000", shaded=True )
    plottingtools.line_y( odi_bin_z_test_m, depth_y_values, e=odi_bin_z_test_e, line_color='#000000', line_width=1, sem_color="#000000", shaded=False )
    plottingtools.finish_panel( ax, title="", ylabel="Depth (micron)", xlabel="ODI", legend="off", y_minmax=[depth_tick_values[0],depth_tick_values[-1]+0.1], y_ticks=depth_tick_values, y_ticklabels=depth_tick_values, y_margin=10.0, y_axis_margin=5.0, x_minmax=[-0.1,0.3], x_step=[0.1,1], x_margin=0.1, x_axis_margin=0.05 )
    ax.invert_yaxis()

    # Save the figure
    savefile = os.path.join(savepath, "Fig-2d-Mean-odi-z-controls-{}-depth-ranges".format(n_depths))
    plottingtools.finish_figure( filename=savefile, wspace=0.2, hspace=0.2 )


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Plot real data vs swap_dist controls, over depth along y-axis

# Plot the mean ODI per depth, vs swap_dist groups
if args.ndepths == 9:
    fig,ax = plottingtools.init_figure(fig_size=(4.5,8))
    for sc_nr,swap_dist in enumerate(swap_dists):
        odi_bin_z_sc = np.nanmean(odi_bins[sc_nr,test_bins[0]:test_bins[1],:,:], axis=0)
        odi_bin_z_sc_m,odi_bin_z_sc_e,_ = statstools.mean_sem(odi_bin_z_sc, axis=1)

        if swap_dist == 0:
            plottingtools.line_y( odi_bin_z_sc_m, depth_y_values, e=odi_bin_z_sc_e, line_color=swap_dist_cmap(sc_nr/(len(swap_dists)+1)), line_width=1, sem_color=swap_dist_cmap(sc_nr/(len(swap_dists)+1)), shaded=False )
        else:
            plottingtools.line_y( odi_bin_z_sc_m, depth_y_values, e=odi_bin_z_sc_e, line_color=swap_dist_cmap(sc_nr/(len(swap_dists)+1)), line_width=1, sem_color=swap_dist_cmap(sc_nr/(len(swap_dists)+1)), shaded=True )

    plottingtools.finish_panel( ax, title="", ylabel="Depth (micron)", xlabel="ODI", legend="off", y_minmax=[depth_tick_values[0],depth_tick_values[-1]+0.1], y_ticks=depth_tick_values, y_ticklabels=depth_tick_values, y_margin=10.0, y_axis_margin=5.0, x_minmax=[-0.1,0.3], x_step=[0.1,1], x_margin=0.1, x_axis_margin=0.05 )
    ax.invert_yaxis()

    # Save the figure
    savefile = os.path.join(savepath, "Fig-2e-Mean-odi-z-swap_dist-{}-depth-ranges".format(n_depths))
    plottingtools.finish_figure( filename=savefile, wspace=0.2, hspace=0.2 )


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Loop depths and create plots separately for each depth

in_data = []
out_data = []

in_sh = []
in_vs_sh = []
in_u = []
in_vs_u = []
in_r = []
in_vs_r = []

# Loop depths
for d_nr,(depth1,depth2) in enumerate(zip(depth_bins[:-1],depth_bins[1:])):
    print("Depth range {}-{}".format(depth1,depth2))

    # Calculate mean across mice
    odi_mn, odi_sem, odi_n = statstools.mean_sem( odi_bins[:,:,d_nr,:], axis=2 )
    odi_sh_mn, odi_sh_sem, odi_sh_n = statstools.mean_sem( odi_bins_sh[:,:,d_nr,:], axis=2 )
    odi_u_mn, odi_u_sem, odi_u_n = statstools.mean_sem( odi_bins_u[:,:,d_nr,:], axis=2 )
    odi_r_mn, odi_r_sem, odi_r_n = statstools.mean_sem( odi_bins_r[:,:,d_nr,:], axis=2 )

    # Calculate test and control data
    odi_test = np.nanmean(odi_bins[:,test_bins[0]:test_bins[1],d_nr,:],axis=1)
    odi_ctrl = np.nanmean(odi_bins[:,ctrl_bins[0]:ctrl_bins[1],d_nr,:],axis=1)

    odi_test_mn, odi_test_sem, odi_test_n = statstools.mean_sem( odi_test, axis=1 )
    odi_ctrl_mn, odi_ctrl_sem, odi_ctrl_n = statstools.mean_sem( odi_ctrl, axis=1 )

    odi_test_sh = np.nanmean(odi_bins_sh[:,test_bins[0]:test_bins[1],d_nr,:],axis=1)
    odi_test_mn_sh, odi_test_sem_sh, odi_test_n_sh = statstools.mean_sem( odi_test_sh, axis=1 )

    odi_test_u = np.nanmean(odi_bins_u[:,test_bins[0]:test_bins[1],d_nr,:],axis=1)
    odi_test_mn_u, odi_test_sem_u, odi_test_n_u = statstools.mean_sem( odi_test_u, axis=1 )

    odi_test_r = np.nanmean(odi_bins_r[:,test_bins[0]:test_bins[1],d_nr,:],axis=1)
    odi_test_mn_r, odi_test_sem_r, odi_test_n_r = statstools.mean_sem( odi_test_r, axis=1 )


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Display mean ODI as function of distance to cluster center in figure with individual mice

    if args.ndepths == 4:
        fig,ax = plottingtools.init_figure(fig_size=(3.5,4))
        for m_nr,mouse in enumerate(mice):
            plt.plot(xvalues, odi_bins[swap_dist_nr,:,d_nr,m_nr], ".-", color="#999999", linewidth=0.5, markersize=1, zorder=1)
        plottingtools.line( xvalues, odi_mn[swap_dist_nr,:], e=odi_sem[swap_dist_nr,:], line_color='#000000', line_width=1, sem_color=None, shaded=False )
        plt.plot( [xvalues[test_bins[0]]-5,xvalues[test_bins[1]-1]+5], [0.5, 0.5], color="#000000", linewidth=1 ) # -1 to correct for list range end
        plt.plot( [xvalues[ctrl_bins[0]]-5,xvalues[ctrl_bins[1]-1]+5], [0.5, 0.5], color="#000000", linewidth=1 ) # -1 to correct for list range end
        plottingtools.finish_panel( ax, title="", ylabel="ODI", xlabel="Distance from cluster center (micron)", legend="off", y_minmax=[-0.4,0.6], y_step=[0.2,1], y_margin=0.0, y_axis_margin=0.0, x_minmax=[0.0,distance_range[1]+20.01], x_step=[100.0,0], x_margin=15, x_axis_margin=5 )
        savefile = os.path.join(savepath, "Fig-S10-odi-bins-indiv-mice-d{}".format(d_nr))
        plottingtools.finish_figure( filename=savefile, wspace=0.5, hspace=0.5 )

    # Show figure with mean odi values of test and control range
    if args.ndepths == 4:
        xvalues_2bin = 0,1

        # Test versus control range
        fig,ax = plottingtools.init_figure(fig_size=(2.5,4))
        for nr in range(odi_test.shape[1]):
            plt.plot(xvalues_2bin, [odi_test[swap_dist_nr,nr],odi_ctrl[swap_dist_nr,nr]], ".-", color="#AAAAAA", linewidth=0.5, markersize=1, zorder=1)
        plottingtools.line( xvalues_2bin, [odi_test_mn[swap_dist_nr],odi_ctrl_mn[swap_dist_nr]], e=[odi_test_sem,odi_ctrl_sem], line_color='#000000', line_width=1, sem_color='#000000', shaded=False, top_bar_width=0.02 )
        plottingtools.finish_panel( ax, title="", ylabel="ODI", xlabel="Position", legend="off", y_minmax=[-0.2,0.4], y_step=[0.2,1], y_margin=0.0, y_axis_margin=0.0, x_minmax=[0,1.01], x_margin=0.4, x_axis_margin=0.2, x_ticks=xvalues_2bin, x_ticklabels=["In","Out"] )
        savefile = os.path.join(savepath, "Fig-S10-paired-odi-in-out-cluster-d{}".format(d_nr))
        plottingtools.finish_figure( filename=savefile, wspace=0.5, hspace=0.5 )

    # Collect data for stats
    in_data.append(odi_test[swap_dist_nr,:])
    out_data.append(odi_ctrl[swap_dist_nr,:])


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Show figure with odi plot of all three global control conditions

    if args.ndepths == 4:
        fig,ax = plottingtools.init_figure(fig_size=(3.5,4))
        plottingtools.line( xvalues, odi_sh_mn[swap_dist_nr,:], e=odi_sh_sem[swap_dist_nr,:], line_color='#000088', line_width=1, sem_color="#000088", shaded=True )
        plottingtools.line( xvalues, odi_u_mn[swap_dist_nr,:], e=odi_u_sem[swap_dist_nr,:], line_color='#008800', line_width=1, sem_color="#008800", shaded=True )
        plottingtools.line( xvalues, odi_r_mn[swap_dist_nr,:], e=odi_r_sem[swap_dist_nr,:], line_color='#880000', line_width=1, sem_color="#880000", shaded=True )
        plottingtools.line( xvalues, odi_mn[swap_dist_nr,:], e=odi_sem[swap_dist_nr,:], line_color='#000000', line_width=1, sem_color=None, shaded=False )
        plottingtools.finish_panel( ax, title="", ylabel="ODI", xlabel="Distance from cluster center (micron)", legend="off", y_minmax=[-0.4,0.6], y_step=[0.2,1], y_margin=0.0, y_axis_margin=0.0, x_minmax=[0.0,distance_range[1]+20.01], x_step=[100.0,0], x_margin=15, x_axis_margin=5 )
        savefile = os.path.join(savepath, "Fig-S10-odi-bins-incl-controls-d{}".format(d_nr))
        plottingtools.finish_figure( filename=savefile, wspace=0.5, hspace=0.5 )

    # Show figure with mean odi values of test range, for real data and the different global control conditions
    if args.ndepths == 4:
        xvalues_4bin = 0,1,2,3
        fig,ax = plottingtools.init_figure(fig_size=(3,4))
        for nr in range(odi_test.shape[1]):
            plt.plot(xvalues_4bin, [odi_test[swap_dist_nr,nr],odi_test_sh[swap_dist_nr,nr],odi_test_u[swap_dist_nr,nr],odi_test_r[swap_dist_nr,nr]], ".-", color="#AAAAAA", linewidth=0.5, markersize=1, zorder=1)
        plottingtools.bar( 0, odi_test_mn[swap_dist_nr], e=0, width=0.8, edge="on", bar_color='None', sem_color='#000000', label=None, bottom=0, error_width=0.5 )
        plottingtools.bar( 1, odi_test_mn_sh[swap_dist_nr], e=0, width=0.8, edge="on", bar_color='None', sem_color='#000088', label=None, bottom=0, error_width=0.5 )
        plottingtools.bar( 2, odi_test_mn_u[swap_dist_nr], e=0, width=0.8, edge="on", bar_color='None', sem_color='#008800', label=None, bottom=0, error_width=0.5 )
        plottingtools.bar( 3, odi_test_mn_r[swap_dist_nr], e=0, width=0.8, edge="on", bar_color='None', sem_color='#880000', label=None, bottom=0, error_width=0.5 )
        plottingtools.finish_panel( ax, title="", ylabel="ODI", xlabel="Condition", legend="off", y_minmax=[-0.4,0.6], y_step=[0.2,1], y_margin=0.0, y_axis_margin=0.0, x_minmax=[-0.5,3.51], x_step=[1.0,0], x_margin=0.4, x_axis_margin=0.1, x_ticks=xvalues_4bin, x_ticklabels=["D","Sh","Uni","Rnd"] )
        savefile = os.path.join(savepath, "Fig-S10-paired-odi-vs-controls-d{}".format(d_nr))
        plottingtools.finish_figure( filename=savefile, wspace=0.5, hspace=0.5 )

    # Collect data for stats
    in_sh.append(odi_test_sh[swap_dist_nr,:])
    in_u.append(odi_test_u[swap_dist_nr,:])
    in_r.append(odi_test_r[swap_dist_nr,:])


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Display mean ODI as function of distance to cluster center in figure with different lines for different swap_dist conditions

    if args.ndepths == 4:
        fig,ax = plottingtools.init_figure(fig_size=(3.5,4))
        for sc_nr,swap_dist in enumerate(swap_dists):
            if swap_dist == 0:
                plt.plot(xvalues, odi_mn[sc_nr,:], ".-", color=swap_dist_cmap(sc_nr/(len(swap_dists)+1)), markersize=3, zorder=n_swap_dists-sc_nr)
            else:
                plt.plot(xvalues, odi_mn[sc_nr,:], ".:", color=swap_dist_cmap(sc_nr/(len(swap_dists)+1)), markersize=3, zorder=n_swap_dists-sc_nr)
        plottingtools.finish_panel( ax, title="", ylabel="ODI", xlabel="Distance from cluster center (micron)", legend="off", y_minmax=[-0.2,0.4], y_step=[0.2,1], y_margin=0.0, y_axis_margin=0.0, x_minmax=[0.0,distance_range[1]+20.01], x_step=[100.0,0], x_margin=15, x_axis_margin=5 )
        savefile = os.path.join(savepath, "Fig-S10-odi-bins-swap_dists-d{}".format(d_nr))
        plottingtools.finish_figure( filename=savefile, wspace=0.5, hspace=0.5 )

    # Show figure with mean odi values of test range, for real data and the local controls
    if args.ndepths == 4:
        xvalues_sct = list(range(n_swap_dists))
        fig,ax = plottingtools.init_figure(fig_size=(3,4))
        for nr in range(odi_test.shape[1]):
            plt.plot(xvalues_sct, odi_test[:,nr], "-", color="#AAAAAA", linewidth=0.5, markersize=1, zorder=1)
        for sc_nr,swap_dist in enumerate(swap_dists):
            plottingtools.bar( xvalues_sct[sc_nr], odi_test_mn[sc_nr], e=0, width=0.8, edge="on", bar_color='None', sem_color=swap_dist_cmap(sc_nr/(len(swap_dists)+1)), label=None, bottom=0, error_width=0.5 )
        plottingtools.finish_panel( ax, title="", ylabel="ODI", xlabel="swap_dist", legend="off", y_minmax=[-0.2,0.4], y_step=[0.2,1], y_margin=0.0, y_axis_margin=0.0, x_minmax=[-0.5,n_swap_dists+0.51], x_step=[1.0,0], x_margin=0.4, x_axis_margin=0.1, x_ticks=xvalues_sct, x_ticklabels=swap_dists )
        savefile = os.path.join(savepath, "Fig-S10-paired-odi-vs-swap_dists-bins-d{}".format(d_nr))
        plottingtools.finish_figure( filename=savefile, wspace=0.5, hspace=0.5 )


# _____________________________________________________________________________
#    Statistics
#
# Loop depths
print("\nEach depth separately, real data vs all controls, normalized per mouse (so within subject, i.e. all data subtracted by 'real_data' / 'in_data')")
for d_nr,(depth1,depth2) in enumerate(zip(depth_bins[:-1],depth_bins[1:])):

    # Add  all data and global controls in a single list
    ctrl_names = ["Out","Shuffled ODI","Uniform XY","Random clusters"]
    real_data = in_data[d_nr]-in_data[d_nr]
    ctrl_data = []
    ctrl_data.append(out_data[d_nr]-in_data[d_nr])
    ctrl_data.append(in_sh[d_nr]-in_data[d_nr])
    ctrl_data.append(in_u[d_nr]-in_data[d_nr])
    ctrl_data.append(in_r[d_nr]-in_data[d_nr])

    #  add all swap controls
    for sc_nr,swap_dist in enumerate(swap_dists):
        if sc_nr > 0:
            sc_data = np.nanmean( odi_bins[sc_nr,test_bins[0]:test_bins[1],d_nr,:], axis=0 )
            ctrl_data.append(sc_data-in_data[d_nr])
            ctrl_names.append("Swapped at {}um".format(swap_dist))

    # Do kruskal wallis test across all data for one depth, bonferroni correct
    all_depth_data = list(ctrl_data)
    all_depth_data.append(real_data)
    p_krusk = statstools.report_kruskalwallis( all_depth_data, n_indents=0, alpha=0.05, bonferroni=len(all_depth_data), preceding_text="\nDepth {}, ".format(d_nr) )

    # Loop controls with posthoc WMPSR test
    if p_krusk < 0.05:
        for ctrl_nr, ctrl_name in enumerate(ctrl_names):
            statstools.report_wmpsr_test( real_data, ctrl_data[ctrl_nr], n_indents=1, alpha=0.05, bonferroni=1, alternative="two-sided", preceding_text="- {}: ".format(ctrl_name))
    else:
        print("** Groupwise test was not significant, no posthoc testing")


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# That's all folks !!
print("\nDone.\n")
