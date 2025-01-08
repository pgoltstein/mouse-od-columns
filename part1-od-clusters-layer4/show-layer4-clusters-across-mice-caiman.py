#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This script loads -CaImAn- processed data of a L4 clusters and analyzes ODI as function of cluster center distance, e.g. including shuffles

python show-layer4-clusters-across-mice.py b6-gcamp6s

Created on Sunday 16 Jan 2022

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


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Settings

# Path settings
savepath = "../../figureout"
processeddatapath = os.path.join("../../data/part1-processeddata-layer4")
print(f"{processeddatapath=}")

# Select mice
mice = ["O02","O03","O06","O07","O09","O10","O11","O12","O13"]
n_mice = len(mice)

# Test settings
test_bins = [0,4]
ctrl_bins = [4,8]

# Binning and local shuffle settings
n_shuffles = 100
bin_size = 25
distance_range=[0,410]
distance_bins = np.arange(distance_range[0],distance_range[1],bin_size)
xvalues = distance_bins[:-1] + (0.5*bin_size)
xvalues_fit = np.arange(distance_range[0],distance_range[1],1)
n_bins = len(distance_bins)-1
swap_dists = list(range(0,251,50))
n_swap_dists = len(swap_dists)
swap_dist_cmap = matplotlib.cm.get_cmap("hot")

# Test settings
swap_dist_nr = 0


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Load data

# Actual data
datafile = np.load(os.path.join(processeddatapath,"cluster-odi-mice-caiman.npz"))
odi_bins, n_clusters = datafile["odi_bins"], datafile["n_clusters"]

# Data with shuffled ODI
datafile_sh = np.load(os.path.join(processeddatapath,"cluster-odi-mice-caiman-shuffled_odi.npz"))
odi_bins_sh, n_clusters_sh = datafile_sh["odi_bins"], datafile_sh["n_clusters"]

# Data with uniform XY
datafile_u = np.load(os.path.join(processeddatapath,"cluster-odi-mice-caiman-uniform_xy.npz"))
odi_bins_u, n_clusters_u = datafile_u["odi_bins"], datafile_u["n_clusters"]

# Data with randomized clusters
datafile_r = np.load(os.path.join(processeddatapath,"cluster-odi-mice-caiman-randomized_clusters.npz"))
odi_bins_r, n_clusters_r = datafile_r["odi_bins"], datafile_r["n_clusters"]


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Process data

# Calculate mean across shuffles
odi_bins_sh = np.nanmean(odi_bins_sh,axis=3)
odi_bins_u = np.nanmean(odi_bins_u,axis=3)
odi_bins_r = np.nanmean(odi_bins_r,axis=3)

# Calculate mean across mice
odi_mn, odi_sem, odi_n = statstools.mean_sem( odi_bins, axis=2 )
odi_sh_mn, odi_sh_sem, odi_sh_n = statstools.mean_sem( odi_bins_sh, axis=2 )
odi_u_mn, odi_u_sem, odi_u_n = statstools.mean_sem( odi_bins_u, axis=2 )
odi_r_mn, odi_r_sem, odi_r_n = statstools.mean_sem( odi_bins_r, axis=2 )

# Calculate test and control data
odi_test = np.nanmean(odi_bins[:,test_bins[0]:test_bins[1],:],axis=1)
odi_ctrl = np.nanmean(odi_bins[:,ctrl_bins[0]:ctrl_bins[1],:],axis=1)
odi_test_mn, odi_test_sem, odi_test_n = statstools.mean_sem( odi_test, axis=1 )
odi_ctrl_mn, odi_ctrl_sem, odi_ctrl_n = statstools.mean_sem( odi_ctrl, axis=1 )

odi_test_sh = np.nanmean(odi_bins_sh[:,test_bins[0]:test_bins[1],:],axis=1)
odi_test_mn_sh, odi_test_sem_sh, odi_test_n_sh = statstools.mean_sem( odi_test_sh, axis=1 )

odi_test_u = np.nanmean(odi_bins_u[:,test_bins[0]:test_bins[1],:],axis=1)
odi_test_mn_u, odi_test_sem_u, odi_test_n_u = statstools.mean_sem( odi_test_u, axis=1 )

odi_test_r = np.nanmean(odi_bins_r[:,test_bins[0]:test_bins[1],:],axis=1)
odi_test_mn_r, odi_test_sem_r, odi_test_n_r = statstools.mean_sem( odi_test_r, axis=1 )



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Display mean ODI as function of distance to cluster center in figure with individual mice

fig,ax = plottingtools.init_figure(fig_size=(3.5,4))
for m_nr,mouse in enumerate(mice):
    plt.plot(xvalues, odi_bins[swap_dist_nr,:,m_nr], ".-", color="#999999", linewidth=0.5, markersize=1, zorder=1)
plottingtools.line( xvalues, odi_mn[swap_dist_nr,:], e=odi_sem[swap_dist_nr,:], line_color='#000000', line_width=1, sem_color=None, shaded=False )
plt.plot( [xvalues[test_bins[0]]-5,xvalues[test_bins[1]-1]+5], [0.88, 0.88], color="#000000", linewidth=1 ) # -1 to correct for list range end
plt.plot( [xvalues[ctrl_bins[0]]-5,xvalues[ctrl_bins[1]-1]+5], [0.88, 0.88], color="#000000", linewidth=1 ) # -1 to correct for list range end
plottingtools.finish_panel( ax, title="", ylabel="ODI", xlabel="Distance from cluster center (micron)", legend="off", y_minmax=[-0.4,0.8], y_step=[0.2,1], y_margin=0.1, y_axis_margin=0.1, x_minmax=[0.0,distance_range[1]+20.01], x_step=[100.0,0], x_margin=15, x_axis_margin=5 )
savefile = os.path.join(savepath, "Fig-S6c-odi-bins-indiv-mice-caiman")
plottingtools.finish_figure( filename=savefile, wspace=0.5, hspace=0.5 )


# Show figure with mean odi values of test and control range
xvalues_2bin = 0,1
fig,ax = plottingtools.init_figure(fig_size=(2.5,4))
for nr in range(odi_test.shape[1]):
    plt.plot(xvalues_2bin, [odi_test[swap_dist_nr,nr],odi_ctrl[swap_dist_nr,nr]], ".-", color="#AAAAAA", linewidth=0.5, markersize=1, zorder=1)
plottingtools.line( xvalues_2bin, [odi_test_mn[swap_dist_nr],odi_ctrl_mn[swap_dist_nr]], e=[odi_test_sem,odi_ctrl_sem], line_color='#000000', line_width=1, sem_color='#000000', shaded=False, top_bar_width=0.02 )
plottingtools.finish_panel( ax, title="", ylabel="ODI", xlabel="Position", legend="off", y_minmax=[-0.4,0.8], y_step=[0.2,1], y_margin=0.1, y_axis_margin=0.1, x_minmax=[0,1.01], x_margin=0.4, x_axis_margin=0.2, x_ticks=xvalues_2bin, x_ticklabels=["In","Out"] )
savefile = os.path.join(savepath, "Fig-S6c-odi-in-out-cluster-caiman")
plottingtools.finish_figure( filename=savefile, wspace=0.5, hspace=0.5 )

# Statistics
print("\nNormal data: test versus control range")
statstools.report_mean( odi_test[swap_dist_nr,:], odi_ctrl[swap_dist_nr,:] )
statstools.report_wmpsr_test( odi_test[swap_dist_nr,:], odi_ctrl[swap_dist_nr,:] , n_indents=0, alpha=0.05, bonferroni=1, alternative="two-sided", preceding_text="* ")


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Show figure with odi plot of all three global control conditions

fig,ax = plottingtools.init_figure(fig_size=(3.5,4))
plottingtools.line( xvalues, odi_sh_mn[swap_dist_nr,:], e=odi_sh_sem[swap_dist_nr,:], line_color='#000088', line_width=1, sem_color="#000088", shaded=True )
plottingtools.line( xvalues, odi_u_mn[swap_dist_nr,:], e=odi_u_sem[swap_dist_nr,:], line_color='#008800', line_width=1, sem_color="#008800", shaded=True )
plottingtools.line( xvalues, odi_r_mn[swap_dist_nr,:], e=odi_r_sem[swap_dist_nr,:], line_color='#880000', line_width=1, sem_color="#880000", shaded=True )
plottingtools.line( xvalues, odi_mn[swap_dist_nr,:], e=odi_sem[swap_dist_nr,:], line_color='#000000', line_width=1, sem_color=None, shaded=False )
plottingtools.finish_panel( ax, title="", ylabel="ODI", xlabel="Distance from cluster center (micron)", legend="off", y_minmax=[-0.4,0.6], y_step=[0.2,1], y_margin=0.1, y_axis_margin=0.1, x_minmax=[0.0,distance_range[1]+20.01], x_step=[100.0,0], x_margin=15, x_axis_margin=5 )
savefile = os.path.join(savepath, "Fig-S6d-odi-bins-global-controls-caiman")
plottingtools.finish_figure( filename=savefile, wspace=0.5, hspace=0.5 )

# Show figure with mean odi values of test range ('in' clusters), for real data and the different global control conditions
xvalues_4bin = 0,1,2,3
fig,ax = plottingtools.init_figure(fig_size=(3,4))
for nr in range(odi_test.shape[1]):
    plt.plot(xvalues_4bin, [odi_test[swap_dist_nr,nr],odi_test_sh[swap_dist_nr,nr],odi_test_u[swap_dist_nr,nr],odi_test_r[swap_dist_nr,nr]], ".-", color="#AAAAAA", linewidth=0.5, markersize=1, zorder=1)
plottingtools.bar( 0, odi_test_mn[swap_dist_nr], e=0, width=0.8, edge="on", bar_color='None', sem_color='#000000', label=None, bottom=0, error_width=0.5 )
plottingtools.bar( 1, odi_test_mn_sh[swap_dist_nr], e=0, width=0.8, edge="on", bar_color='None', sem_color='#000088', label=None, bottom=0, error_width=0.5 )
plottingtools.bar( 2, odi_test_mn_u[swap_dist_nr], e=0, width=0.8, edge="on", bar_color='None', sem_color='#008800', label=None, bottom=0, error_width=0.5 )
plottingtools.bar( 3, odi_test_mn_r[swap_dist_nr], e=0, width=0.8, edge="on", bar_color='None', sem_color='#880000', label=None, bottom=0, error_width=0.5 )
plottingtools.finish_panel( ax, title="", ylabel="ODI", xlabel="Condition", legend="off", y_minmax=[-0.4,0.6], y_step=[0.2,1], y_margin=0.1, y_axis_margin=0.1, x_minmax=[-0.5,3.51], x_step=[1.0,0], x_margin=0.4, x_axis_margin=0.1, x_ticks=xvalues_4bin, x_ticklabels=["D","Sh","Uni","Rnd"] )
savefile = os.path.join(savepath, "Fig-S6d-odi-in-cluster-vs-global-controls-caiman")
plottingtools.finish_figure( filename=savefile, wspace=0.5, hspace=0.5 )

# Statistics
print("\nTesting within cluster ODI for normal data and all shuffled ODI groups")
samplelist = [odi_test[swap_dist_nr,:],odi_test_sh[swap_dist_nr,:],odi_test_u[swap_dist_nr,:],odi_test_r[swap_dist_nr,:]]
statstools.report_kruskalwallis( samplelist, n_indents=2, alpha=0.05 )

print("\n  Post hoc normal data versus shuffled ODI")
statstools.report_mean( odi_test[swap_dist_nr,:], odi_test_sh[swap_dist_nr,:] )
statstools.report_wmpsr_test( odi_test[swap_dist_nr,:], odi_test_sh[swap_dist_nr,:] , n_indents=2, alpha=0.05, bonferroni=1, alternative="two-sided", preceding_text="* ")

print("\n  Post hoc cluster ODI for normal data versus uniform XY")
statstools.report_mean( odi_test[swap_dist_nr,:], odi_test_u[swap_dist_nr,:] )
statstools.report_wmpsr_test( odi_test[swap_dist_nr,:], odi_test_u[swap_dist_nr,:] , n_indents=2, alpha=0.05, bonferroni=1, alternative="two-sided", preceding_text="* ")

print("\n  Post hoc cluster ODI for normal data versus random clusters")
statstools.report_mean( odi_test[swap_dist_nr,:], odi_test_r[swap_dist_nr,:] )
statstools.report_wmpsr_test( odi_test[swap_dist_nr,:], odi_test_r[swap_dist_nr,:] , n_indents=2, alpha=0.05, bonferroni=1, alternative="two-sided", preceding_text="* ")


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Display mean ODI as function of distance to cluster center in figure with different lines for different swap_dist conditions

fig,ax = plottingtools.init_figure(fig_size=(3.5,4))
for sc_nr,swap_dist in enumerate(swap_dists):
    if swap_dist == 0:
        plt.plot(xvalues, odi_mn[sc_nr,:], ".-", color=swap_dist_cmap(sc_nr/(len(swap_dists)+1)), markersize=3, zorder=n_swap_dists-sc_nr)
    else:
        plt.plot(xvalues, odi_mn[sc_nr,:], ".:", color=swap_dist_cmap(sc_nr/(len(swap_dists)+1)), markersize=3, zorder=n_swap_dists-sc_nr)
plottingtools.finish_panel( ax, title="", ylabel="ODI", xlabel="Distance from cluster center (micron)", legend="off", y_minmax=[-0.2,0.6], y_step=[0.2,1], y_margin=0.1, y_axis_margin=0.1, x_minmax=[0.0,distance_range[1]+20.01], x_step=[100.0,0], x_margin=15, x_axis_margin=5 )
savefile = os.path.join(savepath, "Fig-S6e-odi-bins-local-controls-caiman")
plottingtools.finish_figure( filename=savefile, wspace=0.5, hspace=0.5 )

# Show figure with mean odi values of test range ('in' clusters), for real data and the different local control conditions
xvalues_sct = list(range(n_swap_dists))
fig,ax = plottingtools.init_figure(fig_size=(3,4))
for nr in range(odi_test.shape[1]):
    plt.plot(xvalues_sct, odi_test[:,nr], "-", color="#AAAAAA", linewidth=0.5, markersize=1, zorder=1)
for sc_nr,swap_dist in enumerate(swap_dists):
    plottingtools.bar( xvalues_sct[sc_nr], odi_test_mn[sc_nr], e=0, width=0.8, edge="on", bar_color='None', sem_color=swap_dist_cmap(sc_nr/(len(swap_dists)+1)), label=None, bottom=0, error_width=0.5 )
plottingtools.finish_panel( ax, title="", ylabel="ODI", xlabel="swap_dist", legend="off", y_minmax=[-0.2,0.6], y_step=[0.2,1], y_margin=0.1, y_axis_margin=0.1, x_minmax=[-0.5,n_swap_dists+0.51], x_step=[1.0,0], x_margin=0.4, x_axis_margin=0.1, x_ticks=xvalues_sct, x_ticklabels=swap_dists )
savefile = os.path.join(savepath, "Fig-S6e-odi-in-cluster-vs-local-controls-caiman")
plottingtools.finish_figure( filename=savefile, wspace=0.5, hspace=0.5 )

# Statistics
print("\nTesting within cluster ODI for normal data and all local randomization control groups")
samplelist = []
for sc_nr,swap_dist in enumerate(swap_dists):
    samplelist.append(odi_test[sc_nr,:])
statstools.report_kruskalwallis( samplelist, n_indents=2, alpha=0.05 )

print("\nPost-hoc wmpsr tests:")
for sc_nr1 in range(0,len(swap_dists)-1):
    for sc_nr2 in range(sc_nr1+1,len(swap_dists)):
        statstools.report_mean( odi_test[sc_nr1,:], odi_test[sc_nr2,:] )
        statstools.report_wmpsr_test( odi_test[sc_nr1,:], odi_test[sc_nr2,:], n_indents=2, alpha=0.05, bonferroni=1, alternative="two-sided", preceding_text="* swap_dist {} vs {}, ".format(swap_dists[sc_nr1],swap_dists[sc_nr2]))


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Display information on the number of detected clusters

print("\nNumber of clusters per mouse")
for nr,m in enumerate(mice):
    nclust_mouse = n_clusters[nr]
    print(f"{nr}) {m}: {nclust_mouse}")

print("\nMean number of clusters per condition")

def getmean(n_clust):
    mn = np.mean(n_clust)
    sd = np.std(n_clust)
    n_clust[n_clust>3] = 3
    mn_t = np.mean(n_clust)
    sd_t = np.std(n_clust)
    return mn_t, sd_t, mn, sd

print("Data: {:3.1f}+-{:3.1f} ({:3.1f}+-{:3.1f} unthresholded)".format(*getmean(n_clusters)))
print("Control, shuffle ODI: {:3.1f}+-{:3.1f} ({:3.1f}+-{:3.1f} unthresholded)".format(*getmean(n_clusters_sh)))
print("Control, uniform XY: {:3.1f}+-{:3.1f} ({:3.1f}+-{:3.1f} unthresholded)".format(*getmean(n_clusters_u)))
print("Control, randomized clusters: {:3.1f}+-{:3.1f} ({:3.1f}+-{:3.1f} unthresholded)".format(*getmean(n_clusters_r)))


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# That's all folks !!
print("\nDone.\n")
