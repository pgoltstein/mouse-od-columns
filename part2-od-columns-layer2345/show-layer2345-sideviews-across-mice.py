#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This script loads processedsideview data of the L23-L4-L5 imaging volumes of all mice and displays sideviews centered on where the clusters are.

Created on Friday 15 Dec 2023

python show-layer2345-sideviews-across-mice.py

@author: pgoltstein
"""

# Global imports
import sys, os
import numpy as np
import matplotlib.pyplot as plt

# Local imports
sys.path.append('../xx_analysissupport')
import plottingtools
import statstools

# Probably shouldn't do this, but got tired of "mean of empty slice" warnings
import warnings
warnings.filterwarnings('ignore')

# Arguments
import argparse

#_______________________________________________________
# Arguments

# Arguments
parser = argparse.ArgumentParser( description = "This script loads processedsideview data of the L23-L4-L5 imaging volumes of all mice and displays sideviews centered on where the clusters are..\n (written by Pieter Goltstein - Dec 2023)")
parser.add_argument('-s', '--sideview', type=str, default="YZ", help= 'Side view orientation XZ or YZ (default=YZ)')
args = parser.parse_args()

#_______________________________________________________
# Settings
settingspath = "../settings"
savepath = "../../figureout"
processeddatapath = os.path.join("../../data/part2-sideviewdata-od-layer2345")
print(f"{processeddatapath=}")
if args.sideview == "YZ":
    figname = "Fig-S9a-"
elif args.sideview == "XZ":
    figname = "Fig-S9b-"
    
# Sideview settings
x_shift = 1200
y_shift = 1000
max_depth = 700
tst_step = 50

#_______________________________________________________
# Load data
side_view = args.sideview
loadname = os.path.join( processeddatapath, "od-sideview-{}-cluster-aligned.npz".format(side_view) )
datafile = np.load( loadname, allow_pickle=True )
sideview_per_mouse = datafile['sideview_per_mouse']
sideview_per_mouse_sh = datafile['sideview_per_mouse_sh']
aspect_ratio_od = datafile['aspect_ratio_od']
y_res,x_res,n_mice = sideview_per_mouse.shape

#_______________________________________________________
# Statistical test compared to shuffle-swap control
samplelist = []
plist = []
for y in range(0,y_res,tst_step):
    plist.append([])
    for x in range(0,x_res,tst_step):
        samplelist.append(sideview_per_mouse[y,x,:]-sideview_per_mouse_sh[y,x,:])
        p,Z,n = statstools.wilcoxon_matched_pairs_signed_rank_test( sideview_per_mouse[y,x,:], sideview_per_mouse_sh[y,x,:], alternative="two-sided" )
        plist[-1].append(p)
p_krusk = statstools.report_kruskalwallis( samplelist, n_indents=2, alpha=0.05, preceding_text="" )

#_______________________________________________________
# Show sideview of real data
mean_map = np.nanmean(sideview_per_mouse,axis=2)
fig,ax = plottingtools.init_figure(fig_size=(20*aspect_ratio_od*2,20))
im_handle = plt.imshow(mean_map, cmap="RdBu", vmin=-0.5, vmax=0.5, interpolation="None")
if side_view == "XZ":
    plt.plot([x_shift,x_shift],[50, max_depth-50],"-", linewidth=0.5, color="#000000")
if side_view == "YZ":
    plt.plot([y_shift,y_shift],[50, max_depth-50],"-", linewidth=0.5, color="#000000")
    
cbar = plt.colorbar(im_handle, fraction=0.02, pad=0.04, ticks=[-0.5,0,0.5])
cbar.set_label("ODI", size=8)
cbar.ax.tick_params(labelsize=8)
plt.title("ODI {}sideview, all mice".format(side_view), size=10)
plt.axis('off')

savename = os.path.join( savepath, figname+"od-sideview-{}-cluster-aligned-all-mice".format(side_view) )
plottingtools.finish_figure( filename=savename, wspace=0.3, hspace=0.1 )

#_______________________________________________________
# Show sideview of shuffle-swap control data
mean_map = np.nanmean(sideview_per_mouse_sh,axis=2)
fig,ax = plottingtools.init_figure(fig_size=(20*aspect_ratio_od*2,20))
im_handle = plt.imshow(mean_map, cmap="RdBu", vmin=-0.5, vmax=0.5, interpolation="None")
if side_view == "XZ":
    plt.plot([x_shift,x_shift],[50, max_depth-50],"-", linewidth=0.5, color="#000000")
if side_view == "YZ":
    plt.plot([y_shift,y_shift],[50, max_depth-50],"-", linewidth=0.5, color="#000000")
    
cbar = plt.colorbar(im_handle, fraction=0.02, pad=0.04, ticks=[-0.5,0,0.5])
cbar.set_label("ODI", size=8)
cbar.ax.tick_params(labelsize=8)
plt.title("Shuffle-swap (200um) ODI {}sideview, all mice".format(side_view), size=10)
plt.axis('off')

savename = os.path.join( savepath, figname+"od-sideview-{}-cluster-aligned-all-mice-shuffle-swap-control".format(side_view) )
plottingtools.finish_figure( filename=savename, wspace=0.3, hspace=0.1 )

#______________________________________________________________________________
# Show sideview of difference between read data and shuffle-swap control data
mean_map = np.nanmean(sideview_per_mouse-sideview_per_mouse_sh,axis=2)
fig,ax = plottingtools.init_figure(fig_size=(20*aspect_ratio_od*2,20))
im_handle = plt.imshow(mean_map, cmap="RdBu", vmin=-0.5, vmax=0.5, interpolation="None")
if side_view == "XZ":
    plt.plot([x_shift,x_shift],[50, max_depth-50],"-", linewidth=0.5, color="#000000")
if side_view == "YZ":
    plt.plot([y_shift,y_shift],[50, max_depth-50],"-", linewidth=0.5, color="#000000")
    
for y in range(len(plist)):
    for x in range(len(plist[y])):
        if plist[y][x] < 0.01:
            plt.plot( x*tst_step, y*tst_step, '*', color="#000000", linewidth=1, markersize=5)

cbar = plt.colorbar(im_handle, fraction=0.02, pad=0.04, ticks=[-0.5,0,0.5])
cbar.set_label("ODI", size=8)
cbar.ax.tick_params(labelsize=8)
plt.title("Real minus shuffle ODI {}sideview, all mice".format(side_view), size=10)
plt.axis('off')

savename = os.path.join( savepath, figname+"od-sideview-{}-cluster-aligned-all-mice-real-minus-shuffle-swap-control-stats".format(side_view) )
plottingtools.finish_figure( filename=savename, wspace=0.3, hspace=0.1 )


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# That's all folks !!
print("\nDone.\n")
