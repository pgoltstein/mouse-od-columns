#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This script loads analyzed crosscorrelation maps and the randomization control maps, shows the maps, and analyses peak amplitude and location

python show-cellbased-crosscorrelationmaps.py

Created on Thursday 19 May 2022

@author: pgoltstein
"""

import sys, os, glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.formula.api as smf

# Local imports
sys.path.append('../xx_analysissupport')
import plottingtools
import statstools

# Module settings
plottingtools.font_size = { "title": 6, "label": 5, "tick": 5, "text": 5, "legend": 5 }


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Settings

# Path settings
ccmapsdatapath = os.path.join("../../data/part2-ccmapsdata-layer2345")
savepath = "../../figureout"
print(f"{ccmapsdatapath=}")

# Select mice
mice = ["O02","O03","O06","O07","O09","O10","O11","O12","O13"]
n_mice = len(mice)

# ccmap settings
odi_cell_sigma = 16 # should be 25 probably
cc_map_npix = 64
map_scale = cc_map_npix / 1024
depth_bins = np.arange(170,531,90) # 4 bins
depth_names = ["L2/3 up","L2/3 low","L4","L5 up"]
n_depth_bins = len(depth_bins)-1
ref_map_nr = 2 #2=350-430 (inclusive)
margin_fraction_peak_search=0.25
inset_range = 15
crosshair_length = 16

# Crosssection settings
cross_sec_range = 5
p_thresh = 0.01
cross_plot_range = [-600,600]
cross_plot_step = [300,0]

# Plotting settings
peak_plot_range = [-600,600.1]
peak_plot_step = [300,0]
peak_plot_marker = "o"
peak_plot_marker_size = 2
peak_plot_marker_sh = "x"
peak_plot_marker_size_sh = 2


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Functions

def crop_nan_range( cc_map ):
    y_res,x_res = cc_map.shape
    x_proj = np.nanmean(cc_map,axis=0)
    y_proj = np.nanmean(cc_map,axis=1)
    x1 = int(np.argwhere(~np.isnan(x_proj))[0])
    x2 = int(np.argwhere(~np.isnan(x_proj))[-1]+1)
    y1 = int(np.argwhere(~np.isnan(y_proj))[0])
    y2 = int(np.argwhere(~np.isnan(y_proj))[-1]+1)
    cc_map = cc_map[y1:y2,x1:x2]
    return cc_map

def get_map_peak_offset( cc_map_orig, margin_fraction, map_scale ):
    cc_map = np.array(cc_map_orig)
    y_res,x_res = cc_map.shape
    ctr_x = int(np.round(x_res/2))
    ctr_y = int(np.round(y_res/2))
    x_marg = int(np.round(x_res*margin_fraction))
    y_marg = int(np.round(y_res*margin_fraction))
    cc_map[:y_marg,:] = np.NaN
    cc_map[(y_res-y_marg):,:] = np.NaN
    cc_map[:,:x_marg] = np.NaN
    cc_map[:,(x_res-x_marg):] = np.NaN
    maxlist = np.argwhere( cc_map == np.nanmax(cc_map))
    if len(maxlist) > 0:
        peak_y = (maxlist[0][0]-ctr_y)/map_scale
        peak_x = (maxlist[0][1]-ctr_x)/map_scale
        peak_y_indx = int(maxlist[0][0])
        peak_x_indx = int(maxlist[0][1])
    else:
        peak_y = np.NaN
        peak_x = np.NaN
        peak_y_indx = np.NaN
        peak_x_indx = np.NaN
    return peak_y, peak_x, peak_y_indx, peak_x_indx


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Load all crosscorrelation maps

# Real data maps
datafilename = os.path.join(ccmapsdatapath, "cc-maps-sigma{}-{}bins-size{}.npz".format( odi_cell_sigma, n_depth_bins, cc_map_npix ))
print("\nLoading main data: {}".format(datafilename))
datafile = np.load(datafilename)
crosscorr_map_mice, cc_plot_x_mice, cc_plot_y_mice, n_maps = datafile["crosscorr_map_mice"], datafile["cc_plot_x_mice"], datafile["cc_plot_y_mice"], datafile["n_maps"]
y_res,x_res,n_maps,n_mice = crosscorr_map_mice.shape
print("Crosscorr map dimensions: {}, {} (Y,X)".format(y_res,x_res))
print("# of maps (depth bins): {}".format(n_maps))
print("# of mice: {}".format(n_mice))

# Data with plane shuffled maps
datafilename = os.path.join(ccmapsdatapath, "cc-maps-sigma{}-{}bins-size{}-shuffled_planes-*.npz".format( odi_cell_sigma, n_depth_bins, cc_map_npix ))
datafilenames = glob.glob(datafilename)
n_shuffles = len(datafilenames)
crosscorr_map_mice_shp = np.zeros((y_res,x_res,n_maps,n_mice,n_shuffles))
cc_plot_x_mice_shp = np.zeros((x_res,n_maps,n_mice,n_shuffles))
cc_plot_y_mice_shp = np.zeros((y_res,n_maps,n_mice,n_shuffles))
print("\nLoading {} plane-shuffled data sets:".format(n_shuffles))
for nr,f in enumerate(datafilenames):
    print("{}) {}".format(nr,f))
    datafile = np.load(f)
    crosscorr_map_mice_shp[:,:,:,:,nr], cc_plot_x_mice_shp[:,:,:,nr], cc_plot_y_mice_shp[:,:,:,nr], n_maps = datafile["crosscorr_map_mice"], datafile["cc_plot_x_mice"], datafile["cc_plot_y_mice"], datafile["n_maps"]

# Data with odi shuffled maps
datafilename = os.path.join(ccmapsdatapath, "cc-maps-sigma{}-{}bins-size{}-shuffled_odi-*.npz".format( odi_cell_sigma, n_depth_bins, cc_map_npix ))
datafilenames = glob.glob(datafilename)
n_shuffles = len(datafilenames)
crosscorr_map_mice_sho = np.zeros((y_res,x_res,n_maps,n_mice,n_shuffles))
cc_plot_x_mice_sho = np.zeros((x_res,n_maps,n_mice,n_shuffles))
cc_plot_y_mice_sho = np.zeros((y_res,n_maps,n_mice,n_shuffles))
print("\nLoading {} odi-shuffled data sets:".format(n_shuffles))
for nr,f in enumerate(datafilenames):
    print("{}) {}".format(nr,f))
    datafile = np.load(f)
    crosscorr_map_mice_sho[:,:,:,:,nr], cc_plot_x_mice_sho[:,:,:,nr], cc_plot_y_mice_sho[:,:,:,nr], n_maps = datafile["crosscorr_map_mice"], datafile["cc_plot_x_mice"], datafile["cc_plot_y_mice"], datafile["n_maps"]

# Data with 100 um shuffled maps
datafilename = os.path.join(ccmapsdatapath, "cc-maps-sigma{}-{}bins-size{}-swapped_odi_100-*.npz".format( odi_cell_sigma, n_depth_bins, cc_map_npix ))
datafilenames = glob.glob(datafilename)
n_shuffles = len(datafilenames)
crosscorr_map_mice_100 = np.zeros((y_res,x_res,n_maps,n_mice,n_shuffles))
cc_plot_x_mice_100 = np.zeros((x_res,n_maps,n_mice,n_shuffles))
cc_plot_y_mice_100 = np.zeros((y_res,n_maps,n_mice,n_shuffles))
print("\nLoading {} 100 um swapped data sets:".format(n_shuffles))
for nr,f in enumerate(datafilenames):
    print("{}) {}".format(nr,f))
    datafile = np.load(f)
    crosscorr_map_mice_100[:,:,:,:,nr], cc_plot_x_mice_100[:,:,:,nr], cc_plot_y_mice_100[:,:,:,nr], n_maps = datafile["crosscorr_map_mice"], datafile["cc_plot_x_mice"], datafile["cc_plot_y_mice"], datafile["n_maps"]

# Data with 200 um shuffled maps
datafilename = os.path.join(ccmapsdatapath, "cc-maps-sigma{}-{}bins-size{}-swapped_odi_200-*.npz".format( odi_cell_sigma, n_depth_bins, cc_map_npix ))
datafilenames = glob.glob(datafilename)
n_shuffles = len(datafilenames)
crosscorr_map_mice_200 = np.zeros((y_res,x_res,n_maps,n_mice,n_shuffles))
cc_plot_x_mice_200 = np.zeros((x_res,n_maps,n_mice,n_shuffles))
cc_plot_y_mice_200 = np.zeros((y_res,n_maps,n_mice,n_shuffles))
print("\nLoading {} 200 um swapped data sets:".format(n_shuffles))
for nr,f in enumerate(datafilenames):
    print("{}) {}".format(nr,f))
    datafile = np.load(f)
    crosscorr_map_mice_200[:,:,:,:,nr], cc_plot_x_mice_200[:,:,:,nr], cc_plot_y_mice_200[:,:,:,nr], n_maps = datafile["crosscorr_map_mice"], datafile["cc_plot_x_mice"], datafile["cc_plot_y_mice"], datafile["n_maps"]

n_maps = int(n_maps)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Calculate peak location, amplitude for all maps

map_ctr_x = int(np.round(x_res/2))
map_ctr_y = int(np.round(y_res/2))
map_ctr_x_range = np.arange(-1*cross_sec_range,cross_sec_range)+map_ctr_x
map_ctr_y_range = np.arange(-1*cross_sec_range,cross_sec_range)+map_ctr_y

# Data containers for real data
peak_x = np.full(( n_maps, n_mice ), np.NaN)
peak_y = np.full(( n_maps, n_mice ), np.NaN)
cross_x = np.full(( n_maps, n_mice, x_res ), np.NaN)
cross_y = np.full(( n_maps, n_mice, y_res ), np.NaN)
ctr_amp = np.full(( n_maps, n_mice ), np.NaN)
peak_amp = np.full(( n_maps, n_mice ), np.NaN)
peak_displ = np.full(( n_maps, n_mice ), np.NaN)

# Loop mice, planes
for m in range(n_mice):
    for nr in range(n_maps):

        # Real data
        peak_y[nr,m], peak_x[nr,m], peak_y_indx, peak_x_indx = get_map_peak_offset( crosscorr_map_mice[:,:,nr,m], margin_fraction=margin_fraction_peak_search, map_scale=map_scale )
        ctr_amp[nr,m] = crosscorr_map_mice[map_ctr_y,map_ctr_x,nr,m]
        cross_x[nr,m,:] = np.nanmean( crosscorr_map_mice[map_ctr_y_range,:,nr,m], axis=0)
        cross_y[nr,m,:] = np.nanmean( crosscorr_map_mice[:,map_ctr_x_range,nr,m], axis=1)
        peak_amp[nr,m] = crosscorr_map_mice[peak_y_indx, peak_x_indx, nr, m]
        peak_displ[nr,m] = np.sqrt(np.power(peak_y[nr,m],2) + np.power(peak_x[nr,m],2))
        print("mouse {}, map {}, peak offset: {:4.0f}, {:4.0f} (Y,X), ctr={:0.2f}, peak={:0.2f}, displ={:4.0f}".format(m,nr, peak_y[nr,m], peak_x[nr,m], ctr_amp[nr,m], peak_amp[nr,m], peak_displ[nr,m]))

# Data containers for all shuffle controls
peak_x_shp = np.full(( n_maps, n_mice, n_shuffles ), np.NaN)
peak_y_shp = np.full(( n_maps, n_mice, n_shuffles ), np.NaN)
cross_x_shp = np.full(( n_maps, n_mice, n_shuffles, x_res ), np.NaN)
cross_y_shp = np.full(( n_maps, n_mice, n_shuffles, y_res ), np.NaN)
ctr_amp_shp = np.full(( n_maps, n_mice, n_shuffles ), np.NaN)
peak_amp_shp = np.full(( n_maps, n_mice, n_shuffles ), np.NaN)
peak_displ_shp = np.full(( n_maps, n_mice, n_shuffles ), np.NaN)

peak_x_sho = np.full(( n_maps, n_mice, n_shuffles ), np.NaN)
peak_y_sho = np.full(( n_maps, n_mice, n_shuffles ), np.NaN)
cross_x_sho = np.full(( n_maps, n_mice, n_shuffles, x_res ), np.NaN)
cross_y_sho = np.full(( n_maps, n_mice, n_shuffles, y_res ), np.NaN)
ctr_amp_sho = np.full(( n_maps, n_mice, n_shuffles ), np.NaN)
peak_amp_sho = np.full(( n_maps, n_mice, n_shuffles ), np.NaN)
peak_displ_sho = np.full(( n_maps, n_mice, n_shuffles ), np.NaN)

peak_x_100 = np.full(( n_maps, n_mice, n_shuffles ), np.NaN)
peak_y_100 = np.full(( n_maps, n_mice, n_shuffles ), np.NaN)
cross_x_100 = np.full(( n_maps, n_mice, n_shuffles, x_res ), np.NaN)
cross_y_100 = np.full(( n_maps, n_mice, n_shuffles, y_res ), np.NaN)
ctr_amp_100 = np.full(( n_maps, n_mice, n_shuffles ), np.NaN)
peak_amp_100 = np.full(( n_maps, n_mice, n_shuffles ), np.NaN)
peak_displ_100 = np.full(( n_maps, n_mice, n_shuffles ), np.NaN)

peak_x_200 = np.full(( n_maps, n_mice, n_shuffles ), np.NaN)
peak_y_200 = np.full(( n_maps, n_mice, n_shuffles ), np.NaN)
cross_x_200 = np.full(( n_maps, n_mice, n_shuffles, x_res ), np.NaN)
cross_y_200 = np.full(( n_maps, n_mice, n_shuffles, y_res ), np.NaN)
ctr_amp_200 = np.full(( n_maps, n_mice, n_shuffles ), np.NaN)
peak_amp_200 = np.full(( n_maps, n_mice, n_shuffles ), np.NaN)
peak_displ_200 = np.full(( n_maps, n_mice, n_shuffles ), np.NaN)

# Loop mice, planes and shuffles
for m in range(n_mice):
    for nr in range(n_maps):

        # Real data
        print("mouse {}, map {}".format(m,nr))

        # Loop shuffles
        for sh in range(n_shuffles):

            # Shuffled planes data
            peak_y_shp[nr,m,sh], peak_x_shp[nr,m,sh], peak_y_indx, peak_x_indx = get_map_peak_offset( crosscorr_map_mice_shp[:,:,nr,m,sh], margin_fraction=margin_fraction_peak_search, map_scale=map_scale )
            ctr_amp_shp[nr,m,sh] = crosscorr_map_mice_shp[map_ctr_y,map_ctr_x,nr,m,sh]
            cross_x_shp[nr,m,sh,:] = np.nanmean( crosscorr_map_mice_shp[map_ctr_y_range,:,nr,m,sh], axis=0)
            cross_y_shp[nr,m,sh,:] = np.nanmean( crosscorr_map_mice_shp[:,map_ctr_x_range,nr,m,sh], axis=1)
            peak_amp_shp[nr,m,sh] = crosscorr_map_mice_shp[peak_y_indx, peak_x_indx, nr, m, sh]
            peak_displ_shp[nr,m,sh] = np.sqrt(np.power(peak_y_shp[nr,m,sh],2) + np.power(peak_x_shp[nr,m,sh],2))
            print("    shuffle planes ({}), peak offset: {:4.0f}, {:4.0f} (Y,X), ctr={:0.2f}, peak={:0.2f}, displ={:4.0f}".format( sh, peak_y_shp[nr,m,sh], peak_x_shp[nr,m,sh], ctr_amp_shp[nr,m,sh], peak_amp_shp[nr,m,sh], peak_displ_shp[nr,m,sh]))

            # Shuffled ODI data
            peak_y_sho[nr,m,sh], peak_x_sho[nr,m,sh], peak_y_indx, peak_x_indx = get_map_peak_offset( crosscorr_map_mice_sho[:,:,nr,m,sh], margin_fraction=margin_fraction_peak_search, map_scale=map_scale )
            ctr_amp_sho[nr,m,sh] = crosscorr_map_mice_sho[map_ctr_y,map_ctr_x,nr,m,sh]
            cross_x_sho[nr,m,sh,:] = np.nanmean( crosscorr_map_mice_sho[map_ctr_y_range,:,nr,m,sh], axis=0)
            cross_y_sho[nr,m,sh,:] = np.nanmean( crosscorr_map_mice_sho[:,map_ctr_x_range,nr,m,sh], axis=1)
            peak_amp_sho[nr,m,sh] = crosscorr_map_mice_sho[peak_y_indx, peak_x_indx, nr, m, sh]
            peak_displ_sho[nr,m,sh] = np.sqrt(np.power(peak_y_sho[nr,m,sh],2) + np.power(peak_x_sho[nr,m,sh],2))
            print("       shuffle odi ({}), peak offset: {:4.0f}, {:4.0f} (Y,X), ctr={:0.2f}, peak={:0.2f}, displ={:4.0f}".format( sh, peak_y_sho[nr,m,sh], peak_x_sho[nr,m,sh], ctr_amp_sho[nr,m,sh], peak_amp_sho[nr,m,sh], peak_displ_sho[nr,m,sh]))

            # swapped 100 um data
            peak_y_100[nr,m,sh], peak_x_100[nr,m,sh], peak_y_indx, peak_x_indx = get_map_peak_offset( crosscorr_map_mice_100[:,:,nr,m,sh], margin_fraction=margin_fraction_peak_search, map_scale=map_scale )
            ctr_amp_100[nr,m,sh] = crosscorr_map_mice_100[map_ctr_y,map_ctr_x,nr,m,sh]
            cross_x_100[nr,m,sh,:] = np.nanmean( crosscorr_map_mice_100[map_ctr_y_range,:,nr,m,sh], axis=0)
            cross_y_100[nr,m,sh,:] = np.nanmean( crosscorr_map_mice_100[:,map_ctr_x_range,nr,m,sh], axis=1)
            peak_amp_100[nr,m,sh] = crosscorr_map_mice_100[peak_y_indx, peak_x_indx, nr, m, sh]
            peak_displ_100[nr,m,sh] = np.sqrt(np.power(peak_y_100[nr,m,sh],2) + np.power(peak_x_100[nr,m,sh],2))
            print("  swap xy 100 um({}), peak offset: {:4.0f}, {:4.0f} (Y,X), ctr={:0.2f}, peak={:0.2f}, displ={:4.0f}".format( sh, peak_y_100[nr,m,sh], peak_x_100[nr,m,sh], ctr_amp_100[nr,m,sh], peak_amp_100[nr,m,sh], peak_displ_100[nr,m,sh]))

            # swapped 200 um data
            peak_y_200[nr,m,sh], peak_x_200[nr,m,sh], peak_y_indx, peak_x_indx = get_map_peak_offset( crosscorr_map_mice_200[:,:,nr,m,sh], margin_fraction=margin_fraction_peak_search, map_scale=map_scale )
            ctr_amp_200[nr,m,sh] = crosscorr_map_mice_200[map_ctr_y,map_ctr_x,nr,m,sh]
            cross_x_200[nr,m,sh,:] = np.nanmean( crosscorr_map_mice_200[map_ctr_y_range,:,nr,m,sh], axis=0)
            cross_y_200[nr,m,sh,:] = np.nanmean( crosscorr_map_mice_200[:,map_ctr_x_range,nr,m,sh], axis=1)
            peak_amp_200[nr,m,sh] = crosscorr_map_mice_200[peak_y_indx, peak_x_indx, nr, m, sh]
            peak_displ_200[nr,m,sh] = np.sqrt(np.power(peak_y_200[nr,m,sh],2) + np.power(peak_x_200[nr,m,sh],2))
            print("  swap xy 200 um({}), peak offset: {:4.0f}, {:4.0f} (Y,X), ctr={:0.2f}, peak={:0.2f}, displ={:4.0f}".format( sh, peak_y_200[nr,m,sh], peak_x_200[nr,m,sh], ctr_amp_200[nr,m,sh], peak_amp_200[nr,m,sh], peak_displ_200[nr,m,sh]))


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Show crosscorrelation maps of Fig 2f

# Main figure settings
show_maps = [0,1,3]
map_cnt = 0
mouse_nr = 1 # Mouse M02=O03
n_example_maps = 3

# Full maps
fig = plottingtools.init_fig(fig_size=(4*(n_example_maps+1),4))
for map_nr in show_maps:
    
    # Plot map
    ax = plt.subplot2grid((1,n_example_maps+1), (0,map_cnt))
    plt.imshow(crop_nan_range(crosscorr_map_mice[:,:,map_nr,mouse_nr]), vmin=-1.0, vmax=1.0, cmap="PiYG_r")
    
    # Scalebar, title, format
    y_res_m = crop_nan_range(crosscorr_map_mice[:,:,nr,m]).shape[0]
    x_res_m = crop_nan_range(crosscorr_map_mice[:,:,nr,m]).shape[1]
    plt.plot((x_res_m,x_res_m-(200*map_scale)),(y_res_m+5,y_res_m+5),"-k", linewidth=2)
    plt.title("{} vs {}".format( depth_names[ref_map_nr], depth_names[map_nr]), fontsize=5)
    ax.axis("equal")
    plt.axis("off")
    map_cnt += 1

ax = plt.subplot2grid((1,n_example_maps+1), (0,map_cnt))
plt.imshow(np.array([[0,0],[0,0]]), vmin=-1.0, vmax=1.0, cmap="PiYG_r")
cb=plt.colorbar()
cb.outline.set_linewidth(0.5)
cb.ax.tick_params(labelsize=5,width=0.5)
plt.axis("off")

savefile = os.path.join(savepath, "Fig-2f-ccmaps-realdata")
plottingtools.finish_figure( filename=savefile, wspace=0.5, hspace=0.6 )

# Insets
map_cnt = 0
fig = plottingtools.init_fig(fig_size=(4*n_example_maps,4))
for map_nr in show_maps:
    
    # Make inset
    inset_map = crosscorr_map_mice[:,:,map_nr,mouse_nr]
    inset_map = inset_map[ map_ctr_y-inset_range:map_ctr_y+inset_range, map_ctr_x-inset_range:map_ctr_x+inset_range ]
    inset_ctr_x = int(np.round(inset_map.shape[0]/2))
    inset_ctr_y = int(np.round(inset_map.shape[1]/2))
    center_val = inset_map[inset_ctr_y,inset_ctr_x]
    map_plot_min, map_plot_max = np.nanmin(inset_map), np.nanmax(inset_map)
    inset_map = crop_nan_range(inset_map)
    
    # Plot map    
    ax = plt.subplot2grid((1,n_example_maps), (0,map_cnt))
    plt.imshow(inset_map, vmin=map_plot_min, vmax=map_plot_max, cmap="PiYG_r")

    # Crosshair
    inset_ctr_x = int(np.round(inset_map.shape[0]/2))
    inset_ctr_y = int(np.round(inset_map.shape[1]/2))
    plt.plot([inset_ctr_x - crosshair_length, inset_ctr_x + crosshair_length], [inset_ctr_y, inset_ctr_y], color='black', linewidth=0.5, linestyle='-')  # Horizontal line
    plt.plot([inset_ctr_x, inset_ctr_x], [inset_ctr_y - crosshair_length, inset_ctr_y + crosshair_length], color='black', linewidth=0.5, linestyle='-')  # Vertical line

    # Min and max as text
    x_position = inset_map.shape[1] + 2 # Just outside the right of the image
    y_bot = inset_map.shape[0] - 1      # Aligned with the top
    y_top = 0.5                           # Aligned with the bottom

    # Add the text
    plt.text(x_position, y_top, "{:0.2f}".format(map_plot_max), fontsize=5, fontname='Arial', verticalalignment='top')
    plt.text(x_position, y_bot, "{:0.2f}".format(map_plot_min), fontsize=5, fontname='Arial', verticalalignment='bottom')
    
    # Scalebar
    y_res_m = inset_map.shape[0]
    x_res_m = inset_map.shape[1]
    plt.plot((x_res_m,x_res_m-(200*map_scale)),(y_res_m+5,y_res_m+5),"-k", linewidth=2)

    # Title and axis formatting
    plt.title("{} vs {}".format( depth_names[ref_map_nr], depth_names[map_nr]), fontsize=5)
    ax.axis("equal")
    plt.axis("off")
    map_cnt += 1

savefile = os.path.join(savepath, "Fig-2f-ccmaps-realdata-insets")
plottingtools.finish_figure( filename=savefile, wspace=0.5, hspace=0.6 )


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Show crosscorrelation maps of Fig S11a

# Supplementary figure settings
show_maps = [0,1,2,3]
map_cnt = 0
mouse_nr = 5 # Mouse M06=O10
n_example_maps = 4

# Full maps
fig = plottingtools.init_fig(fig_size=(4,4*(n_example_maps+1)))
for map_nr in show_maps:
    
    # Plot map
    ax = plt.subplot2grid((n_example_maps+1,1), (map_cnt,0))
    plt.imshow(crop_nan_range(crosscorr_map_mice[:,:,map_nr,mouse_nr]), vmin=-1.0, vmax=1.0, cmap="PiYG_r")
    
    # Scalebar, title, format
    y_res_m = crop_nan_range(crosscorr_map_mice[:,:,nr,m]).shape[0]
    x_res_m = crop_nan_range(crosscorr_map_mice[:,:,nr,m]).shape[1]
    plt.plot((x_res_m,x_res_m-(200*map_scale)),(y_res_m+5,y_res_m+5),"-k", linewidth=2)
    plt.title("{} vs {}".format( depth_names[ref_map_nr], depth_names[map_nr]), fontsize=5)
    ax.axis("equal")
    plt.axis("off")
    map_cnt += 1

ax = plt.subplot2grid((n_example_maps+1,1), (map_cnt,0))
plt.imshow(np.array([[0,0],[0,0]]), vmin=-1.0, vmax=1.0, cmap="PiYG_r")
cb=plt.colorbar()
cb.outline.set_linewidth(0.5)
cb.ax.tick_params(labelsize=5,width=0.5)
plt.axis("off")

savefile = os.path.join(savepath, "Fig-S11a-ccmaps-realdata")
plottingtools.finish_figure( filename=savefile, wspace=0.5, hspace=0.6 )

# Insets
map_cnt = 0
fig = plottingtools.init_fig(fig_size=(4,4*n_example_maps))
for map_nr in show_maps:
    
    # Make inset
    inset_map = crosscorr_map_mice[:,:,map_nr,mouse_nr]
    inset_map = inset_map[ map_ctr_y-inset_range:map_ctr_y+inset_range, map_ctr_x-inset_range:map_ctr_x+inset_range ]
    inset_ctr_x = int(np.round(inset_map.shape[0]/2))
    inset_ctr_y = int(np.round(inset_map.shape[1]/2))
    center_val = inset_map[inset_ctr_y,inset_ctr_x]
    map_plot_min, map_plot_max = np.nanmin(inset_map), np.nanmax(inset_map)
    inset_map = crop_nan_range(inset_map)
    
    # Plot map    
    ax = plt.subplot2grid((n_example_maps,1), (map_cnt,0))
    plt.imshow(inset_map, vmin=map_plot_min, vmax=map_plot_max, cmap="PiYG_r")

    # Crosshair
    inset_ctr_x = int(np.round(inset_map.shape[0]/2))
    inset_ctr_y = int(np.round(inset_map.shape[1]/2))
    plt.plot([inset_ctr_x - crosshair_length, inset_ctr_x + crosshair_length], [inset_ctr_y, inset_ctr_y], color='black', linewidth=0.5, linestyle='-')  # Horizontal line
    plt.plot([inset_ctr_x, inset_ctr_x], [inset_ctr_y - crosshair_length, inset_ctr_y + crosshair_length], color='black', linewidth=0.5, linestyle='-')  # Vertical line

    # Min and max as text
    x_position = inset_map.shape[1] + 2 # Just outside the right of the image
    y_bot = inset_map.shape[0] - 1      # Aligned with the top
    y_top = 0.5                           # Aligned with the bottom

    # Add the text
    plt.text(x_position, y_top, "{:0.2f}".format(map_plot_max), fontsize=5, fontname='Arial', verticalalignment='top')
    plt.text(x_position, y_bot, "{:0.2f}".format(map_plot_min), fontsize=5, fontname='Arial', verticalalignment='bottom')
    
    # Scalebar
    y_res_m = inset_map.shape[0]
    x_res_m = inset_map.shape[1]
    plt.plot((x_res_m,x_res_m-(200*map_scale)),(y_res_m+5,y_res_m+5),"-k", linewidth=2)

    # Title and axis formatting
    plt.title("{} vs {}".format( depth_names[ref_map_nr], depth_names[map_nr]), fontsize=5)
    ax.axis("equal")
    plt.axis("off")
    map_cnt += 1

savefile = os.path.join(savepath, "Fig-S11a-ccmaps-realdata-insets")
plottingtools.finish_figure( filename=savefile, wspace=0.5, hspace=0.6 )


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Show crosscorrelation maps of Fig S11b

# Supplementary figure settings
show_mice = [0,1,2,4,3]
n_example_mice = len(show_mice)

# Full maps
fig = plottingtools.init_fig(fig_size=(4*n_maps,4*n_example_mice))
for mouse_cnt,mouse_nr in enumerate(show_mice):
    for map_nr in range(n_maps):
        
        # Plot map
        ax = plt.subplot2grid((n_example_mice,n_maps), (mouse_cnt,map_nr))
        plt.imshow(crop_nan_range(crosscorr_map_mice[:,:,map_nr,mouse_nr]), vmin=-1.0, vmax=1.0, cmap="PiYG_r")
        
        # Scalebar, title, format
        y_res_m = crop_nan_range(crosscorr_map_mice[:,:,map_nr,mouse_nr]).shape[0]
        x_res_m = crop_nan_range(crosscorr_map_mice[:,:,map_nr,mouse_nr]).shape[1]
        plt.plot((x_res_m,x_res_m-(200*map_scale)),(y_res_m+5,y_res_m+5),"-k", linewidth=2)
        plt.title("{} vs {}".format( depth_names[ref_map_nr], depth_names[map_nr]), fontsize=5)
        ax.axis("equal")
        plt.axis("off")

savefile = os.path.join(savepath, "Fig-S11b-ccmaps-realdata")
plottingtools.finish_figure( filename=savefile, wspace=0.5, hspace=0.6 )

# Insets
fig = plottingtools.init_fig(fig_size=(4*n_maps,4*n_example_mice))
for mouse_cnt,mouse_nr in enumerate(show_mice):
    for map_nr in range(n_maps):
        
        # Make inset
        inset_map = crosscorr_map_mice[:,:,map_nr,mouse_nr]
        inset_map = inset_map[ map_ctr_y-inset_range:map_ctr_y+inset_range, map_ctr_x-inset_range:map_ctr_x+inset_range ]
        inset_ctr_x = int(np.round(inset_map.shape[0]/2))
        inset_ctr_y = int(np.round(inset_map.shape[1]/2))
        center_val = inset_map[inset_ctr_y,inset_ctr_x]
        map_plot_min, map_plot_max = np.nanmin(inset_map), np.nanmax(inset_map)
        inset_map = crop_nan_range(inset_map)
        
        # Plot map    
        ax = plt.subplot2grid((n_example_mice,n_maps), (mouse_cnt,map_nr))
        plt.imshow(inset_map, vmin=map_plot_min, vmax=map_plot_max, cmap="PiYG_r")

        # Crosshair
        inset_ctr_x = int(np.round(inset_map.shape[0]/2))
        inset_ctr_y = int(np.round(inset_map.shape[1]/2))
        plt.plot([inset_ctr_x - crosshair_length, inset_ctr_x + crosshair_length], [inset_ctr_y, inset_ctr_y], color='black', linewidth=0.5, linestyle='-')  # Horizontal line
        plt.plot([inset_ctr_x, inset_ctr_x], [inset_ctr_y - crosshair_length, inset_ctr_y + crosshair_length], color='black', linewidth=0.5, linestyle='-')  # Vertical line

        # Min and max as text
        x_position = inset_map.shape[1] + 2 # Just outside the right of the image
        y_bot = inset_map.shape[0] - 1      # Aligned with the top
        y_top = 0.5                           # Aligned with the bottom

        # Add the text
        plt.text(x_position, y_top, "{:0.2f}".format(map_plot_max), fontsize=5, fontname='Arial', verticalalignment='top')
        plt.text(x_position, y_bot, "{:0.2f}".format(map_plot_min), fontsize=5, fontname='Arial', verticalalignment='bottom')
        
        # Scalebar
        y_res_m = inset_map.shape[0]
        x_res_m = inset_map.shape[1]
        plt.plot((x_res_m,x_res_m-(200*map_scale)),(y_res_m+5,y_res_m+5),"-k", linewidth=2)

        # Title and axis formatting
        plt.title("{} vs {}".format( depth_names[ref_map_nr], depth_names[map_nr]), fontsize=5)
        ax.axis("equal")
        plt.axis("off")
        map_cnt += 1

savefile = os.path.join(savepath, "Fig-S11b-ccmaps-realdata-insets")
plottingtools.finish_figure( filename=savefile, wspace=0.5, hspace=0.6 )



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Fig 2h,i Show crosscorrelation cross-sections

def show_linear_mixed_effects_model_crosssections(cross_real, cross_sh200, cross_sh_odi, crosssection_name, cross_plot_range, cross_plot_step, data_range, alpha=0.01):

    # Show & test these maps
    crosssec_maps = [0,1,3]
    n_crosssec_maps = len(crosssec_maps)

    # Get mean global shuffle data per mouse (so biological replicates)
    cross_sh_odi_m = np.nanmean(cross_sh_odi,axis=2)

    # Get mean local shuffle data per mouse (so biological replicates)
    cross_sh200_m = np.nanmean(cross_sh200,axis=2)

    # Data subselect
    cross_ = cross_real[:,:,data_range[0]:data_range[1]]
    cross_sh200_m = cross_sh200_m[:,:,data_range[0]:data_range[1]]
    cross_sh_odi_m = cross_sh_odi_m[:,:,data_range[0]:data_range[1]]
    cross_res = cross_.shape[2]

    # Make overall figure and loop map-levels
    print("\n---------")
    fig = plottingtools.init_fig(fig_size=(6*n_crosssec_maps,4))
    for cnt,nr in enumerate(crosssec_maps):
        print("\n\n ************************************************************ \n\n{}:\n\nMap {} vs {}\n".format( crosssection_name, depth_names[ref_map_nr], depth_names[nr]))

        # Convert the data to a pandas DataFrame suitable for statsmodels
        subject_ids = np.repeat(np.arange(1, n_mice+1), cross_res)  # n_mice subjects, repeated x_res times
        measurements = np.tile(np.arange(cross_res), n_mice)       # x_res measurements, tiled for each subject

        # Stack the data from both conditions
        data_real = cross_[nr,:,:].flatten()
        data_sh = cross_sh200_m[nr,:,:].flatten()

        # Create the DataFrame
        df_real = pd.DataFrame({
            'Subject': subject_ids,
            'Measurement': measurements,
            'Condition': 'real',
            'Value': data_real
        })

        df_sh = pd.DataFrame({
            'Subject': subject_ids,
            'Measurement': measurements,
            'Condition': 'sh',
            'Value': data_sh
        })

        # Combine the two DataFrames
        df = pd.concat([df_real, df_sh], axis=0)

        # Convert Measurement to a categorical variable if you want to treat each measurement point separately
        df['Measurement'] = df['Measurement'].astype('category')

        # Mixed-Effects Model with an interaction between Condition and Measurement
        model = smf.mixedlm("Value ~ Condition * Measurement", df, groups=df["Subject"])
        
        # Fit the model
        result = model.fit()

        # Print the summary of the results
        print(result.summary())

        # Extract p-values for the interaction terms
        interaction_p_values = result.pvalues.filter(like="Condition[T.sh]:Measurement")

        # Identify which measurements have significant differences
        significant_measurements = interaction_p_values[interaction_p_values < alpha].index

        # Extract just the index of the measurement
        significant_indices = [int(measurement.split('Measurement[T.')[-1].strip(']')) for measurement in significant_measurements]

        print("\nIndices of measurements with significant differences between conditions:")
        print(significant_indices)

        # Make the plots
        ax = plt.subplot2grid((1,n_crosssec_maps), (0,cnt))

        # Local shuffle
        y,e,n = statstools.mean_sem(cross_sh200_m[nr,:,:],axis=0)
        x = (np.arange(cross_res)-(0.5*cross_res))/map_scale
        plottingtools.line( x, y, e=e, line_color='#888888', line_width=1, sem_color=None, shaded=True, top_bar_width=0.2 )
        
        # Global shuffle
        y,e,n = statstools.mean_sem(cross_sh_odi_m[nr,:,:],axis=0)
        plottingtools.line( x, y, e=None, line_color='#888888', line_width=1.5, sem_color=None, shaded=True, top_bar_width=0.2, linestyle=":" )

        # Real data
        y,e,n = statstools.mean_sem(cross_[nr,:,:],axis=0)
        plottingtools.line( x, y, e=e, line_color='#000000', line_width=1, sem_color=None, shaded=True, top_bar_width=0.2 )

        # Plot significant ranges
        for x in range(cross_res):
            if x in significant_indices:
                x_pos = (x-(0.5*cross_res))/map_scale
                plt.plot( [x_pos-0.5,x_pos+0.5], [0.95, 0.95], color="#000000", linewidth=1)

        # Finish plot panel nicely
        plottingtools.finish_panel( ax, title="Crosssec, {} vs {}".format( depth_names[ref_map_nr], depth_names[map_nr]), ylabel="r", xlabel="Distance from center (um)", legend="off", y_minmax=[-0.4,1], y_step=[0.2,1], y_margin=0.04, y_axis_margin=0.02, x_minmax=cross_plot_range, x_step=cross_plot_step, x_margin=50.0, x_axis_margin=10.0 )

    # Save the figure
    savefile = os.path.join(savepath, crosssection_name)
    plottingtools.finish_figure( filename=savefile, wspace=0.5, hspace=0.6 )
    
# Define data ranges for cross-section analysis
data_range_x = [int((cross_plot_range[0]*map_scale)+(0.5*x_res)), int((cross_plot_range[1]*map_scale)+(0.5*x_res))]
data_range_y = [int((cross_plot_range[0]*map_scale)+(0.5*y_res)), int((cross_plot_range[1]*map_scale)+(0.5*y_res))]

# # Do the analysis for the M-L (x) and A-P (y) crosssections
# show_linear_mixed_effects_model_crosssections(cross_real=cross_x, cross_sh200=cross_x_200, cross_sh_odi=cross_x_sho, crosssection_name="Fig-2h-crosssect-ML", cross_plot_range=cross_plot_range, cross_plot_step=cross_plot_step, data_range=data_range_x, alpha=0.01)

# show_linear_mixed_effects_model_crosssections(cross_real=cross_y, cross_sh200=cross_y_200, cross_sh_odi=cross_y_sho, crosssection_name="Fig-2i-crosssect-AP", cross_plot_range=cross_plot_range, cross_plot_step=cross_plot_step, data_range=data_range_y, alpha=0.01)



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Fig 2g show crosscorrelation peaks for real data, all mice, all planes in 2d

show_maps = [0,1,3]
n_show_maps = len(show_maps)

fig = plottingtools.init_fig(fig_size=(4*n_show_maps,4.5))
for cnt,nr in enumerate(show_maps):
    ax = plt.subplot2grid((1,n_show_maps), (0,cnt))

    # Loop shuffles
    for sh in range(n_shuffles):
        for m in range(n_mice):
            plt.plot(peak_x_100[nr,m,sh], peak_y_100[nr,m,sh], peak_plot_marker_sh, markersize=peak_plot_marker_size_sh, markeredgecolor="#888888", markerfacecolor="None")

    for m in range(n_mice):
        plt.plot(peak_x[nr,m], peak_y[nr,m], peak_plot_marker, markersize=peak_plot_marker_size, markeredgecolor="#000000", markerfacecolor="None")

    plottingtools.finish_panel( ax, title="Peak pos, {} vs {}".format( depth_names[ref_map_nr], depth_names[nr]), ylabel="Peak pos (um)", xlabel="Peak pos (um)", legend="off", y_minmax=peak_plot_range, y_step=peak_plot_step, y_margin=50.0, y_axis_margin=20.0, x_minmax=peak_plot_range, x_step=peak_plot_step, x_margin=50.0, x_axis_margin=10.0 )

savefile = os.path.join(savepath, "Fig-2g-cc-peakposition-real-vs-swap100")
plottingtools.finish_figure( filename=savefile, wspace=0.5, hspace=0.6 )


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Fig S11c show crosscorrelation peaks, L4 vs L2/3 low, for real data and all controls

# General settings
map_nr = 1

# Fig S11c, peak positions
fig = plottingtools.init_fig(fig_size=(4*5,4.5))

# Real data
ax = plt.subplot2grid((1,5), (0,0))
for m in range(n_mice):
    plt.plot(peak_x[map_nr,m], peak_y[map_nr,m], peak_plot_marker, markersize=peak_plot_marker_size, markeredgecolor="#000000", markerfacecolor="None")
plottingtools.finish_panel( ax, title="L4 vs L2/3, real data only", ylabel="Peak pos (um)", xlabel="Peak pos (um)", legend="off", y_minmax=peak_plot_range, y_step=peak_plot_step, y_margin=50.0, y_axis_margin=20.0, x_minmax=peak_plot_range, x_step=peak_plot_step, x_margin=50.0, x_axis_margin=10.0 )

# Real data vs sw 100
ax = plt.subplot2grid((1,5), (0,1))
for sh in range(n_shuffles):
    for m in range(n_mice):
        plt.plot(peak_x_100[map_nr,m,sh], peak_y_100[map_nr,m,sh], peak_plot_marker_sh, markersize=peak_plot_marker_size_sh, markeredgecolor="#888888", markerfacecolor="None")
for m in range(n_mice):
    plt.plot(peak_x[map_nr,m], peak_y[map_nr,m], peak_plot_marker, markersize=peak_plot_marker_size, markeredgecolor="#000000", markerfacecolor="None")
plottingtools.finish_panel( ax, title="L4 vs L2/3, real vs sw100", ylabel="Peak pos (um)", xlabel="Peak pos (um)", legend="off", y_minmax=peak_plot_range, y_step=peak_plot_step, y_margin=50.0, y_axis_margin=20.0, x_minmax=peak_plot_range, x_step=peak_plot_step, x_margin=50.0, x_axis_margin=10.0 )

# Real data vs sw 200
ax = plt.subplot2grid((1,5), (0,2))
for sh in range(n_shuffles):
    for m in range(n_mice):
        plt.plot(peak_x_200[map_nr,m,sh], peak_y_200[map_nr,m,sh], peak_plot_marker_sh, markersize=peak_plot_marker_size_sh, markeredgecolor="#888888", markerfacecolor="None")
for m in range(n_mice):
    plt.plot(peak_x[map_nr,m], peak_y[map_nr,m], peak_plot_marker, markersize=peak_plot_marker_size, markeredgecolor="#000000", markerfacecolor="None")
plottingtools.finish_panel( ax, title="L4 vs L2/3, real vs sw200", ylabel="Peak pos (um)", xlabel="Peak pos (um)", legend="off", y_minmax=peak_plot_range, y_step=peak_plot_step, y_margin=50.0, y_axis_margin=20.0, x_minmax=peak_plot_range, x_step=peak_plot_step, x_margin=50.0, x_axis_margin=10.0 )

# Real data vs shuffle planes
ax = plt.subplot2grid((1,5), (0,3))
for sh in range(n_shuffles):
    for m in range(n_mice):
        plt.plot(peak_x_shp[map_nr,m,sh], peak_y_shp[map_nr,m,sh], peak_plot_marker_sh, markersize=peak_plot_marker_size_sh, markeredgecolor="#888888", markerfacecolor="None")
for m in range(n_mice):
    plt.plot(peak_x[map_nr,m], peak_y[map_nr,m], peak_plot_marker, markersize=peak_plot_marker_size, markeredgecolor="#000000", markerfacecolor="None")
plottingtools.finish_panel( ax, title="L4 vs L2/3, real vs sh planes", ylabel="Peak pos (um)", xlabel="Peak pos (um)", legend="off", y_minmax=peak_plot_range, y_step=peak_plot_step, y_margin=50.0, y_axis_margin=20.0, x_minmax=peak_plot_range, x_step=peak_plot_step, x_margin=50.0, x_axis_margin=10.0 )

# Real data vs shuffle odi
ax = plt.subplot2grid((1,5), (0,4))
for sh in range(n_shuffles):
    for m in range(n_mice):
        plt.plot(peak_x_sho[map_nr,m,sh], peak_y_sho[map_nr,m,sh], peak_plot_marker_sh, markersize=peak_plot_marker_size_sh, markeredgecolor="#888888", markerfacecolor="None")
for m in range(n_mice):
    plt.plot(peak_x[map_nr,m], peak_y[map_nr,m], peak_plot_marker, markersize=peak_plot_marker_size, markeredgecolor="#000000", markerfacecolor="None")
plottingtools.finish_panel( ax, title="L4 vs L2/3, real vs sh odi", ylabel="Peak pos (um)", xlabel="Peak pos (um)", legend="off", y_minmax=peak_plot_range, y_step=peak_plot_step, y_margin=50.0, y_axis_margin=20.0, x_minmax=peak_plot_range, x_step=peak_plot_step, x_margin=50.0, x_axis_margin=10.0 )
                           
savefile = os.path.join(savepath, "Fig-S11c-cc-peakposition-L4-vs-L23")
plottingtools.finish_figure( filename=savefile, wspace=0.5, hspace=0.6 )


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Fig S11c, single mouse map examples

# Supplementary figure settings
map_nr = 1
use_shuffle_nr = 0
show_mice = [0,1]
n_example_mice = len(show_mice)
data_and_controls = [crosscorr_map_mice, crosscorr_map_mice_100[:,:,:,:,use_shuffle_nr], crosscorr_map_mice_200[:,:,:,:,use_shuffle_nr], crosscorr_map_mice_shp[:,:,:,:,use_shuffle_nr], crosscorr_map_mice_sho[:,:,:,:,use_shuffle_nr]]
data_names = ["Real", "Swap 100", "Swap 200", "Shuf planes", "Shuf ODI"]
n_data = 5

# Full maps
fig = plottingtools.init_fig(fig_size=(4*n_data,4*n_example_mice))
for mouse_cnt,mouse_nr in enumerate(show_mice):
    for d_nr in range(n_data):
        
        # Plot map
        ax = plt.subplot2grid((n_example_mice,n_data), (mouse_cnt,d_nr))
        plt.imshow(crop_nan_range(data_and_controls[d_nr][:,:,map_nr,mouse_nr]), vmin=-1.0, vmax=1.0, cmap="PiYG_r")
        
        # Scalebar, title, format
        y_res_m = crop_nan_range(data_and_controls[d_nr][:,:,map_nr,mouse_nr]).shape[0]
        x_res_m = crop_nan_range(data_and_controls[d_nr][:,:,map_nr,mouse_nr]).shape[1]
        plt.plot((x_res_m,x_res_m-(200*map_scale)),(y_res_m+5,y_res_m+5),"-k", linewidth=2)
        plt.title("{}: {} vs {}".format( data_names[d_nr], depth_names[ref_map_nr], depth_names[map_nr]), fontsize=5)
        ax.axis("equal")
        plt.axis("off")

savefile = os.path.join(savepath, "Fig-S11c-ccmaps-examples")
plottingtools.finish_figure( filename=savefile, wspace=0.5, hspace=0.6 )

# Insets
fig = plottingtools.init_fig(fig_size=(4*n_maps,4*n_example_mice))
for mouse_cnt,mouse_nr in enumerate(show_mice):
    for d_nr in range(n_data):
        
        # Make inset
        inset_map = data_and_controls[d_nr][:,:,map_nr,mouse_nr]
        inset_map = inset_map[ map_ctr_y-inset_range:map_ctr_y+inset_range, map_ctr_x-inset_range:map_ctr_x+inset_range ]
        inset_ctr_x = int(np.round(inset_map.shape[0]/2))
        inset_ctr_y = int(np.round(inset_map.shape[1]/2))
        center_val = inset_map[inset_ctr_y,inset_ctr_x]
        map_plot_min, map_plot_max = np.nanmin(inset_map), np.nanmax(inset_map)
        inset_map = crop_nan_range(inset_map)
        
        # Plot map    
        ax = plt.subplot2grid((n_example_mice,n_data), (mouse_cnt,d_nr))
        plt.imshow(inset_map, vmin=map_plot_min, vmax=map_plot_max, cmap="PiYG_r")

        # Crosshair
        inset_ctr_x = int(np.round(inset_map.shape[0]/2))
        inset_ctr_y = int(np.round(inset_map.shape[1]/2))
        plt.plot([inset_ctr_x - crosshair_length, inset_ctr_x + crosshair_length], [inset_ctr_y, inset_ctr_y], color='black', linewidth=0.5, linestyle='-')  # Horizontal line
        plt.plot([inset_ctr_x, inset_ctr_x], [inset_ctr_y - crosshair_length, inset_ctr_y + crosshair_length], color='black', linewidth=0.5, linestyle='-')  # Vertical line

        # Min and max as text
        x_position = inset_map.shape[1] + 2 # Just outside the right of the image
        y_bot = inset_map.shape[0] - 1      # Aligned with the top
        y_top = 0.5                           # Aligned with the bottom

        # Add the text
        plt.text(x_position, y_top, "{:0.2f}".format(map_plot_max), fontsize=5, fontname='Arial', verticalalignment='top')
        plt.text(x_position, y_bot, "{:0.2f}".format(map_plot_min), fontsize=5, fontname='Arial', verticalalignment='bottom')
        
        # Scalebar
        y_res_m = inset_map.shape[0]
        x_res_m = inset_map.shape[1]
        plt.plot((x_res_m,x_res_m-(200*map_scale)),(y_res_m+5,y_res_m+5),"-k", linewidth=2)

        # Title and axis formatting
        plt.title("{}: {} vs {}".format( data_names[d_nr], depth_names[ref_map_nr], depth_names[map_nr]), fontsize=5)
        ax.axis("equal")
        plt.axis("off")
        map_cnt += 1

savefile = os.path.join(savepath, "Fig-S11c-ccmaps-examples-insets")
plottingtools.finish_figure( filename=savefile, wspace=0.5, hspace=0.6 )



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Fig S11d, show figures with peak error (euclidian distance of peak to map center, i.e. peak displacement)

# Plotting variables
xvalues_5bin = np.arange(5)
show_maps = [0,1,3]
n_show_maps = len(show_maps)

# Get the mean error across shuffles
peak_displ_shp = np.nanmean(peak_displ_shp,axis=2)
peak_displ_sho = np.nanmean(peak_displ_sho,axis=2)
peak_displ_100 = np.nanmean(peak_displ_100,axis=2)
peak_displ_200 = np.nanmean(peak_displ_200,axis=2)

# Mean across mice
peak_displ_mn, peak_displ_sem, _ = statstools.mean_sem( peak_displ, axis=1 )
peak_displ_shp_mn, peak_displ_shp_sem, _ = statstools.mean_sem( peak_displ_shp, axis=1 )
peak_displ_sho_mn, peak_displ_sho_sem, _ = statstools.mean_sem( peak_displ_sho, axis=1 )
peak_displ_100_mn, peak_displ_100_sem, _ = statstools.mean_sem( peak_displ_100, axis=1 )
peak_displ_200_mn, peak_displ_200_sem, _ = statstools.mean_sem( peak_displ_200, axis=1 )

# Plot in bar chart with single mouse data
fig = plottingtools.init_fig(fig_size=(3*n_show_maps,4))
for cnt,nr in enumerate(show_maps):
    ax = plt.subplot2grid((1,int(n_show_maps)), (0,cnt))
    for m in range(n_mice):
        plt.plot(xvalues_5bin, [peak_displ[nr,m], peak_displ_100[nr,m], peak_displ_200[nr,m], peak_displ_shp[nr,m], peak_displ_sho[nr,m]], ".-", color="#AAAAAA", linewidth=0.5, markersize=1, zorder=1)
    plottingtools.bar( 0, peak_displ_mn[nr], e=0, width=0.8, edge="on", bar_color='None', sem_color='#000000', label=None, bottom=0, error_width=0.5 )
    plottingtools.bar( 1, peak_displ_100_mn[nr], e=0, width=0.8, edge="on", bar_color='None', sem_color='#888800', label=None, bottom=0, error_width=0.5 )
    plottingtools.bar( 2, peak_displ_200_mn[nr], e=0, width=0.8, edge="on", bar_color='None', sem_color='#aa6600', label=None, bottom=0, error_width=0.5 )
    plottingtools.bar( 3, peak_displ_shp_mn[nr], e=0, width=0.8, edge="on", bar_color='None', sem_color='#008888', label=None, bottom=0, error_width=0.5 )
    plottingtools.bar( 4, peak_displ_sho_mn[nr], e=0, width=0.8, edge="on", bar_color='None', sem_color='#000088', label=None, bottom=0, error_width=0.5 )
    plottingtools.finish_panel( ax, title="", ylabel="Peak error (um)", xlabel="Condition", legend="off", y_minmax=[0,1000], y_step=[250,0], y_margin=50, y_axis_margin=20, x_minmax=[-0.5,4.51], x_step=[1.0,0], x_margin=0.4, x_axis_margin=0.1, x_ticks=xvalues_5bin, x_ticklabels=["D","100","200","ShP","ShO"] )
savefile = os.path.join(savepath, "Fig-S11d-cc-peak-error-vs-controls")
plottingtools.finish_figure( filename=savefile, wspace=0.5, hspace=0.5 )



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Fig 2g, show inset figures with peak error (euclidian distance of peak to map center, i.e. peak displacement)

# Plotting variables
xvalues_2bin = 0,1
show_maps = [0,1,3]
n_show_maps = len(show_maps)

# Test versus 100um swap
fig = plottingtools.init_fig(fig_size=(2*n_show_maps,4))
for cnt,nr in enumerate(show_maps):
    ax = plt.subplot2grid((1,int(n_show_maps)), (0,cnt))
    for m in range(peak_displ.shape[1]):
        plt.plot(xvalues_2bin, [peak_displ[nr,m],peak_displ_100[nr,m]], ".-", color="#AAAAAA", linewidth=0.5, markersize=1, zorder=1)
    plottingtools.line( xvalues_2bin, [peak_displ_mn[nr],peak_displ_100_mn[nr]], e=[peak_displ_sem[nr],peak_displ_100_sem[nr]], line_color='#000000', line_width=1, sem_color='#000000', shaded=False, top_bar_width=0.02 )
    plottingtools.finish_panel( ax, title="", ylabel="ODI", xlabel="Group", legend="off", y_minmax=[0,600.1], y_step=[200,0], y_margin=20.0, y_axis_margin=10.0, x_minmax=[0,1.01], x_margin=0.4, x_axis_margin=0.2, x_ticks=xvalues_2bin, x_ticklabels=["D","100um"] )
savefile = os.path.join(savepath, "Fig-2g-inset-paired-peak-error-vs-100um-swap")
plottingtools.finish_figure( filename=savefile, wspace=0.8, hspace=0.5 )


print("\n\n-----------------------------------------------------------")
print("\nFig. 2g, testing mean map peak error:")
for cnt,nr in enumerate(show_maps):

    print("\nCrosscorr {} vs {}".format( depth_names[ref_map_nr], depth_names[nr] ))

    print("Data vs swap by 100 um")
    statstools.report_mean( peak_displ[nr,:], peak_displ_100[nr,:] )
    statstools.report_wmpsr_test( peak_displ[nr,:], peak_displ_100[nr,:] , n_indents=0, alpha=0.05, bonferroni=1, alternative="two-sided", preceding_text="* ")



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Fig S11e, show figures with peak error (euclidian distance of peak to map center, i.e. peak displacement)

# Plotting variables
xvalues_6bin = [-0.2,0.2,0.8,1.2,1.8,2.2]
show_maps = [0,1,3]
n_show_maps = len(show_maps)

# Get the mean error across shuffles
peak_displ_x_sho = np.nanmean(np.abs(peak_x_sho),axis=2)
peak_displ_x_100 = np.nanmean(np.abs(peak_x_100),axis=2)
peak_displ_y_sho = np.nanmean(np.abs(peak_y_sho),axis=2)
peak_displ_y_100 = np.nanmean(np.abs(peak_y_100),axis=2)

# Mean across mice
peak_displ_x_mn, peak_displ_x_sem, _ = statstools.mean_sem( np.abs(peak_x), axis=1 )
peak_displ_x_sho_mn, peak_displ_x_sho_sem, _ = statstools.mean_sem( peak_displ_x_sho, axis=1 )
peak_displ_x_100_mn, peak_displ_x_100_sem, _ = statstools.mean_sem( peak_displ_x_100, axis=1 )
peak_displ_y_mn, peak_displ_y_sem, _ = statstools.mean_sem( np.abs(peak_y), axis=1 )
peak_displ_y_sho_mn, peak_displ_y_sho_sem, _ = statstools.mean_sem( peak_displ_y_sho, axis=1 )
peak_displ_y_100_mn, peak_displ_y_100_sem, _ = statstools.mean_sem( peak_displ_y_100, axis=1 )

# X and Y displacement from image center, separately
fig = plottingtools.init_fig(fig_size=(4.5*n_show_maps,4))
for cnt,nr in enumerate(show_maps):
    ax = plt.subplot2grid((1,int(n_show_maps)), (0,cnt))

    for m in range(n_mice):
        plt.plot([-0.2,0.2], [np.abs(peak_x[nr,m]), np.abs(peak_y[nr,m])], ".-", color="#AAAAAA", linewidth=0.5, markersize=1, zorder=1)
        plt.plot([0.8,1.2], [peak_displ_x_100[nr,m], peak_displ_y_100[nr,m]], ".-", color="#AAAAAA", linewidth=0.5, markersize=1, zorder=1)
        plt.plot([1.8,2.2], [peak_displ_x_sho[nr,m], peak_displ_y_sho[nr,m]], ".-", color="#AAAAAA", linewidth=0.5, markersize=1, zorder=1)
    plottingtools.bar( -0.2, peak_displ_x_mn[nr], e=0, width=0.36, edge="on", bar_color='None', sem_color='#00ffff', label=None, bottom=0, error_width=0.5 )
    plottingtools.bar( 0.2, peak_displ_y_mn[nr], e=0, width=0.36, edge="on", bar_color='None', sem_color='#000088', label=None, bottom=0, error_width=0.5 )
    plottingtools.bar( 0.8, peak_displ_x_100_mn[nr], e=0, width=0.36, edge="on", bar_color='None', sem_color='#00ffff', label=None, bottom=0, error_width=0.5 )
    plottingtools.bar( 1.2, peak_displ_y_100_mn[nr], e=0, width=0.36, edge="on", bar_color='None', sem_color='#000088', label=None, bottom=0, error_width=0.5 )
    plottingtools.bar( 1.8, peak_displ_x_sho_mn[nr], e=0, width=0.36, edge="on", bar_color='None', sem_color='#00ffff', label=None, bottom=0, error_width=0.5 )
    plottingtools.bar( 2.2, peak_displ_y_sho_mn[nr], e=0, width=0.36, edge="on", bar_color='None', sem_color='#000088', label=None, bottom=0, error_width=0.5 )
    plottingtools.finish_panel( ax, title="", ylabel="Peak error (um)", xlabel="Condition", legend="off", y_minmax=[0,600.01], y_step=[200,0], y_margin=30, y_axis_margin=12, x_minmax=[-0.5,2.51], x_step=[1.0,0], x_margin=0.2, x_axis_margin=0.05, x_ticks=[0,1,2], x_ticklabels=["D","sw100um","shODI"] )
savefile = os.path.join(savepath, "Fig-S11e-cc-peak-error-x-and-y-vs-controls")
plottingtools.finish_figure( filename=savefile, wspace=0.5, hspace=0.5 )



print("\n\n-----------------------------------------------------------")
print("\nSupplementary Fig 11d, testing mean map peak error:")
for cnt,nr in enumerate(show_maps):

    print("\nCrosscorr {} vs {}".format( depth_names[ref_map_nr], depth_names[nr] ))
    samplelist = [peak_displ[nr,:], peak_displ_100[nr,:], peak_displ_200[nr,:], peak_displ_shp[nr,:], peak_displ_sho[nr,:]]
    statstools.report_kruskalwallis( samplelist, n_indents=2, alpha=0.05 )


    print("Data vs swap by 100 um")
    statstools.report_mean( peak_displ[nr,:], peak_displ_100[nr,:] )
    statstools.report_wmpsr_test( peak_displ[nr,:], peak_displ_100[nr,:] , n_indents=0, alpha=0.05, bonferroni=1, alternative="two-sided", preceding_text="* ")

    print("Data vs swap by 200 um")
    statstools.report_mean( peak_displ[nr,:], peak_displ_200[nr,:] )
    statstools.report_wmpsr_test( peak_displ[nr,:], peak_displ_200[nr,:] , n_indents=0, alpha=0.05, bonferroni=1, alternative="two-sided", preceding_text="* ")

    print("Data vs shuffled planes")
    statstools.report_mean( peak_displ[nr,:], peak_displ_shp[nr,:] )
    statstools.report_wmpsr_test( peak_displ[nr,:], peak_displ_shp[nr,:] , n_indents=0, alpha=0.05, bonferroni=1, alternative="two-sided", preceding_text="* ")

    print("Data vs shuffled odi")
    statstools.report_mean( peak_displ[nr,:], peak_displ_sho[nr,:] )
    statstools.report_wmpsr_test( peak_displ[nr,:], peak_displ_sho[nr,:] , n_indents=0, alpha=0.05, bonferroni=1, alternative="two-sided", preceding_text="* ")


print("\n\n-----------------------------------------------------------")
print("\nSupplementary Fig 11d, testing mean map peak error in x vs y:")
for cnt,nr in enumerate(show_maps):

    print("\nCrosscorr {} vs {}".format( depth_names[ref_map_nr], depth_names[nr] ))
    samplelist = [np.abs(peak_x[nr,:]), np.abs(peak_y[nr,:]), peak_displ_x_100[nr,:], peak_displ_y_100[nr,:], peak_displ_x_sho[nr,:], peak_displ_y_sho[nr,:]]
    statstools.report_kruskalwallis( samplelist, n_indents=2, alpha=0.05 )

    print("Data, x vs y")
    statstools.report_mean( np.abs(peak_x[nr,:]), np.abs(peak_y[nr,:]) )
    statstools.report_wmpsr_test( np.abs(peak_x[nr,:]), np.abs(peak_y[nr,:]), n_indents=0, alpha=0.05, bonferroni=1, alternative="two-sided", preceding_text="* ")

    print("swap by 100 um, x vs y")
    statstools.report_mean( peak_displ_x_100[nr,:], peak_displ_y_100[nr,:] )
    statstools.report_wmpsr_test( peak_displ_x_100[nr,:], peak_displ_y_100[nr,:], n_indents=0, alpha=0.05, bonferroni=1, alternative="two-sided", preceding_text="* ")

    print("Shuffled odi, x vs y")
    statstools.report_mean( peak_displ_x_sho[nr,:], peak_displ_y_sho[nr,:] )
    statstools.report_wmpsr_test( peak_displ_y_sho[nr,:], peak_displ_x_sho[nr,:], n_indents=0, alpha=0.05, bonferroni=1, alternative="two-sided", preceding_text="* ")
    

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Done

print("\nDone.. that's all folks!")
