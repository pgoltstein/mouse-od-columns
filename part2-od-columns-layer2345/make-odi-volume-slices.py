#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This script loads data of a single L23-4-5 imaging volume and projects it as 3D plot and shows slices through the 3D volume

Created on Thursday 19 Dec 2024

python make-cell-scatter-odi-volume.py O03

@author: pgoltstein
"""

# Imports
import sys, os
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
import skimage

# Local imports
sys.path.append('../xx_analysissupport')
import plottingtools
import singlevolumeodfunctions

# Module settings
plottingtools.font_size = { "title": 6, "label": 5, "tick": 5, "text": 5, "legend": 5 }

# Arguments
import argparse


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Arguments

parser = argparse.ArgumentParser( description = "This script loads data of a single L23-4-5 imaging volume and projects it as 3D plot and shows slices through the 3D volume.\n (written by Pieter Goltstein - Dec 2024)")
parser.add_argument('mousename', type=str, help='name of the mouse to analyze')
args = parser.parse_args()


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Settings

# Path settings
savepath = "../../figureout"
datapath = os.path.join("../../data/part2-planedata-od-layer2345")
print(f"{datapath=}")

figname = {"O02": "Fig-S8b-",
           "O03": "Fig-S8d-",
           "O06": "Fig-S8f-",
           "O09": "Fig-S8h-",
           "O10": "Fig-S8j-"}

# Data
start_depth=170
depth_increment=10
skip_first_plane=True
include_very_first_plane=True
include_sign = 0.05
exclude_double_xy=3 # micron
exclude_double_z=depth_increment+5 # micron

# Experiment type specific settings
convert_to_micron_x = 1192/1024
convert_to_micron_y = 1019/1024
invert_odi_values = False

side_bins_x = np.arange(0,1200.1,100)
n_side_bins = len(side_bins_x)-1

front_bins_y = np.arange(0,1000.1,100)
n_front_bins = len(front_bins_y)-1

top_bins_z = np.array([170,260,350,440,531])
n_top_bins = len(top_bins_z)-1

odi_cell_sigma=50
max_y=1024
max_x=1200
max_z=700

odi_cmap = "RdBu"
odi_minmax = 0.25
iso_odi_contour_range = [0, 0.2]
iso_odi_contour_linestyle = ["-", "--"]


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Functions

def grid_indices( n_plots, n_columns ):
    """ Returns a list of x and y indices, as well as total plot number to conveniently index subplot2grid """
    if n_columns == -5:
        panel_ids_y = [0,0,0,0,1,1,1,1,1]
        panel_ids_x = [0,1,2,3,0,1,2,3,4]
        n_y=2
        n_x = 5
    else:
        panel_ids_y = np.floor(np.arange(n_plots)/n_columns).astype(int)
        panel_ids_x = np.mod(np.arange(n_plots),n_columns).astype(int)
        n_x = panel_ids_x.max()+1
        n_y = panel_ids_y.max()+1
    return (panel_ids_y,panel_ids_x),(n_y,n_x)

def calc_odi_map_cells( local_XY, local_ODI, odi_cell_sigma, max_y=None, max_x=None ):
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
    odi_im = scipy.ndimage.gaussian_filter(odi_im, sigma=odi_cell_sigma)
    coverage_im = scipy.ndimage.gaussian_filter(coverage_im, sigma=odi_cell_sigma)

    # odi_im = odi_im / n_neurons
    odi_im[np.isnan(coverage_im)] = np.NaN
    odi_im = odi_im / coverage_im

    # Get min/max for colormap
    vmax = max(abs(np.nanmin(odi_im)),abs(np.nanmax(odi_im)))

    # Get iso-odi contour at doi=0
    odi_contours = []
    for odi in iso_odi_contour_range:
        odi_contours.append( skimage.measure.find_contours(image=odi_im, level=odi) )
    return odi_im, odi_contours, vmax


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Load data

print("Loading imaging volume of mouse {}".format(args.mousename))
print("  << {} >>".format(datapath))
volume,parameter_names,aspect_ratio,_,_ = singlevolumeodfunctions.load_volume( datapath, args.mousename, start_depth=start_depth, depth_increment=depth_increment, skip_first_plane=skip_first_plane, include_very_first_plane=include_very_first_plane, convert_to_micron_x=convert_to_micron_x, convert_to_micron_y=convert_to_micron_y, include_sign=include_sign, exclude_double_xy=exclude_double_xy, exclude_double_z=exclude_double_z, invert_odi_values=invert_odi_values )
print("Loaded parameters:")
for nr,name in enumerate(parameter_names):
    print("{:>2d}: {}".format(nr,name))

# Get data
x = volume[:, parameter_names.index("x")]
y = volume[:, parameter_names.index("y")]
z = volume[:, parameter_names.index("z")]
ODI = volume[:, parameter_names.index("ODI")]



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Side views tiled

# Loop side bins
(panel_ids_y,panel_ids_x),(n_y,n_x) = grid_indices( n_side_bins, 4 )

fig,_ = plottingtools.init_figure(fig_size=(6*n_x,6*n_y))
for b in range(n_side_bins):

    # Select depth range
    bin_selector = np.logical_and(x>side_bins_x[b], x<=side_bins_x[b+1])
    s_y = y[bin_selector]
    s_z = z[bin_selector]
    s_odi = ODI[bin_selector]
    YZ = np.stack([s_y,s_z],axis=1)

    # Get ODI map
    odi_im, odi_contours, vmax = calc_odi_map_cells( YZ, s_odi, odi_cell_sigma=odi_cell_sigma, max_y=max_z, max_x=max_y )
    odi_im[:160,:] = np.NaN
    odi_im[530:,:] = np.NaN
    
    # Flip x-axis so aligns with top-views
    odi_im = odi_im[:,::-1]

    # Show image in subplot
    ax = plt.subplot2grid((n_y,n_x), (panel_ids_y[b],panel_ids_x[b]))
    plt.imshow(odi_im, vmin=-1.0*odi_minmax, vmax=odi_minmax, cmap=odi_cmap)
    ax.set_yticks(np.arange(0,700,200))

    # Save as image
    if b==7 and args.mousename=="O03":
        odimap_file = os.path.join(savepath, "Fig-2a-{}-side-odimap-s{}-d{}.png".format(args.mousename, side_bins_x[b], side_bins_x[b+1]-1))
        print("Saving side odi map to file: {}".format(odimap_file))
        plt.imsave(fname=odimap_file, arr=odi_im, vmin=-1.0*odi_minmax, vmax=odi_minmax, cmap=odi_cmap)

# Save figure
savefile = os.path.join( savepath, figname[args.mousename]+"{}-odimap-sideview".format(args.mousename) )
plottingtools.finish_figure( filename=savefile, wspace=0.5, hspace=0.5 )



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Top views tiled

fig,ax = plottingtools.init_figure(fig_size=(6,6))

# Select depth range for L4
bin_selector = np.logical_and(z>top_bins_z[2], z<=top_bins_z[3])
s_x = x[bin_selector]
s_y = y[bin_selector]
s_odi = ODI[bin_selector]
XY = np.stack([s_x,s_y],axis=1)

# Get ODI map
odi_im, odi_contours, vmax = calc_odi_map_cells( XY, s_odi, odi_cell_sigma=odi_cell_sigma, max_y=max_y, max_x=max_x )

# Show image
plt.imshow(odi_im, vmin=-1.0*odi_minmax, vmax=odi_minmax, cmap=odi_cmap)

# Save figure
savefile = os.path.join( savepath, figname[args.mousename]+"{}-odimap-topview".format(args.mousename) )
plottingtools.finish_figure( filename=savefile, wspace=0.5, hspace=0.5 )



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 3d scatter plot with ODI as color

if args.mousename=="O03":

    fig = plt.figure()

    ax = fig.add_subplot(projection='3d', auto_add_to_figure=False)
    fig.subplots_adjust(top=1.1, bottom=-0.1, left=-0.1, right=1.1)

    # Select cells
    _selector = ODI > -100 # include all neurons
    p_x = x[_selector]
    p_y = y[_selector]
    p_z = z[_selector]
    p_odi = ODI[_selector]

    # Show 3D scatter plot
    ax.scatter(p_x, p_y, p_z, c=p_odi, marker=".", cmap="seismic_r", vmin=-1.0, vmax=1.0, s=2.0)
    ax.view_init(elev=25, azim=-60)

    # Save figure
    savefile = os.path.join( savepath, "Fig-2a-{}-3d-odimap-cells".format(args.mousename) )
    plottingtools.finish_figure( filename=savefile, wspace=0.1, hspace=0.1 )


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Done

print("\nDone.. that's all folks!")
