#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This script loads responsemaps of a L23-L4-L5 experiment and makes cellbased ODI maps across depth

python make-volume-cellbased-odimaps.py O03

Created on Wednesday 5 May 2022

@author: pgoltstein
"""


# Imports
import sys, os
import numpy as np
import matplotlib.pyplot as plt
import skimage.morphology
import skimage.filters
import scipy.ndimage
from skimage import measure
from skimage.transform import resize as imresize
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

parser = argparse.ArgumentParser( description = "This script loads responsemaps of a L23-L4-L5 experiment and makes cellbased ODI maps across depth.\n (written by Pieter Goltstein - May 2022)")
parser.add_argument('mousename', type=str, help='name of the mouse to analyze')
args = parser.parse_args()


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Settings

# Path settings
savepath = "../../figureout"
datapath = os.path.join("../../data/part2-planedata-od-layer2345")
print(f"{datapath=}")

# Data
start_depth = 170
depth_increment = 10
skip_first_plane=True
include_sign = 0.05
exclude_double_xy=3 # micron
exclude_double_z=depth_increment+5 # micron
n_stimuli = 2
convert_to_micron_x = 1192/1024
convert_to_micron_y = 1019/1024
x_res, y_res = 1024,1024

# Depth bins
depth_bins = [170,260,350,440,540] # last bin should include extra plane step
n_bins = len(depth_bins)-1

# Cluster settings
fraction = 0.05
rho_min = 0.2
delta_min = 0.2
rho_x_delta_min = None

# Map settings
odi_cmap = "RdBu"
odi_minmax = 0.5
filter_sigma = 16
blank_low_density_regions = False
coverage_thr_nstd = 1
coverage_thr_ncells = 5


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Functions

def plot_clusters(clusters, markercolor="#000000"):
    for c in clusters:
        plt.plot( c["X"], c["Y"], marker="o", markersize=4, markeredgewidth=1.5, markeredgecolor=markercolor, markerfacecolor='None')

def calc_odi_map_cells( local_XY, local_ODI ):
    # Prepare a matrix that represents the image
    max_y = np.ceil(max(local_XY[:,1])).astype(int)+1
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
        coverage_im[round_XY[n,1],round_XY[n,0]] = coverage_im[round_XY[n,1],round_XY[n,0]] + 1.0

    # Smooth maps
    odi_im = scipy.ndimage.gaussian_filter(odi_im, sigma=filter_sigma)
    coverage_im = scipy.ndimage.gaussian_filter(coverage_im, sigma=filter_sigma)

    # odi_im = odi_im / n_neurons
    odi_im[np.isnan(coverage_im)] = np.NaN
    odi_im = odi_im / coverage_im

    # Get min/max for colormap
    vmax = max(abs(np.nanmin(odi_im)),abs(np.nanmax(odi_im)))

    return odi_im, vmax, coverage_im


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Load complete volume

print("Loading imaging volume of mouse {}".format(args.mousename))
print("  << {} >>".format(datapath))
volume,parameter_names,aspect_ratio,_,_ = singlevolumeodfunctions.load_volume( datapath, args.mousename, start_depth=start_depth, depth_increment=depth_increment, skip_first_plane=skip_first_plane, convert_to_micron_x=convert_to_micron_x, convert_to_micron_y=convert_to_micron_y, include_sign=include_sign, exclude_double_xy=exclude_double_xy, exclude_double_z=exclude_double_z, invert_odi_values=False )
print("Loaded parameters:")
for nr,name in enumerate(parameter_names):
    print("{:>2d}: {}".format(nr,name))


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Select data

# All data
XY = volume[:, [parameter_names.index("x"),parameter_names.index("y")]]
z = volume[:, parameter_names.index("z")]
ODI = volume[:,parameter_names.index("ODI")]

# L4 data
L4_selector = np.logical_and(z>365, z<435)
volume_L4 = volume[L4_selector,:]
XY_L4 = volume_L4[:, [parameter_names.index("x"),parameter_names.index("y")]]
ODI_L4 = volume_L4[:,parameter_names.index("ODI")]

# Data that should be clustered
XY_ipsi = XY_L4[ODI_L4<=0,:]
XY_contra = XY_L4[ODI_L4>0,:]


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Get clusters

# Detect ipsi clusters
clusters = densityclustering.find_clusters(XY_ipsi, fraction=0.05, rho_min=rho_min, delta_min=delta_min, rho_x_delta_min=rho_x_delta_min, show_rho_vs_delta=False)
print("Detected {} ipsi clusters in L4".format(len(clusters)))
for nr,c in enumerate(clusters):
    print("{}) {}, {}: rho={:0.3f}, delta={:0.3f}".format(nr, c["X"], c["Y"], c["rho"], c["delta"]))


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Average planes into x bins, create ODI maps and save

# Cell based scatter maps
fig = plottingtools.init_fig(fig_size=(4*aspect_ratio,16))
for bin_nr,(depth1,depth2) in enumerate(zip(depth_bins[:-1],depth_bins[1:])):

    # Select cells in depth range
    print("Selecting cells at z {} to {} (inclusive)".format(depth1,depth2-depth_increment))
    XY_d = XY[np.logical_and(z>=depth1,z<depth2),:]
    ODI_d = ODI[np.logical_and(z>=depth1,z<depth2)]

    # For making the background grey
    grey_bg = np.zeros((1024,int(1024*aspect_ratio)))+0.5

    # Show cell ODI map
    ax = plt.subplot2grid((n_bins,1),(bin_nr,0))
    plt.imshow(grey_bg,cmap="Greys",vmin=0,vmax=1)
    singlevolumeodfunctions.show_param_2d( ax, XY_d[:,0], XY_d[:,1], ODI_d, title=None, name=None, cmap="seismic_r", vmin=-1, vmax=1, size=0.5 )

    # Add clusters and contours
    if bin_nr == 2 and args.mousename == "O03":
        plot_clusters(clusters)
    plt.gca().invert_yaxis()

# Save figure
if args.mousename == "O03":
    odimap_file = os.path.join(savepath, "Fig-2b-{}-cellbased-odimaps".format(args.mousename))
elif args.mousename == "O10":
    odimap_file = os.path.join(savepath, "Fig-S11a-{}-cellbased-odimaps".format(args.mousename))
print("Saving odi maps to file: {}".format(odimap_file))
plottingtools.finish_figure( filename=odimap_file, wspace=0.05, hspace=0.05 )


# Cell-based average ODI maps
if args.mousename == "O10":

    fig = plottingtools.init_fig(fig_size=(4*aspect_ratio,16))
    for bin_nr,(depth1,depth2) in enumerate(zip(depth_bins[:-1],depth_bins[1:])):

        # Select cells in depth range
        print("Selecting cells at z {} to {} (inclusive)".format(depth1,depth2-depth_increment))
        XY_d = XY[np.logical_and(z>=depth1,z<depth2),:]
        ODI_d = ODI[np.logical_and(z>=depth1,z<depth2)]

        # Get ODI map based on cells, and ODI contours
        odi_im, _, coverage_im = calc_odi_map_cells( XY_d, ODI_d )

        # Blank low density regions with less than 5 cells within 1 std
        if blank_low_density_regions:
            max_est = np.zeros_like( coverage_im )
            max_est[500,500] = 1.0 * coverage_thr_ncells
            max_est = scipy.ndimage.gaussian_filter(max_est, sigma=filter_sigma)
            print("max coverage_im after filt = {}".format(np.nanmax(max_est)))
            thresh = max_est[500-int(np.round(filter_sigma*coverage_thr_nstd)),500]
            print("{}*sigma coverage_im for {} cells, after filt = {}".format(coverage_thr_nstd,coverage_thr_ncells,thresh))

            odi_im[coverage_im<thresh] = np.NaN
            print("coverage_im, min = {}".format(np.nanmin(coverage_im)))
            print("coverage_im, max = {}".format(np.nanmax(coverage_im)))

        ax = plt.subplot2grid((n_bins,1),(bin_nr,0))
        plt.imshow(odi_im, vmin=-1.0*odi_minmax, vmax=odi_minmax, cmap=odi_cmap)
        ax.axis("equal")
        plt.axis("off")

    # Save figure
    odimap_file = os.path.join(savepath, "Fig-S11a-{}-cellbased-average-odimaps".format(args.mousename, depth1, depth2-10))
    print("Saving odi maps to file: {}".format(odimap_file))
    plottingtools.finish_figure( filename=odimap_file, wspace=0.05, hspace=0.05 )

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Done
print("\nDone.. that's all folks!")
