#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This script loads data of a single -Suite2p- or single -CaImAn- imaging volume, shows OD and eye-preference density, and finds clusters and calculates ODI histogram

Created on Monday 14 Dec 2024

python make-od-cluster-maps-preprocesscompare.py caiman

@author: pgoltstein
"""


# Imports
import sys, os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.optimize
import sklearn.metrics
import scipy.ndimage
import skimage

# Local imports
sys.path.append('../xx_analysissupport')
import densityclustering
import plottingtools
import statstools
import singlevolumeodfunctions

# Module settings
plottingtools.font_size = { "title": 6, "label": 5, "tick": 5, "text": 5, "legend": 5 }

# Arguments
import argparse


#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Arguments

parser = argparse.ArgumentParser( description = "This script loads data of a single -CaImAn- imaging volume, shows OD and eye-preference density, and finds clusters and calculates ODI histogram.\n (written by Pieter Goltstein - Dec 2024)")
parser.add_argument('preprocessing', type=str, help= 'Whether to use caiman or suite2p preprocessed data (should be either "caiman" or "suite2p")')
args = parser.parse_args()


#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Settings

# Path settings
settingspath = "../settings"
savepath = "../../figureout"
if args.preprocessing == "suite2p":
    datapath = os.path.join("../../data/part1-planedata-od-layer4")
    print(f"{datapath=}")
    figname = "Fig-S6a"
elif args.preprocessing == "caiman":
    datapath = os.path.join("../../data/part1-planedata-od-layer4-caiman")
    print(f"{datapath=}")
    figname = "Fig-S6b"

# Data
mouse = "O03"
start_depth=370
depth_increment=20
skip_first_plane=False
include_sign = 0.05
exclude_double_xy=3 # micron
exclude_double_z=depth_increment+5 # micron
iso_odi_contour_range = [0, 0.2]
iso_odi_contour_linestyle = ["-", "--"]

# Experiment type specific settings
convert_to_micron_x = 1192/1024
convert_to_micron_y = 1019/1024
invert_odi_values = False

# Cluster settings
fraction = 0.05
rho_min = 0.2
delta_min = 0.2
rho_x_delta_min = None


#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Load data

print("Loading imaging volume of mouse {}".format(mouse))
print("  << {} >>".format(datapath))
volume,parameter_names,aspect_ratio,_,_ = singlevolumeodfunctions.load_volume( datapath, mouse, start_depth=start_depth, depth_increment=depth_increment, skip_first_plane=skip_first_plane, convert_to_micron_x=convert_to_micron_x, convert_to_micron_y=convert_to_micron_y, include_sign=include_sign, exclude_double_xy=exclude_double_xy, exclude_double_z=exclude_double_z, invert_odi_values=invert_odi_values )
print("Loaded parameters:")
for nr,name in enumerate(parameter_names):
    print("{:>2d}: {}".format(nr,name))
min_z = int(np.min(volume[:,parameter_names.index("z")]))
max_z = int(np.max(volume[:,parameter_names.index("z")]))

# Get data
XY = volume[:, [parameter_names.index("x"),parameter_names.index("y")]]
ODI = volume[:,parameter_names.index("ODI")]

# Select data that should be clustered
XY_ipsi = XY[ODI<=0,:]
XY_contra = XY[ODI>0,:]


#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Get clusters

# Detect ipsi clusters
clusters = densityclustering.find_clusters(XY_ipsi, fraction=0.05, rho_min=rho_min, delta_min=delta_min, rho_x_delta_min=rho_x_delta_min, show_rho_vs_delta=False)
print("Detected {} ipsi clusters".format(len(clusters)))
for nr,c in enumerate(clusters):
    print("{}) {}, {}: rho={:0.3f}, delta={:0.3f}".format(nr, c["X"], c["Y"], c["rho"], c["delta"]))


#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Group cells in a pixelwise ODI map, for making ODI contours

# Prepare a matrix that represents the image
max_y = np.ceil(max(XY[:,1])).astype(int)+1
max_x = np.ceil(max(XY[:,0])).astype(int)+1
print("Image dims: {},{}".format(max_y,max_x))
odi_im = np.zeros((max_y,max_x))
coverage_im = np.zeros((max_y,max_x))

# Loop cells and add them to the image
round_XY = np.round(XY).astype(int)
n_neurons = XY.shape[0]
for n in range(n_neurons):

    # Sum ODI and coverage
    odi_im[round_XY[n,1],round_XY[n,0]] = odi_im[round_XY[n,1],round_XY[n,0]] + ODI[n]
    coverage_im[round_XY[n,1],round_XY[n,0]] = 1.0

# Smooth maps
odi_im = scipy.ndimage.gaussian_filter(odi_im, sigma=50)
coverage_im = scipy.ndimage.gaussian_filter(coverage_im, sigma=50)

# odi_im = odi_im / n_neurons
odi_im[np.isnan(coverage_im)] = np.NaN
odi_im = odi_im / coverage_im

# Get min/max for colormap
vmax = max(abs(np.nanmin(odi_im)),abs(np.nanmax(odi_im)))
print("{}, {}, {}".format(np.nanmin(odi_im),np.nanmax(odi_im),vmax))
vmax = 0.50
print("manually setting vmax to {}".format(vmax))

# Get iso-odi contour at doi=0
odi_contours = []
for odi in iso_odi_contour_range:
    odi_contours.append( skimage.measure.find_contours(image=odi_im, level=odi) )


#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Show clusters

def plot_clusters(clusters, markersize=250, markerlength=30, markerwidth=2, markercolor="#000000"):
    mrkr_outer = int(markersize/2)
    mrkr_inner = (int(markersize/2)-markerlength)
    for c in clusters:
        plt.plot( c["X"], c["Y"], marker="o", markersize=5, markeredgewidth=1.0, markeredgecolor=markercolor, markerfacecolor='None')

# Calculate local density for ipsi
D = densityclustering.distance_matrix(XY_ipsi, quiet=True)
d_c = densityclustering.estimate_d_c(D,fraction=0.05)
rho_ipsi = densityclustering.local_density(D, d_c,normalize=True)

# Calculate local density for contra
D = densityclustering.distance_matrix(XY_contra, quiet=True)
d_c = densityclustering.estimate_d_c(D,fraction=0.05)
rho_contra = densityclustering.local_density(D, d_c,normalize=True)

# For making the background grey
grey_bg = np.zeros_like(coverage_im)+0.5

# Init figure
fig,ax = plottingtools.init_figure(fig_size=(12*aspect_ratio,12))

# Show ipsi density
ax = plt.subplot2grid((2,2),(0,0),fig=fig)
plt.imshow(grey_bg,cmap="Greys",vmin=0,vmax=1)
singlevolumeodfunctions.show_param_2d( ax, XY_ipsi[:,0], XY_ipsi[:,1], rho_ipsi, "Ipsi preferring cells", name=None, cmap="magma", vmin=0, vmax=1, d1=min_z, d2=max_z, size=2  ) # color coded rho
plot_clusters(clusters)
plt.gca().invert_yaxis()

# Show contra density
ax = plt.subplot2grid((2,2),(0,1),fig=fig)
plt.imshow(grey_bg,cmap="Greys",vmin=0,vmax=1)
singlevolumeodfunctions.show_param_2d( ax, XY_contra[:,0], XY_contra[:,1], rho_contra, "Contra preferring cells", name=None, cmap="magma", vmin=0, vmax=1, d1=min_z, d2=max_z, size=2 ) # color coded rho
plot_clusters(clusters)
plt.gca().invert_yaxis()

# Show ipsi cluster centers on ODI map
ax = plt.subplot2grid((2,2),(1,1),fig=fig)
plt.imshow(grey_bg,cmap="Greys",vmin=0,vmax=1)
singlevolumeodfunctions.show_param_2d( ax, XY[:,0], XY[:,1], ODI, "ODI map with ipsi clusters", name=None, cmap="seismic_r", vmin=-1, vmax=1, d1=370, d2=430, size=2 )
plot_clusters(clusters)
for odi_ix in range(len(odi_contours)):
    for c in range(len(odi_contours[odi_ix])):
        plt.plot(odi_contours[odi_ix][c][:,1],odi_contours[odi_ix][c][:,0], iso_odi_contour_linestyle[odi_ix], color="#ffffff",  markersize=0, linewidth=0.5)
plt.gca().invert_yaxis()

# Show a histogram of OD values
m,e,_ = statstools.mean_sem(ODI)
ax = plt.subplot2grid((2,2),(1,0),fig=fig)
ax = sns.histplot(data=ODI, ax=ax, binwidth=0.1, stat="probability", color="#999999")
plottingtools.finish_panel( ax, title="Mean ODI of tuned neurons: {:5.3f} (±{:5.3f}))".format( m, e ), ylabel="p", xlabel="ODI", legend="off", y_minmax=[0,0.151], y_step=[0.05,2], y_margin=0.0, y_axis_margin=0.0, x_minmax=[-1.0,1.01], x_step=[0.5,1], x_margin=0.05, x_axis_margin=0.02, x_ticks=None, x_ticklabels=None, y_ticks=None, y_ticklabels=None, x_tick_rotation=0, tick_size=6, label_size=6, title_size=6, legend_size=6, despine=True, legendpos=0)

# Save figure
savefile = os.path.join( savepath, figname+"-{}-denstity-odi-cellmaps-clusters-{}".format(mouse,args.preprocessing) )
plottingtools.finish_figure( filename=savefile, wspace=0.2, hspace=0.2 )



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# That's all folks !!
print("\nDone.\n")
