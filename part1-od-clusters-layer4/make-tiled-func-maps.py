#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This script loads data of a single tiled-FOV imaging volume, displays maps for ODI and preferred direction, finds clusters and calculates ODI for clusters

python make-tiled-func-maps.py O10

Created on Monday 2 May 2022

@author: pgoltstein
"""

# Imports
import sys, os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.optimize
import sklearn.metrics
import scipy.ndimage
import skimage

# Local imports
sys.path.append('../xx_analysissupport')
import densityclustering
import plottingtools
import singlevolumeodfunctions

# Module settings
plottingtools.font_size = { "title": 6, "label": 5, "tick": 5, "text": 5, "legend": 5 }

# Arguments
import argparse


#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Arguments

parser = argparse.ArgumentParser( description = "This script loads data of a single tiled-FOV imaging volume, displays maps for ODI and preferred direction, finds clusters and calculates ODI for clusters.\n (written by Pieter Goltstein - Oct 2021)")
parser.add_argument('mousename', type=str, help='name of the mouse to analyze')
parser.add_argument('-n', '--nswaps', type=int, default=100, help='manually set the number of repeats for the local randomization control, i.e. the swapping-neuron-locations routine (default=100).')
args = parser.parse_args()


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Settings

# Path settings
settingspath = "../settings"
savepath = "../../figureout"
datapath = os.path.join("../../data/part1-tileddata-od-layer4")
print(f"{datapath=}")

# Data
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
delta_min = 0.07 # Set this to half or a third of the normal value because the image is twice as large and therefore the normalized distance if also twice as small
rho_x_delta_min = None

# Binning and local shuffle settings
bin_size = 25
distance_range=[0,410]
swap_dists = list(range(0,251,50))
swap_dist_cmap = matplotlib.cm.get_cmap("hot")
n_swap_dist_iter = 5 # 100
xvalues_fit = np.arange(distance_range[0],distance_range[1],1)

# Sigmoid fit settings
lead_bins = 5
incl_n_pt = 15 + lead_bins
scale_x_to_prevent_overflow = 100


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Load data

print("Loading imaging volume of mouse {}".format(args.mousename))
print("  << {} >>".format(datapath))
volume,parameter_names,aspect_ratio,_,_ = singlevolumeodfunctions.load_volume( datapath, args.mousename, start_depth=start_depth, depth_increment=depth_increment, skip_first_plane=skip_first_plane, convert_to_micron_x=convert_to_micron_x, convert_to_micron_y=convert_to_micron_y, include_sign=include_sign, exclude_double_xy=exclude_double_xy, exclude_double_z=exclude_double_z, invert_odi_values=invert_odi_values, include_fovpos=True )
print("Loaded parameters:")
for nr,name in enumerate(parameter_names):
    print("{:>2d}: {}".format(nr,name))
min_z = int(np.min(volume[:,parameter_names.index("z")]))
max_z = int(np.max(volume[:,parameter_names.index("z")]))

# Get data
XY = volume[:, [parameter_names.index("x"),parameter_names.index("y")]]
ODI = volume[:,parameter_names.index("ODI")]
DIR = volume[:,parameter_names.index("Pref dir")]

# Select data that should be clustered
XY_ipsi = XY[ODI<=0,:]
XY_contra = XY[ODI>0,:]


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Get clusters

# Detect ipsi clusters
clusters = densityclustering.find_clusters(XY_ipsi, fraction=0.05, rho_min=rho_min, delta_min=delta_min, rho_x_delta_min=rho_x_delta_min, show_rho_vs_delta=False)
print("Detected {} ipsi clusters".format(len(clusters)))
for nr,c in enumerate(clusters):
    print("{}) {}, {}: rho={:0.3f}, delta={:0.3f}".format(nr, c["X"], c["Y"], c["rho"], c["delta"]))


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Group cells in a pixelwise ODI map and get odi_contours (iso-ODI lines)

def odi_map( local_XY, local_ODI ):
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
        coverage_im[round_XY[n,1],round_XY[n,0]] = 1.0

    # Smooth maps
    odi_im = scipy.ndimage.gaussian_filter(odi_im, sigma=50)
    coverage_im = scipy.ndimage.gaussian_filter(coverage_im, sigma=50)

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

odi_im, odi_contours, vmax = odi_map( XY, ODI )


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Show clusters on ODI scatter map

def plot_clusters(clusters, markersize=250, markerlength=30, markerwidth=2, markercolor="#000000"):
    mrkr_outer = int(markersize/2)
    mrkr_inner = (int(markersize/2)-markerlength)
    for c in clusters:
        plt.plot( c["X"], c["Y"], marker="o", markersize=10, markeredgewidth=2.0, markeredgecolor=markercolor, markerfacecolor='None')

# For making the background grey
grey_bg = np.zeros_like(odi_im)+0.5

# Show ipsi cluster centers on ODI map
fig,ax = plottingtools.init_figure(fig_size=(8*aspect_ratio,8))
plt.imshow(grey_bg,cmap="Greys",vmin=0,vmax=1)
singlevolumeodfunctions.show_param_2d( ax, XY[:,0], XY[:,1], ODI, "ODI map with ipsi clusters", name=None, cmap="seismic_r", vmin=-1, vmax=1, d1=370, d2=430, size=1 )
plot_clusters(clusters)
for odi_ix in range(len(odi_contours)):
    for c in range(len(odi_contours[odi_ix])):
        plt.plot(odi_contours[odi_ix][c][:,1],odi_contours[odi_ix][c][:,0], iso_odi_contour_linestyle[odi_ix], color="#ffffff",  markersize=0, linewidth=0.5)
plt.gca().invert_yaxis()
savefile = os.path.join( savepath, "Fig-S2a-tiled-cellmaps-odi-{}".format(args.mousename) )
plottingtools.finish_figure( filename=savefile, wspace=0.2, hspace=0.2 )

# Show preferred direction
print("min,max direction: {},{}".format(np.nanmin(DIR),np.nanmax(DIR)))
fig,ax = plottingtools.init_figure(fig_size=(8*aspect_ratio,8))
singlevolumeodfunctions.show_param_2d( ax, XY[:,0], XY[:,1], DIR, "Preferred direction", name=None, cmap="hsv", vmin=0, vmax=360, d1=min_z, d2=max_z, size=1 ) # color coded rho
plot_clusters(clusters)
savefile = os.path.join( savepath, "Fig-S2d-tiled-cellmaps-preferreddirection-{}".format(args.mousename) )
plottingtools.finish_figure( filename=savefile, wspace=0.2, hspace=0.2 )


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Show ODI as function of distance from cluster centers

def sigmoid(p,x):
    # y0: minimum of curve
    # c: maximum of curve relative to y0
    # k: steepness
    # x0: time point of maximum steepness
    x0,y0,c,k=p
    y = c / (1.0 + np.exp(-k*(x-x0))) + y0
    return y

def residuals(p,x,y):
    return y - sigmoid(p,x)

def get_sigmoid_fit( x, y ):
    # returns parameters (x0,y0,c,k)
    p_guess=(0.2*np.max(x),np.min(y),np.max(y),10.0)
    # p_guess=(np.median(x),np.median(y),1.0,1.0)
    p, cov, infodict, mesg, ier = scipy.optimize.leastsq(
                    residuals,p_guess,args=(x,y),full_output=1)
    return p

odi_per_bin,xvalues = densityclustering.value_per_shell(XY[:,0], XY[:,1], ODI, clusters, bin_size=bin_size, start=distance_range[0], end=distance_range[1])

# Get odi cluster bins
odi_curve = np.array(np.mean(odi_per_bin,axis=0))

# Add lead bins to odi curve that have the value of the lowest bin
odi_curve = np.concatenate((np.zeros((lead_bins,))+np.nanmin(odi_curve),odi_curve))
xvalues_ = np.concatenate((xvalues[:lead_bins]*-1,xvalues))

# Fit sigmoid to curve
p = get_sigmoid_fit( xvalues_[:incl_n_pt]/scale_x_to_prevent_overflow, odi_curve[:incl_n_pt] )
x0,y0,c,k = p # k is steepness
odi_fit = sigmoid(p,xvalues_fit/100)

# Prepare figure
fig,ax = plottingtools.init_figure(fig_size=(3.5,4))
for c_nr in range(len(clusters)):
    plt.plot(xvalues, odi_per_bin[c_nr,:], ".-", color="#888888",  markersize=1, linewidth=0.5, zorder=1)
plt.plot(xvalues, np.mean(odi_per_bin,axis=0), ".-", color="#000000", markersize=3, linewidth=1.0, zorder=3)
plt.plot(xvalues_fit, odi_fit, "-", color="#880000", linewidth=0.5, zorder=4)
plottingtools.finish_panel( ax, title="", ylabel="ODI", xlabel="Distance from cluster center (micron)", legend="off", y_minmax=[-0.2,0.4], y_step=[0.2,1], y_margin=0.05, y_axis_margin=0.03, x_minmax=[0.0,distance_range[1]+10.01], x_step=[100.0,0], x_margin=20, x_axis_margin=5 )

# Save the figure
savefile = os.path.join( savepath, "Fig-S2g-tiled-odi-from-cluster-center-{}".format(args.mousename) )
plottingtools.finish_figure( filename=savefile, wspace=0.2, hspace=0.2 )

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Show local cluster ODI as function of local shuffle

def swap_dist_swap_coords(XY, swap_dist_distance, max_dist=50):
    XY_sct = np.zeros_like(XY)
    swap_dist_list = []
    cell_list = np.arange(XY.shape[0])

    # Calculate the distance to all other remaining cells
    D = sklearn.metrics.pairwise_distances(XY, metric="euclidean")
    np.fill_diagonal(D, 100000.0)

    max_dist_exceeded = 0
    not_done = True
    while not_done:

        # take a random cell
        n1 = np.random.choice(cell_list)

        # find a cell exactly the swap_dist distance away
        n2 = np.argmin(np.abs(D[n1,:]-swap_dist_distance))
        swap_dist_list.append(D[n1,n2])
        if np.abs(D[n1,n2]-swap_dist_distance) > max_dist:
            max_dist_exceeded += 1

        # swap these cells
        XY_sct[n1,:] = XY[n2,:]
        XY_sct[n2,:] = XY[n1,:]

        # remove both cells from list
        cell_list = cell_list[cell_list != n1]
        cell_list = cell_list[cell_list != n2]

        # set distances from used cells to large number
        D[n1,:] = 100000.0
        D[:,n1] = 100000.0
        D[n2,:] = 100000.0
        D[:,n2] = 100000.0

        if len(cell_list) < 2:
            not_done = False
    return XY_sct,swap_dist_list,max_dist_exceeded


# Prepare figure and analysis
swap_dists_maps = [0,100,200]
nx_plots = 3
ny_plots = 2
odi_ims = []
odi_contourss = []
vmaxs = []
fig,ax = plottingtools.init_figure(fig_size=(nx_plots*6*aspect_ratio,ny_plots*6))

# Loop swap_dist radii
for nr,swap_dist in enumerate(swap_dists_maps):

    # plot the local cluster ODI as function of local shuffle
    if swap_dist > 0:
        XY_plot,swap_dist_list,max_dist_exceeded = swap_dist_swap_coords(XY, swap_dist_distance=swap_dist)
    else:
        XY_plot = np.array(XY)
    _odi_im, _odi_contours, _vmax = odi_map( XY_plot, ODI )
    odi_ims.append(_odi_im)
    odi_contourss.append(_odi_contours)
    vmaxs.append(_vmax)

    # Show the shuffled cell positions
    ax = plt.subplot2grid((ny_plots,nx_plots),(0,nr),fig=fig)
    plt.imshow(grey_bg,cmap="Greys",vmin=0,vmax=1)
    singlevolumeodfunctions.show_param_2d( ax, XY_plot[:,0], XY_plot[:,1], ODI, title="ODI map (swap_dist={})".format(swap_dist), name=None, cmap="seismic_r", vmin=-1, vmax=1, d1=min_z, d2=max_z, size=1 )

    # # Detect ipsi clusters
    XY_ipsi = XY_plot[ODI<=0,:]
    clusters_swp = densityclustering.find_clusters(XY_ipsi, fraction=0.05, rho_min=rho_min, delta_min=delta_min, rho_x_delta_min=rho_x_delta_min, show_rho_vs_delta=False)
    print("Detected {} ipsi clusters".format(len(clusters_swp)))
    for nr,c in enumerate(clusters_swp):
        print("{}) {}, {}: rho={:0.3f}, delta={:0.3f}".format(nr, c["X"], c["Y"], c["rho"], c["delta"]))

    plot_clusters(clusters_swp)

    for odi_ix in range(len(_odi_contours)):
        for c in range(len(_odi_contours[odi_ix])):
            plt.plot(_odi_contours[odi_ix][c][:,1],_odi_contours[odi_ix][c][:,0], iso_odi_contour_linestyle[odi_ix], color="#ffffff",  markersize=0, linewidth=0.5)
    plt.gca().invert_yaxis()

# Show mean pixelwise ODI map
vmax = np.max(vmaxs)
print("vmax={}".format(vmax))

# Loop swap_dist radii
for nr,swap_dist in enumerate(swap_dists_maps):

    # Show the shuffled cell positions
    ax = plt.subplot2grid((ny_plots,nx_plots),(1,nr),fig=fig)
    plt.imshow(odi_ims[nr], cmap="seismic_r", vmin=-vmax, vmax=vmax)

    odi_contours = odi_contourss[nr]
    for odi_ix in range(len(odi_contours)):
        for c in range(len(odi_contours[odi_ix])):
            plt.plot(odi_contours[odi_ix][c][:,1],odi_contours[odi_ix][c][:,0], iso_odi_contour_linestyle[odi_ix], color="#000000",  markersize=0, linewidth=0.5)
    plt.colorbar()
    ax.axis("equal")
    plt.axis("off")

savefile = os.path.join( savepath, "Fig-S2ef-tiled-cell-odi-maps-swapdists-{}".format(args.mousename) )
plottingtools.finish_figure( filename=savefile, wspace=0.2, hspace=0.2 )


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Show single plot with swap_dists overlaid

# Prepare variables
n_bins = len(np.arange(distance_range[0],distance_range[1],bin_size))-1
n_swap_dists = len(swap_dists)
swap_dist_bin_values = np.zeros(( n_swap_dists, n_bins ))

# Loop swap_dists and get values in matrix
for nr,swap_dist in enumerate(swap_dists):
    if swap_dist > 0:
        print("\nSwapping ODIs")
        bin_values = np.zeros(( args.nswaps, n_bins ))
        for it in range(args.nswaps):
            XY_sct,swap_dist_list,max_dist_exceeded = swap_dist_swap_coords(XY, swap_dist_distance=swap_dist)

            # Recalculate clusters
            XY_sct_ipsi = XY_sct[ODI<=0,:]
            sct_clusters = densityclustering.find_clusters(XY_sct_ipsi, fraction=0.05, rho_min=rho_min, delta_min=delta_min, rho_x_delta_min=rho_x_delta_min, show_rho_vs_delta=False)
            print("Detected {} ipsi clusters (swap_dist={})".format(len(sct_clusters),swap_dist))
            print("Exceeded distance threshold {}x".format(max_dist_exceeded))
            for c_nr,c in enumerate(sct_clusters):
                print("{}) {}, {}: rho={:0.3f}, delta={:0.3f}".format(c_nr, c["X"], c["Y"], c["rho"], c["delta"]))

            bin_v,_ = densityclustering.value_per_shell(XY_sct[:,0], XY_sct[:,1], ODI, sct_clusters, bin_size=bin_size, start=distance_range[0], end=distance_range[1])
            bin_values[it,:] = np.nanmean(bin_v,axis=0)
        bin_values = np.nanmean(bin_values,axis=0)
    else:
        print("\nNormal ODIs")
        bin_values, xvalues = densityclustering.value_per_shell(XY[:,0], XY[:,1], ODI, clusters, bin_size=bin_size, start=distance_range[0], end=distance_range[1])
        bin_values = np.nanmean(bin_values,axis=0)
    swap_dist_bin_values[nr,:] = bin_values

# Prepare figure
fig,ax = plottingtools.init_figure(fig_size=(3.5,4))
for nr,swap_dist in enumerate(swap_dists):
    if swap_dist == 0:
        plt.plot(xvalues, swap_dist_bin_values[nr,:], ".-", color=swap_dist_cmap(nr/(len(swap_dists)+1)), markersize=3, zorder=n_swap_dists-nr)
    else:
        plt.plot(xvalues, swap_dist_bin_values[nr,:], ".--", color=swap_dist_cmap(nr/(len(swap_dists)+1)), markersize=3, zorder=n_swap_dists-nr)
plottingtools.finish_panel( ax, title="", ylabel="ODI", xlabel="Distance from cluster center (micron)", legend="off", y_minmax=[-0.2,0.4], y_step=[0.2,1], y_margin=0.05, y_axis_margin=0.03, x_minmax=[0.0,distance_range[1]+10.01], x_step=[100.0,0], x_margin=20, x_axis_margin=5 )

# Save the figure
savefile = os.path.join( savepath, "Fig-S2h-tiled-odi-bins-swapdists-{}".format(args.mousename) )
plottingtools.finish_figure( filename=savefile, wspace=0.2, hspace=0.2 )


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# That's all folks !!
print("\nDone.\n")
