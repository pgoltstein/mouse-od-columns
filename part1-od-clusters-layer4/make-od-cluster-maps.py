#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This script loads data of a single imaging volume, shows OD and eye-preference density, and finds clusters and calculates ODI for clusters

Created on Monday 14 Dec 2024

python make-od-cluster-maps.py O03

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

parser = argparse.ArgumentParser( description = "This script loads data of a single imaging volume, shows OD and eye-preference density, and finds clusters and calculates ODI for clusters.\n (written by Pieter Goltstein - Dec 2024)")
parser.add_argument('mouse', type=str, help= 'name of the mouse to analyze')
parser.add_argument('-si', '--supplementaryinformation',  action="store_true", default=False, help='Flag for scaling y-axis and doing swap control for supplementary information')
parser.add_argument('-n', '--nswaps', type=int, default=100, help='manually set the number of repeats for the local randomization control, i.e. the swapping-neuron-locations routine (default=100).')
args = parser.parse_args()


#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Settings

# Path settings
settingspath = "../settings"
savepath = "../../figureout"
datapath = os.path.join("../../data/part1-planedata-od-layer4")
print(f"{datapath=}")
if args.mouse == "O03":
    figname = "Fig-1"
elif int(args.mouse[1:]) < 20:
    figname = "Fig-S1"
else:
    figname = "Fig-S7"

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
if int(args.mouse[1:]) < 20:
    convert_to_micron_x = 1192/1024
    convert_to_micron_y = 1019/1024
    invert_odi_values = False
    test_bins = [0,4]
    ctrl_bins = [4,8]
    if args.mouse == "O03":
        y_scale = [-0.25,0.25]
    else:
        y_scale = [-0.4,0.4]
else:
    convert_to_micron_x = 1180/1024
    convert_to_micron_y = 982/1024
    invert_odi_values = True
    test_bins = [0,3]
    ctrl_bins = [3,6]
    y_scale = [-0.4,0.4]

# Cluster settings
fraction = 0.05
rho_min = 0.2
delta_min = 0.2
rho_x_delta_min = None

# Binning and local shuffle settings
bin_size = 25
distance_range=[0,410]
swaps = list(range(0,251,50))
swap_cmap = matplotlib.cm.get_cmap("hot")
xvalues_fit = np.arange(distance_range[0],distance_range[1],1)

# Sigmoid fit settings
lead_bins = 5
incl_n_pt = 15 + lead_bins
scale_x_to_prevent_overflow = 100


#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Load data

print("Loading imaging volume of mouse {}".format(args.mouse))
print("  << {} >>".format(datapath))
volume,parameter_names,aspect_ratio,_,_ = singlevolumeodfunctions.load_volume( datapath, args.mouse, start_depth=start_depth, depth_increment=depth_increment, skip_first_plane=skip_first_plane, convert_to_micron_x=convert_to_micron_x, convert_to_micron_y=convert_to_micron_y, include_sign=include_sign, exclude_double_xy=exclude_double_xy, exclude_double_z=exclude_double_z, invert_odi_values=invert_odi_values )
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
# Show a histogram of OD values

if not args.supplementaryinformation:
    m,e,_ = statstools.mean_sem(ODI)
    fig,ax = plottingtools.init_figure(fig_size=(6,4))
    ax = sns.histplot(data=ODI, ax=ax, binwidth=0.1, stat="probability", color="#999999")
    plottingtools.finish_panel( ax, title="Mean ODI of tuned neurons: {:5.3f} (±{:5.3f}))".format( m, e ), ylabel="p", xlabel="ODI", legend="off", y_minmax=[0,0.151], y_step=[0.05,2], y_margin=0.0, y_axis_margin=0.0, x_minmax=[-1.0,1.01], x_step=[0.5,1], x_margin=0.05, x_axis_margin=0.02, x_ticks=None, x_ticklabels=None, y_ticks=None, y_ticklabels=None, x_tick_rotation=0, tick_size=6, label_size=6, title_size=6, legend_size=6, despine=True, legendpos=0)
    savefile = os.path.join( savepath, figname+"d-{}-odi-histogram".format(args.mouse) )
    print("Saving ODI histogram to file: {}".format(savepath+".pdf"))
    plottingtools.finish_figure( filename=savefile, wspace=0.2, hspace=0.2 )


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
        plt.plot( c["X"], c["Y"], marker="o", markersize=10, markeredgewidth=2.0, markeredgecolor=markercolor, markerfacecolor='None')

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
fig,ax = plottingtools.init_figure(fig_size=(18*aspect_ratio,12))

# Show contra density
ax = plt.subplot2grid((2,3),(0,0),fig=fig)
plt.imshow(grey_bg,cmap="Greys",vmin=0,vmax=1)
singlevolumeodfunctions.show_param_2d( ax, XY_contra[:,0], XY_contra[:,1], rho_contra, "Contra preferring cells", name=None, cmap="magma", vmin=0, vmax=1, d1=min_z, d2=max_z, size=2 ) # color coded rho
plot_clusters(clusters)
plt.gca().invert_yaxis()

# Show ipsi density
ax = plt.subplot2grid((2,3),(1,0),fig=fig)
plt.imshow(grey_bg,cmap="Greys",vmin=0,vmax=1)
singlevolumeodfunctions.show_param_2d( ax, XY_ipsi[:,0], XY_ipsi[:,1], rho_ipsi, "Ipsi preferring cells", name=None, cmap="magma", vmin=0, vmax=1, d1=min_z, d2=max_z, size=2  ) # color coded rho
plot_clusters(clusters)
plt.gca().invert_yaxis()

# Show ipsi cluster centers on ODI map
ax = plt.subplot2grid((2,3),(0,1),fig=fig, rowspan=2, colspan=2)
plt.imshow(grey_bg,cmap="Greys",vmin=0,vmax=1)
singlevolumeodfunctions.show_param_2d( ax, XY[:,0], XY[:,1], ODI, "ODI map with ipsi clusters", name=None, cmap="seismic_r", vmin=-1, vmax=1, d1=370, d2=430, size=4 )
plot_clusters(clusters)
for odi_ix in range(len(odi_contours)):
    for c in range(len(odi_contours[odi_ix])):
        plt.plot(odi_contours[odi_ix][c][:,1],odi_contours[odi_ix][c][:,0], iso_odi_contour_linestyle[odi_ix], color="#ffffff",  markersize=0, linewidth=0.5)
plt.gca().invert_yaxis()

# Save figure
if not args.supplementaryinformation:
    savefile = os.path.join( savepath, figname+"c-{}-od-cellmaps-clusters".format(args.mouse) )
else:
    savefile = os.path.join( savepath, figname+"bc-{}-od-cellmaps-clusters".format(args.mouse) )
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
if not args.supplementaryinformation:

    fig,ax = plottingtools.init_figure(fig_size=(3.5,3))
    for c_nr in range(len(clusters)):
        plt.plot(xvalues, odi_per_bin[c_nr,:], ".-", color="#888888",  markersize=1, linewidth=0.5, zorder=1)
    plt.plot(xvalues, np.mean(odi_per_bin,axis=0), ".-", color="#000000", markersize=3, linewidth=1.0, zorder=3)
    plt.plot(xvalues_fit, odi_fit, "-", color="#880000", linewidth=0.5, zorder=4)
    plottingtools.finish_panel( ax, title="", ylabel="ODI", xlabel="Distance from cluster center (micron)", legend="off", y_minmax=y_scale, y_step=[0.25,2], y_margin=0.04, y_axis_margin=0.01, x_minmax=[0.0,distance_range[1]+10.01], x_step=[100.0,0], x_margin=20, x_axis_margin=5 )

    # Save the figure
    savefile = os.path.join( savepath, figname+"e-{}-odi-from-cluster-center".format(args.mouse) )
    plottingtools.finish_figure( filename=savefile, wspace=0.2, hspace=0.2 )

else:

    fig,ax = plottingtools.init_figure(fig_size=(3,3))
    for c_nr in range(len(clusters)):
        plt.plot(xvalues, odi_per_bin[c_nr,:], ".-", color="#888888",  markersize=1, linewidth=0.5, zorder=1)
    plt.plot(xvalues, np.mean(odi_per_bin,axis=0), ".-", color="#000000", markersize=3, linewidth=1.0, zorder=3)
    plt.plot(xvalues_fit, odi_fit, "-", color="#880000", linewidth=0.5, zorder=4)
    plottingtools.finish_panel( ax, title="", ylabel="ODI", xlabel="Distance from cluster center (micron)", legend="off", y_minmax=y_scale, y_step=[0.2,1], y_margin=0.2, y_axis_margin=0.03, x_minmax=[0.0,distance_range[1]+10.01], x_step=[100.0,0], x_margin=15, x_axis_margin=5 )
    ax.tick_params('both', length=2, width=1, which='major')

    # Save the figure
    savefile = os.path.join( savepath, figname+"e-{}-odi-from-cluster-center-for-ed".format(args.mouse) )
    plottingtools.finish_figure( filename=savefile, wspace=0.2, hspace=0.2 )


#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# If supplementary information flag is set, then do the 'swap neurons' control

if args.supplementaryinformation:

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Functions

    def swap_swap_coords(XY, swap_distance, max_dist=50):
        XY_sct = np.zeros_like(XY)
        swap_dist = []
        cell_list = np.arange(XY.shape[0])

        # Calculate the distance to all other remaining cells
        D = sklearn.metrics.pairwise_distances(XY, metric="euclidean")
        np.fill_diagonal(D, 100000.0)

        max_dist_exceeded = 0
        not_done = True
        while not_done:
            # print("len(cell_list)={}".format(len(cell_list)))

            # take a random cell
            n1 = np.random.choice(cell_list)

            # find a cell exactly the swap distance away
            n2 = np.argmin(np.abs(D[n1,:]-swap_distance))
            swap_dist.append(D[n1,n2])
            if np.abs(D[n1,n2]-swap_distance) > max_dist:
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
        return XY_sct,swap_dist,max_dist_exceeded


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Show plot with ODI from cluster center as function of swap distance

    # Prepare variables
    n_bins = len(np.arange(distance_range[0],distance_range[1],bin_size))-1
    n_swaps = len(swaps)
    swap_bin_values = np.zeros(( n_swaps, n_bins ))

    # Loop swaps and get values in matrix
    for nr,swap in enumerate(swaps):
        if swap > 0:
            bin_values = np.zeros(( args.nswaps, n_bins ))
            for it in range(args.nswaps):

                # Swap coordinates of neurons ar specific distance
                XY_sct,swap_dist,max_dist_exceeded = swap_swap_coords(XY, swap_distance=swap)

                # Recalculate clusters
                XY_sct_ipsi = XY_sct[ODI<=0,:]
                sct_clusters = densityclustering.find_clusters(XY_sct_ipsi, fraction=0.05, rho_min=rho_min, delta_min=delta_min, rho_x_delta_min=rho_x_delta_min, show_rho_vs_delta=False)
                print("Detected {} ipsi clusters (swap={})".format(len(sct_clusters),swap))
                print("Exceeded distance threshold {}x".format(max_dist_exceeded))
                for c_nr,c in enumerate(sct_clusters):
                    print("{}) {}, {}: rho={:0.3f}, delta={:0.3f}".format(c_nr, c["X"], c["Y"], c["rho"], c["delta"]))

                bin_v,_ = densityclustering.value_per_shell(XY_sct[:,0], XY_sct[:,1], ODI, sct_clusters, bin_size=bin_size, start=distance_range[0], end=distance_range[1])
                bin_values[it,:] = np.nanmean(bin_v,axis=0)
            bin_values = np.nanmean(bin_values,axis=0)
        else:
            bin_values, xvalues = densityclustering.value_per_shell(XY[:,0], XY[:,1], ODI, clusters, bin_size=bin_size, start=distance_range[0], end=distance_range[1])
            bin_values = np.nanmean(bin_values,axis=0)
        swap_bin_values[nr,:] = bin_values

    # Make figure
    fig,ax = plottingtools.init_figure(fig_size=(3,3))
    for nr,swap in enumerate(swaps):
        if swap == 0:
            plt.plot(xvalues, swap_bin_values[nr,:], ".-", color=swap_cmap(nr/(len(swaps)+1)), markersize=3, zorder=n_swaps-nr)
        else:
            plt.plot(xvalues, swap_bin_values[nr,:], ".--", color=swap_cmap(nr/(len(swaps)+1)), markersize=2, zorder=n_swaps-nr)
    plottingtools.finish_panel( ax, title="", ylabel="ODI", xlabel="Distance from cluster center (micron)", legend="off", y_minmax=[-0.4,0.4], y_step=[0.2,1], y_margin=0.06, y_axis_margin=0.03, x_minmax=[0.0,distance_range[1]+10.01], x_step=[100.0,0], x_margin=15, x_axis_margin=5 )
    ax.tick_params('both', length=2, width=1, which='major')

    # Save the figure
    savefile = os.path.join( savepath, figname+"f-{}-odi-from-cluster-center-swapcontrol".format(args.mouse) )
    plottingtools.finish_figure( filename=savefile, wspace=0.2, hspace=0.2 )


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# That's all folks !!
print("\nDone.\n")
