#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This script loads data of a single imaging volume, finds clusters and calculates ODI for clusters based on single trial data

Created on Monday 5 Feb 2024

python singlevolume-OD-L4-clusters-single-trial.py "/Users/pgoltstein/OneDrive/OD Mapping/Data/L4-OD-spikes/" /Users/pgoltstein/figures/ O03

@author: pgoltstein
"""


# Imports
import sys, os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Local imports
sys.path.append('../xx_analysissupport')
import densityclustering
import analysistools
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

parser = argparse.ArgumentParser( description = "This script loads data of a single imaging volume, finds for single trial data clusters and calculates ODI for clusters.\n (written by Pieter Goltstein - Feb 2024)")
parser.add_argument('mousename', type=str, help='name of the mouse to analyze')
args = parser.parse_args()


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Settings

# Path settings
settingspath = "../settings"
savepath = "../../figureout"
datapath = os.path.join("../../data/part1-planedata-od-layer4")
processeddatapath = os.path.join("../../data/part1-processeddata-layer4")
print(f"{datapath=}")
if args.mousename == "O03":
    figname = ["FigS3a","FigS3b","FigS3c"]
if args.mousename == "O09":
    figname = ["FigS3d","FigS3e","FigS3f"]

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
test_bins = [0,4]
ctrl_bins = [4,8]

# Cluster settings
fraction = 0.05
rho_min = 0.2
delta_min = 0.2
rho_x_delta_min = None

# Binning and local shuffle settings
bin_size = 25
distance_range=[0,410]
scatters = list(range(0,251,50))
scatter_cmap = matplotlib.cm.get_cmap("hot")
n_scatter_iter = 10
xvalues_fit = np.arange(distance_range[0],distance_range[1],1)

# Sigmoid fit settings
lead_bins = 5
incl_n_pt = 15 + lead_bins
scale_x_to_prevent_overflow = 100

# Colormaps
trialno_cmap = matplotlib.cm.get_cmap('cool', 256)
trialno_cmap.set_bad(color="#888888")


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Functions

def plot_clusters(clusters, markersize=250, markerlength=30, markerwidth=2, markercolor="#000000"):
    mrkr_outer = int(markersize/2)
    mrkr_inner = (int(markersize/2)-markerlength)
    for c in clusters:
        plt.plot( c["X"], c["Y"], marker="o", markersize=5, markeredgewidth=1.0, markeredgecolor=markercolor, markerfacecolor='None')


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Load data

print("Loading imaging volume of mouse {}".format(args.mousename))
print("  << {} >>".format(datapath))
volume,parameter_names,aspect_ratio,_,tuningmatrix = singlevolumeodfunctions.load_volume( datapath, args.mousename, start_depth=start_depth, depth_increment=depth_increment, skip_first_plane=skip_first_plane, convert_to_micron_x=convert_to_micron_x, convert_to_micron_y=convert_to_micron_y, include_sign=include_sign, exclude_double_xy=exclude_double_xy, exclude_double_z=exclude_double_z, invert_odi_values=invert_odi_values )
print("Loaded parameters:")
for nr,name in enumerate(parameter_names):
    print("{:>2d}: {}".format(nr,name))
min_z = int(np.min(volume[:,parameter_names.index("z")]))
max_z = int(np.max(volume[:,parameter_names.index("z")]))

# Get data
XY = volume[:, [parameter_names.index("x"),parameter_names.index("y")]]
n_neurons,n_eyes,n_oris,n_trials = tuningmatrix.shape
ODI_per_trial = np.zeros((n_neurons,n_trials))
for nr in range(n_neurons):

    # ODI
    for t in range(n_trials):
        ipsi_tc = tuningmatrix[nr,0,:,t].ravel()
        contra_tc = tuningmatrix[nr,1,:,t].ravel()
        ODI_per_trial[nr,t] = analysistools.odi(ipsi_tc, contra_tc, method=0)

ODI = volume[:,parameter_names.index("ODI")]


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ODI correlation across trials
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

fig,ax = plottingtools.init_figure(fig_size=(4,4))
t0=0 # zero-based, is trial 1
t1=8 # zero-based, is trial 9
rand_odi = np.array(ODI_per_trial[:,t1])
np.random.shuffle(rand_odi)

plt.plot(ODI_per_trial[:,t0], rand_odi, ".", color="#888888", markersize=1, linewidth=1.0, zorder=0)
plt.plot(ODI_per_trial[:,t0], ODI_per_trial[:,t1], ".", color="#000000", markersize=1, linewidth=1.0, zorder=0)

ax.axis("equal")
plottingtools.finish_panel( ax, title="Trial {} and {}".format(t0+1,t1+1), ylabel="ODI", xlabel="ODI", legend="off", y_minmax=[-1.0,1.0], y_step=[0.5,1], y_margin=0.06, y_axis_margin=0.03, x_minmax=[-1.0,1.0], x_step=[0.5,1], x_margin=0.06, x_axis_margin=0.03 )

# Save the figure
savefile = os.path.join( savepath, figname[0]+"-{}-trial1-vs-trial9-odi-correlation".format(args.mousename) )
plottingtools.finish_figure( filename=savefile, wspace=0.2, hspace=0.2 )


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ODI maps for single trials
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

fig,ax = plottingtools.init_figure(fig_size=(18*aspect_ratio,6))
for nr,trialno in enumerate([0,2,8]): # zero-based, is trials 1, 3 and 9
    ODI_t = ODI_per_trial[:,trialno]

    # Select data that should be clustered
    XY_ipsi = XY[ODI_t<=0,:]
    XY_contra = XY[ODI_t>0,:]

    # Detect ipsi clusters
    clusters = densityclustering.find_clusters(XY_ipsi, fraction=0.05, rho_min=rho_min, delta_min=delta_min, rho_x_delta_min=rho_x_delta_min, show_rho_vs_delta=False)
    print("Detected {} ipsi clusters".format(len(clusters)))
    for cnr,c in enumerate(clusters):
        print("{}) {}, {}: rho={:0.3f}, delta={:0.3f}".format(cnr, c["X"], c["Y"], c["rho"], c["delta"]))

    # Show ipsi cluster centers on ODI map
    ax = plt.subplot2grid((1,3),(0,nr),fig=fig)
    singlevolumeodfunctions.show_param_2d( ax, XY[:,0], XY[:,1], ODI_t, "ODI map with ipsi clusters (trial {})".format(trialno+1), name=None, cmap="seismic_r", vmin=-1, vmax=1, d1=370, d2=430, size=1 )
    plot_clusters(clusters)

# Save figure
savefile = os.path.join( savepath, figname[1]+"-{}-cellmaps-clusters-singletrials".format(args.mousename) )
plottingtools.finish_figure( filename=savefile, wspace=0.2, hspace=0.2 )




#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Clusters across trials
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

fig,ax = plottingtools.init_figure(fig_size=(6*aspect_ratio,6))

plt.scatter( XY[:,0], XY[:,1], c=np.full_like(ODI, fill_value=0.5), s=1, alpha=1.0, vmin=0, vmax=1, cmap="gray", edgecolors="None" )
ax.axis("equal")
ax.invert_yaxis()
plt.axis("off")

for trialno in range(n_trials):
    ODI_t = ODI_per_trial[:,trialno]

    # Select data that should be clustered
    XY_ipsi = XY[ODI_t<=0,:]
    XY_contra = XY[ODI_t>0,:]

    # Detect ipsi clusters
    clusters = densityclustering.find_clusters(XY_ipsi, fraction=0.05, rho_min=rho_min, delta_min=delta_min, rho_x_delta_min=rho_x_delta_min, show_rho_vs_delta=False)
    print("Detected {} ipsi clusters".format(len(clusters)))
    for nr,c in enumerate(clusters):
        print("{}) {}, {}: rho={:0.3f}, delta={:0.3f}".format(nr, c["X"], c["Y"], c["rho"], c["delta"]))

    # Show contra density
    for c in clusters:
        plt.plot( c["X"], c["Y"], marker="o", markersize=5, markeredgewidth=0.5, markeredgecolor=trialno_cmap(trialno/n_trials), markerfacecolor='None')


# Save figure
savefile = os.path.join( savepath, figname[2]+"-{}-clusters-across-trials".format(args.mousename) )
plottingtools.finish_figure( filename=savefile, wspace=0.2, hspace=0.2 )



fig,ax = plottingtools.init_figure(fig_size=(3,8))

for trialno in range(n_trials):
    plt.plot( 0, trialno, marker="o", markersize=2, markeredgewidth=0.5, markeredgecolor=trialno_cmap(trialno/n_trials), markerfacecolor='None')


# Save figure
savefile = os.path.join( savepath, figname[2]+"-clusters-colors-legend" )
plottingtools.finish_figure( filename=savefile, wspace=0.2, hspace=0.2 )





#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# That's all folks !!
print("\nDone.\n")
# plt.show()
