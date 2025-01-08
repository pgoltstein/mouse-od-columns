#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This script loads data of the L23-L4-L5 imaging volumes of all mice, finds clusters in the L4 subsections of the data, displays sideviews centered on where the clusters are.

Created on Friday 15 Dec 2023

python process-layer2345-sideviews-across-mice.py

@author: pgoltstein
"""

# Global imports
import sys, os
import numpy as np
import sklearn.metrics

# Local imports
sys.path.append('../xx_analysissupport')
import odcfunctions

# Probably shouldn't do this, but got tired of "mean of empty slice" warnings
import warnings
warnings.filterwarnings('ignore')

# Arguments
import argparse

#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Arguments

# Arguments
parser = argparse.ArgumentParser( description = "This script loads data of the L23-L4-L5 imaging volumes of all mice, finds clusters in the L4 subsections of the data, displays sideviews centered on where the clusters are.\n (written by Pieter Goltstein - Dec 2023)")
parser.add_argument('-s', '--sideview', type=str, default="YZ", help= 'Side view orientation XZ or YZ (default=YZ)')
parser.add_argument('-n', '--nrandomizations', type=int, default=10, help='manually set the number of repeats for the local or global randomization control, i.e. the swapping-neuron-locations routine (default=10).')
args = parser.parse_args()

# Settings
settingspath = "../settings"
savepath = "../../figureout"
datapath = os.path.join("../../data/part2-planedata-od-layer2345")
processeddatapath = os.path.join("../../data/part2-sideviewdata-od-layer2345")
print(f"{datapath=}")

mice = ["O02", "O03", "O06", "O07", "O09", "O10", "O11", "O12", "O13"]
settings = odcfunctions.generalsettings("L2345")
settings.cluster.method = "density"
n_shuffles = args.nrandomizations

# max size settings
side_view = args.sideview
max_depth = 700
x_shift = 1200
y_shift = 1000

print("General settings:\n-----------------")
odcfunctions.print_dict(settings)

def scatter_swap_coords_1d(XY, scatter_distance, max_dist=50):
    #swaps by distance aling X only
    XY_sct = np.zeros_like(XY)
    swap_dist = []
    cell_list = np.arange(XY.shape[0])

    # Calculate the distance to all other remaining cells
    XY_1d = np.array(XY)
    XY_1d[:,1] = 1.0
    D = sklearn.metrics.pairwise_distances(XY_1d, metric="euclidean")
    np.fill_diagonal(D, 100000.0)

    max_dist_exceeded = 0
    max_dist_underceeded = 0
    not_done = True
    while not_done:
        # take a random cell
        n1 = np.random.choice(cell_list)

        # find a cell exactly the scatter distance away
        n2 = np.argmin(np.abs(D[n1,:]-scatter_distance))
        swap_dist.append(D[n1,n2])
        if (D[n1,n2]-scatter_distance) > max_dist:
            max_dist_exceeded += 1
        if (D[n1,n2]-scatter_distance) < -max_dist:
            max_dist_underceeded += 1

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
    return XY_sct,swap_dist,max_dist_exceeded,max_dist_underceeded

#  Loop mice
sideview_per_mouse = []
sideview_per_mouse_sh = []
for m_nr,mouse in enumerate(mice):
    msettings = odcfunctions.mousesettings(mouse, datapath, "skip", savepath)
    print("\nMouse specific settings:\n------------------------")
    odcfunctions.print_dict(msettings)

    # -----------------------------
    # Load the data and parameters

    print("Loading OD imaging volume of mouse {}".format(msettings.name))
    print("  << {} >>".format(msettings.datapath_od))
    volume_od,parameter_names_od,aspect_ratio_od,_,_ = odcfunctions.load_volume( settings, msettings, exp_type="od" )

    # Get data
    params = odcfunctions.get_params(volume_od, parameter_names_od, msettings, volume_ret=None, parameter_names_ret=None)

    # Select layer 4 data
    L4_selector = np.logical_and(params.od.Z>=settings.data.z_L4_min, params.od.Z<settings.data.z_L4_max)
    L4_volume_od = volume_od[L4_selector,:]
    L4_params = odcfunctions.get_params(L4_volume_od, parameter_names_od, msettings, volume_ret=None, parameter_names_ret=None)

    # Get L4 clusters
    settings.cluster.type = "ipsi"
    L4_clusters = odcfunctions.find_clusters(L4_params, settings, msettings, v1_mask=None)
    print("Detected {} ipsi clusters".format(len(L4_clusters)))
    for nr,c in enumerate(L4_clusters):
        print("{}) {}, {}: rho={:0.3f}, delta={:0.3f}".format(nr, c["X"], c["Y"], c["rho"], c["delta"]))

    # Get horizontal ODI map per cluster
    mean_map = []
    mean_map_sh = []
    for nr,c in enumerate(L4_clusters):

        if side_view == "XZ":
            XZ = np.array(params.od.XY)
            XZ[:,0] = params.od.XY[:,0] - c["X"] + x_shift
            XZ[:,1] = params.od.Z
            Y_selector = np.logical_and(params.od.XY[:,1]>(c["Y"]-50), params.od.XY[:,1]<(c["Y"]+50))
            ODI = params.od.ODI[Y_selector]
            XZ = XZ[Y_selector,:]

            # Get map
            odi_im,_,odi_mask = odcfunctions.feature_map(ODI, XZ, settings, msettings, smooth_sigma=50, max_x=msettings.max_x+x_shift, max_y=max_depth)

            # Run shuffles
            sh_maps = []
            for sh in range(n_shuffles):
                # np.random.shuffle(ODI)
                XZ,_,_,_ = scatter_swap_coords_1d(XZ, scatter_distance=200, max_dist=50)
                odi_im_sh,_,odi_mask_sh = odcfunctions.feature_map(ODI, XZ, settings, msettings, smooth_sigma=50, max_x=msettings.max_x+x_shift, max_y=max_depth)
                odi_im_sh[~odi_mask_sh] = np.NaN
                sh_maps.append(odi_im_sh)

        if side_view == "YZ":
            YZ = np.array(params.od.XY)
            YZ[:,0] = params.od.XY[:,1] - c["Y"] + y_shift
            YZ[:,1] = params.od.Z
            X_selector = np.logical_and(params.od.XY[:,0]>(c["X"]-50), params.od.XY[:,0]<(c["X"]+50))
            ODI = params.od.ODI[X_selector]
            YZ = YZ[X_selector,:]

            odi_im,_,odi_mask = odcfunctions.feature_map(ODI, YZ, settings, msettings, smooth_sigma=50, max_x=msettings.max_y+y_shift, max_y=max_depth)

            # Run shuffles
            sh_maps = []
            for sh in range(n_shuffles):
                # np.random.shuffle(ODI)
                YZ,_,_,_ = scatter_swap_coords_1d(YZ, scatter_distance=200, max_dist=50)
                odi_im_sh,_,odi_mask_sh = odcfunctions.feature_map(ODI, YZ, settings, msettings, smooth_sigma=50, max_x=msettings.max_y+y_shift, max_y=max_depth)
                odi_im_sh[~odi_mask_sh] = np.NaN
                sh_maps.append(odi_im_sh)

        # Mask the map and add to data container
        odi_im[~odi_mask] = np.NaN
        mean_map.append(odi_im)

        sh_maps = np.stack(sh_maps,axis=2)
        mean_map_sh.append(np.nanmean(sh_maps,axis=2))

    mean_map = np.stack(mean_map,axis=2)
    mean_map = np.nanmean(mean_map,axis=2)
    mean_map_sh = np.stack(mean_map_sh,axis=2)
    mean_map_sh = np.nanmean(mean_map_sh,axis=2)

    # Store mouse data
    sideview_per_mouse.append(mean_map)
    sideview_per_mouse_sh.append(mean_map_sh)

sideview_per_mouse = np.stack(sideview_per_mouse,axis=2)
sideview_per_mouse_sh = np.stack(sideview_per_mouse_sh,axis=2)

# save data
savename = os.path.join( processeddatapath, "od-sideview-{}-cluster-aligned".format(side_view) )
np.savez( savename, sideview_per_mouse=sideview_per_mouse, sideview_per_mouse_sh=sideview_per_mouse_sh, settings=settings, msettings=msettings, aspect_ratio_od=aspect_ratio_od )


# -----------------------------
# Done

print("\nDone.. that's all folks!")
