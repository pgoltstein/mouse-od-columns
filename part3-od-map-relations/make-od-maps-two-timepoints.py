#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This script plots OD maps for all animals in a PDF

Created on Monday 11 Mar 2024

python make-od-maps-two-timepoints.py

@author: pgoltstein
"""

# Global imports
import sys, os
import numpy as np
from skimage.registration import phase_cross_correlation
from scipy.ndimage import fourier_shift

# Local imports
sys.path.append('../xx_analysissupport')
import odcfunctions
import plottingtools

# Module settings
plottingtools.font_size = { "title": 5, "label": 5, "tick": 5, "text": 5, "legend": 5 }

# Probably shouldn't do this, but got tired of "mean of empty slice" warnings
import warnings
warnings.filterwarnings('ignore')

#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Settings

# Mice to show
mice = ["O02", "O03", "O06", "O07", "O09", "O10", "O11", "O12", "O13"]
datapath_od = "../../data/part1-planedata-od-layer4",

# Settings for L4 volume
settings_L4 = odcfunctions.generalsettings("L4")
settings_L4.data.include_z_start = 370
settings_L4.data.include_z_end = 431
settings_L4.cluster.method = "od-geometry"
print("General settings L4:\n-----------------")
odcfunctions.print_dict(settings_L4)

# Settings for L4 part of L2345 volume
settings_L2345 = odcfunctions.generalsettings("L2345")
settings_L2345.data.include_z_start = 370
settings_L2345.data.include_z_end = 431
settings_L2345.cluster.method = "od-geometry"
print("General settings settings L2345:\n-----------------")
odcfunctions.print_dict(settings_L2345)

#  Loop mice
subplot_odi_cnt = 1
subplot_ori_cnt = 1
fig_odi = plottingtools.init_fig(fig_size=(30,15))
for m_nr,mouse in enumerate(mice):
    
    msettings_L4= odcfunctions.mousesettings(mousename=mouse)
    print("\nMouse specific settings L4:\n------------------------")
    odcfunctions.print_dict(msettings_L4)
    
    msettings_L2345 = odcfunctions.mousesettings(mousename=mouse, datapath_od="../../data/part2-planedata-od-layer2345", datapath_ret="None")
    print("\nMouse specific settings L2345:\n------------------------")
    odcfunctions.print_dict(msettings_L2345)


    # -----------------------------
    # Load the data and parameters

    print("Loading L4 OD imaging volume of mouse {}".format(msettings_L4.name))
    print("  << {} >>".format(msettings_L4.datapath_od))
    volume_od_L4,parameter_names_od_L4,aspect_ratio_od_L4,all_ims_L4,tuningmatrix_od_L4 = odcfunctions.load_volume( settings_L4, msettings_L4, exp_type="od" )

    print("Loading Ret imaging volume of mouse {}".format(msettings_L4.name))
    print("  << {} >>".format(msettings_L4.datapath_ret))
    volume_ret,parameter_names_ret,aspect_ratio_ret,_,tuningmatrix_ret = odcfunctions.load_volume( settings_L4, msettings_L4, exp_type="ret" )

    print("Loading L2345 OD imaging volume of mouse {}".format(msettings_L2345.name))
    print("  << {} >>".format(msettings_L2345.datapath_od))
    volume_od_L2345,parameter_names_od_L2345,aspect_ratio_od_L2345,all_ims_L2345,tuningmatrix_od_L2345 = odcfunctions.load_volume( settings_L2345, msettings_L2345, exp_type="od" )

    # Get basic parameters
    params_L4 = odcfunctions.get_params(volume_od_L4, parameter_names_od_L4, msettings_L4, volume_ret=volume_ret, parameter_names_ret=parameter_names_ret)
    params_L2345 = odcfunctions.get_params(volume_od_L2345, parameter_names_od_L2345, msettings_L2345, volume_ret=None, parameter_names_ret=None)


    # -----------------------------
    # Get rigid position shift between stacks

    im_L4 = np.mean(np.stack(all_ims_L4,axis=2),axis=2)
    im_L2345 = np.mean(np.stack(all_ims_L2345,axis=2),axis=2)

    # Get X-Y image shift from L4 to L2345 image
    shift_yx, error, diffphase = phase_cross_correlation(im_L2345, im_L4, upsample_factor=1)
    print("YX shift of L4 relative to L2345: {}".format(shift_yx))


    # -----------------------------
    # Get feature maps
    print("Creating maps")

    # Get ODI map
    odi_im_L4,_,odi_mask_L4 = odcfunctions.feature_map(params_L4.od.ODI, params_L4.od.XY, settings_L4, msettings_L4)
    odi_im_L2345,_,odi_mask_L2345 = odcfunctions.feature_map(params_L2345.od.ODI, params_L2345.od.XY, settings_L2345, msettings_L2345)

    # Get retinotopic map
    azi_im,_,azi_mask = odcfunctions.feature_map(params_L4.ret.AZI, params_L4.ret.XY, settings_L4, msettings_L4)
    ele_im,_,ele_mask = odcfunctions.feature_map(params_L4.ret.ELE, params_L4.ret.XY, settings_L4, msettings_L4)


    # -----------------------------
    # Get masks

    # Get V1 mask
    v1_mask_L4, v1_contour_L4 = odcfunctions.get_v1_mask(params_L4, settings_L4, msettings_L4)

    # Realign mask for L2345 data
    v1_mask_L2345_i = fourier_shift(np.fft.fftn(v1_mask_L4*1.0), shift_yx)
    v1_mask_L2345 = np.fft.ifftn(v1_mask_L2345_i)
    v1_mask_L2345 = v1_mask_L2345.real > 0.5

    # Realign contour for L2345 data
    v1_contour_L2345 = []
    for ix in range(len(v1_contour_L4)):
        if isinstance(v1_contour_L4[ix], list):
            v1_contour_L2345.append([])
            for c in range(len(v1_contour_L4[ix])):
                conts_L2345 = np.zeros_like(v1_contour_L4[ix][c])
                conts_L2345[:,1] = v1_contour_L4[ix][c][:,1] + shift_yx[1] # x
                conts_L2345[:,0] = v1_contour_L4[ix][c][:,0] + shift_yx[0] # y
                v1_contour_L2345[ix].append(conts_L2345)
        else:
            conts_L2345 = np.zeros_like(v1_contour_L4[ix])
            conts_L2345[:,1] = v1_contour_L4[ix][:,1] + shift_yx[1] # x
            conts_L2345[:,0] = v1_contour_L4[ix][:,0] + shift_yx[0] # y
            v1_contour_L2345.append(conts_L2345)

    # Get mask across maps and v1
    overall_msk_L4 = odcfunctions.mask_maps( None, [odi_mask_L4,v1_mask_L4])
    overall_msk_L2345 = odcfunctions.mask_maps( None, [odi_mask_L2345,v1_mask_L2345])

    # Mask maps
    [odi_im_masked_L4,],_ = \
            odcfunctions.mask_maps([odi_im_L4,], [odi_mask_L4,])
    [odi_im_masked_L2345,],_ = odcfunctions.mask_maps([odi_im_L2345,], [odi_mask_L2345,])


    # -----------------------------
    # Get clusters

    # Detect ipsi clusters
    settings_L4.cluster.type = "ipsi"
    ipsi_clusters_L4, ipsi_cluster_props_L4, ipsi_cluster_contours_L4 = \
            odcfunctions.find_clusters(params_L4, settings_L4, msettings_L4, v1_mask=overall_msk_L4 )
    settings_L2345.cluster.type = "ipsi"
    ipsi_clusters_L2345, ipsi_cluster_props_L2345, ipsi_cluster_contours_L2345 = \
            odcfunctions.find_clusters(params_L2345, settings_L2345, msettings_L2345, v1_mask=overall_msk_L2345 )

    # Detect contra clusters
    settings_L4.cluster.type = "contra"
    contra_clusters_L4, contra_cluster_props_L4, contra_cluster_contours_L4 = \
            odcfunctions.find_clusters(params_L4, settings_L4, msettings_L4, v1_mask=overall_msk_L4 )
    settings_L2345.cluster.type = "contra"
    contra_clusters_L2345, contra_cluster_props_L2345, contra_cluster_contours_L2345 = \
            odcfunctions.find_clusters(params_L2345, settings_L2345, msettings_L2345, v1_mask=overall_msk_L2345 )


    # -----------------------------
    # Make the plot showing ODI maps

    # Show OD of 'first' experiment (L2345)
    ax = fig_odi.add_subplot(3,6,subplot_odi_cnt)
    im_handle = ax.imshow(odi_im_masked_L2345, cmap="seismic_r", vmin=-0.5, vmax=0.5, interpolation="None")
    odcfunctions.plot_clusters(clusters=None, cluster_contours=ipsi_cluster_contours_L2345, marker="^", markersize=3, contour_linestyle="-", contour_linewidth=0.25, ax=ax)
    odcfunctions.plot_clusters(clusters=None, cluster_contours=contra_cluster_contours_L2345, marker="s", markersize=3, markercolor="#ffffff",
                      contour_linestyle="-", contour_linewidth=0.5, contour_color="#ffffff", ax=ax )
    odcfunctions.plot_iso_contours(v1_contour_L2345, "V1", settings_L4, msettings_L4, linewidth=0.5, ax=ax)
    ax.set_title("L2345 ODI map ({})".format(msettings_L2345.name), size=5)
    ax.axis('off')
    subplot_odi_cnt += 1
    
    # Show OD of 'second' experiment (L4)
    ax = fig_odi.add_subplot(3,6,subplot_odi_cnt)
    im_handle = ax.imshow(odi_im_masked_L4, cmap="seismic_r", vmin=-0.5, vmax=0.5, interpolation="None")
    odcfunctions.plot_clusters(clusters=None, cluster_contours=ipsi_cluster_contours_L4, marker="^", markersize=3, contour_linestyle="-", contour_linewidth=0.25, ax=ax)
    odcfunctions.plot_clusters(clusters=None, cluster_contours=contra_cluster_contours_L4, marker="s", markersize=3, markercolor="#ffffff",
                      contour_linestyle="-", contour_linewidth=0.5, contour_color="#ffffff", ax=ax )
    odcfunctions.plot_iso_contours(v1_contour_L4, "V1", settings_L4, msettings_L4, linewidth=0.5, ax=ax)
    ax.set_title("L4 ODI map ({})".format(msettings_L4.name), size=5)
    ax.axis('off')
    subplot_odi_cnt += 1


# Save figures
savename = os.path.join( msettings_L4.savepath, "Fig-S12c-od-maps-two-timepoints.pdf" )
fig_odi.subplots_adjust( wspace=0.2, hspace=0.2 )
fig_odi.savefig(savename, transparent=True)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# That's all folks !!
print("\nDone.\n")
