#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This script identifies ipsi and contra clusters using the geometry method, and additionally identifies the overall v1b area and the intermediate area between clusters, and extracts summary statistics for all the detected areas

Created on Tuesday 21 Nov 2023

python process-od-map-relations.py

@author: pgoltstein
"""

# Global imports
import sys, os
import numpy as np

# Local imports
sys.path.append('../xx_analysissupport')
import odcfunctions
import plottingtools

# Module settings
plottingtools.font_size = { "title": 5, "label": 5, "tick": 5, "text": 5, "legend": 5 }

# Probably shouldn't do this, but got tired of "mean of empty slice" warnings
import warnings
warnings.filterwarnings('ignore')

# Arguments
import argparse

# Arguments
parser = argparse.ArgumentParser( description = "This script identifies ipsi and contra clusters using the geometry method, and additionally identifies the overall v1b area and the intermediate area between clusters, and extracts summary statistics for all the detected areas.\n (written by Pieter Goltstein - Now 2023)")
parser.add_argument('-shodi', '--shuffleodi', type=int, default=-1, help='Flag enables shuffling of odi and sets the shuffle number (-1, or any negative number stands for no shuffle)')
parser.add_argument('-shori', '--shuffleori', type=int, default=-1, help='Flag enables shuffling of preferred orientation and sets the shuffle number (-1, or any negative number stands for no shuffle)')
args = parser.parse_args()

# Settings
mice = ["O02", "O03",]
# mice = ["O02", "O03", "O06", "O07", "O09", "O10", "O11", "O12", "O13"]
savepath = os.path.join("../../data/part3-processeddata-layer4")
settings = odcfunctions.generalsettings("L4")
settings.cluster.method = "od-geometry"
settings.map.smooth_sigma = 42
fn_sh_append = "-sm{}".format(settings.map.smooth_sigma)
if args.shuffleodi > -1:
    settings.data.shuffle = "odi"
    fn_sh_append += "-sh-odi-{}".format(args.shuffleodi)
if args.shuffleori > -1:
    settings.data.shuffle = "ori"
    fn_sh_append += "-sh-ori-{}".format(args.shuffleori)

print("General settings:\n-----------------")
odcfunctions.print_dict(settings)

#  Loop mice
cluster_descriptives = []
for m_nr,mouse in enumerate(mice):
    msettings = odcfunctions.mousesettings(mouse)
    print("\nMouse specific settings:\n------------------------")
    odcfunctions.print_dict(msettings)


    # -----------------------------
    # Load the data and parameters

    print("Loading OD imaging volume of mouse {}".format(msettings.name))
    print("  << {} >>".format(msettings.datapath_od))
    volume_od,parameter_names_od,aspect_ratio_od,_,tuningmatrix_od = odcfunctions.load_volume( settings, msettings, exp_type="od" )

    print("Loading Ret imaging volume of mouse {}".format(msettings.name))
    print("  << {} >>".format(msettings.datapath_ret))
    volume_ret,parameter_names_ret,aspect_ratio_ret,_,tuningmatrix_ret = odcfunctions.load_volume( settings, msettings, exp_type="ret" )

    # Get data
    params = odcfunctions.get_params(volume_od, parameter_names_od, msettings, volume_ret=volume_ret, parameter_names_ret=parameter_names_ret)
    params = odcfunctions.add_ori_params(params, volume_od, parameter_names_od, tuningmatrix_od, msettings)
    fparams = odcfunctions.get_fitted_ret_params(volume_ret, parameter_names_ret, tuningmatrix_ret, settings, msettings)


    # -----------------------------
    # Shuffle ODI if requested

    if settings.data.shuffle.lower() == "odi":
        print("\n!!! Shuffling ODI values !!!")

        # Shuffle the data
        np.random.shuffle( params.od.ODI )


    # -----------------------------
    # Shuffle pref ori if requested

    if settings.data.shuffle.lower() == "ori":
        print("\n!!! Shuffling preferred orientation values !!!")

        # Shuffle the data
        np.random.shuffle( params.od.PD )
        np.random.shuffle( params.od.fPD )


    # -----------------------------
    # Get feature maps

    # Get ODI map, and ODI steepness
    odi_im,_,odi_mask = odcfunctions.feature_map(params.od.ODI, params.od.XY, settings, msettings)
    odi_steepness_im,odi_grad_ang_deg = odcfunctions.calculate_gradient_magnitude_angle_unit_pix(odi_im, settings)

    # Get retinotopic map
    azi_im,_,azi_mask = odcfunctions.feature_map(params.ret.AZI, params.ret.XY, settings, msettings)
    ele_im,_,ele_mask = odcfunctions.feature_map(params.ret.ELE, params.ret.XY, settings, msettings)

    # Get the cortical magnification factor and retinotopy angle
    cmf, cmf_azi, cmf_ele, azi_angle, ele_angle, azi_ele_ang, azi_ele_ratio = odcfunctions.calculate_cmf(azi_im, ele_im, settings)

    # Get map of RF size
    width_im,_,width_mask = odcfunctions.feature_map(fparams.ret.WDTH, fparams.ret.XY, settings, msettings)

    # Calculate Rs and M
    Rs_r0, Rs_w, Mmax_Mmin, all_coords = odcfunctions.calculate_Rs_and_Mmax_Mmin_per_pixel(cmf_azi, cmf_ele, azi_angle, ele_angle )

    # Calculate orientation map
    ori_im_ang,ori_im_len,_,dir_mask = odcfunctions.circ_feature_map(np.mod(params.od.fPD*2,360), params.od.XY, settings, msettings)
    ori_im_ang = ori_im_ang / 2

    # Get the gradient of fPO
    ori_grad_mag, ori_grad_ang_deg = odcfunctions.circ_calculate_gradient_magnitude_angle_unit_pix(ori_im_ang*2, settings)

    # Get a map with the orientation of the iso-ODI lines
    iso_odi_ori_deg,_ = odcfunctions.get_iso_odi_from_odi_angle_map(odi_grad_ang_deg)

    # Calculate the angle between Rs and the iso-ODI lines
    Rs_w_to_iso_odi = odcfunctions.angle_between_180deg_maps(Rs_w, iso_odi_ori_deg)

    # Calculate the angle between ori and ODI gradient
    ori_grad_ang_deg_180 = np.mod(ori_grad_ang_deg,180)
    odi_grad_ang_deg_180 = np.mod(odi_grad_ang_deg,180)
    ori_to_odi_angle = odcfunctions.angle_between_180deg_maps(ori_grad_ang_deg_180, odi_grad_ang_deg_180)


    # -----------------------------
    # Get masks

    # Get V1 mask
    v1_mask, v1_contour = odcfunctions.get_v1_mask(params, settings, msettings)

    # Get mask across maps and v1
    overall_msk = odcfunctions.mask_maps( None, [odi_mask,azi_mask,ele_mask,v1_mask])


    # -----------------------------
    # Get clusters

    # Detect ipsi clusters
    settings.cluster.type = "ipsi"
    ipsi_clusters, ipsi_cluster_props, ipsi_cluster_contours = odcfunctions.find_clusters(params, settings, msettings, v1_mask=overall_msk )

    # Detect ipsi clusters
    settings.cluster.type = "contra"
    contra_clusters, contra_cluster_props, contra_cluster_contours = odcfunctions.find_clusters(params, settings, msettings, v1_mask=overall_msk )


    # -----------------------------
    # Get ipsi cluster descriptives

    ipsi_cluster_data = odcfunctions.quantify_od_patterns(ipsi_cluster_props, params, fparams, odi_im, odi_steepness_im, azi_im, ele_im, cmf, cmf_azi, cmf_ele, azi_angle, ele_angle, azi_ele_ang, azi_ele_ratio, width_im, Rs_r0, Mmax_Mmin, Rs_w_to_iso_odi, ori_im_ang, ori_im_len, ori_to_odi_angle, msettings, cluster_type=1)
    for cnr,cdat in enumerate(ipsi_cluster_data):
        print("Cluster {}) type={:1.0f}, length={:3.0f}, width={:3.0f}, area={:5.0f}, angle={:3.0f}, ecc={:0.2f}, odi={:5.2f}, cmf={:0.5f}, rfsize={:4.1f}".format(cnr, cdat.type, cdat.length, cdat.width, cdat.area, cdat.angle, cdat.ecc, cdat.odi, cdat.cmf, cdat.rfsize))
    cluster_descriptives.extend( ipsi_cluster_data )


    # -----------------------------
    # Get contra cluster descriptives

    contra_cluster_data = odcfunctions.quantify_od_patterns(contra_cluster_props, params, fparams, odi_im, odi_steepness_im, azi_im, ele_im, cmf, cmf_azi, cmf_ele, azi_angle, ele_angle, azi_ele_ang, azi_ele_ratio, width_im, Rs_r0, Mmax_Mmin, Rs_w_to_iso_odi, ori_im_ang, ori_im_len, ori_to_odi_angle, msettings, cluster_type=2)
    for cnr,cdat in enumerate(contra_cluster_data):
        print("Cluster {}) type={:1.0f}, length={:3.0f}, width={:3.0f}, area={:5.0f}, angle={:3.0f}, ecc={:0.2f}, odi={:5.2f}, cmf={:0.5f}, rfsize={:4.1f}".format(cnr, cdat.type, cdat.length, cdat.width, cdat.area, cdat.angle, cdat.ecc, cdat.odi, cdat.cmf, cdat.rfsize))
    cluster_descriptives.extend( contra_cluster_data )


    # -----------------------------
    # Get non-cluster V1b descriptives

    intercluster_msk = odcfunctions.remove_regions_from_mask(overall_msk, ipsi_cluster_props)
    intercluster_msk = odcfunctions.remove_regions_from_mask(intercluster_msk, contra_cluster_props)
    intercluster_props = odcfunctions.get_largest_region(intercluster_msk)
    inter_cluster_data = odcfunctions.quantify_od_patterns(intercluster_props, params, fparams, odi_im, odi_steepness_im, azi_im, ele_im, cmf, cmf_azi, cmf_ele, azi_angle, ele_angle, azi_ele_ang, azi_ele_ratio, width_im, Rs_r0, Mmax_Mmin, Rs_w_to_iso_odi, ori_im_ang, ori_im_len, ori_to_odi_angle, msettings, cluster_type=3)
    for cnr,cdat in enumerate(inter_cluster_data):
        print("Cluster {}) type={:1.0f}, length={:3.0f}, width={:3.0f}, area={:5.0f}, angle={:3.0f}, ecc={:0.2f}, odi={:5.2f}, cmf={:0.5f}, rfsize={:4.1f}".format(cnr, cdat.type, cdat.length, cdat.width, cdat.area, cdat.angle, cdat.ecc, cdat.odi, cdat.cmf, cdat.rfsize))
    cluster_descriptives.extend( inter_cluster_data )


    # -----------------------------
    # Get overall V1b descriptives

    v1b_props = odcfunctions.get_largest_region(overall_msk)
    v1b_data = odcfunctions.quantify_od_patterns(v1b_props, params, fparams, odi_im, odi_steepness_im, azi_im, ele_im, cmf, cmf_azi, cmf_ele, azi_angle, ele_angle, azi_ele_ang, azi_ele_ratio, width_im, Rs_r0, Mmax_Mmin, Rs_w_to_iso_odi, ori_im_ang, ori_im_len, ori_to_odi_angle, msettings, cluster_type=4)
    for cnr,cdat in enumerate(v1b_data):
        print("Cluster {}) type={:1.0f}, length={:3.0f}, width={:3.0f}, area={:5.0f}, angle={:3.0f}, ecc={:0.2f}, odi={:5.2f}, cmf={:0.5f}, rfsize={:4.1f}".format(cnr, cdat.type, cdat.length, cdat.width, cdat.area, cdat.angle, cdat.ecc, cdat.odi, cdat.cmf, cdat.rfsize))
    cluster_descriptives.extend( v1b_data )


# Save data as list of named tuples
cluster_descriptives = np.array(cluster_descriptives, dtype=object)
savename = os.path.join( savepath, "od-clusters-descriptives{}.npz".format(fn_sh_append) )
np.savez( savename, cluster_descriptives=cluster_descriptives, settings=settings, msettings=msettings, allow_pickle=True )


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# That's all folks !!
print("\nDone.\n")
