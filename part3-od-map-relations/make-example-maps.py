#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This script loads OD and retinotopy data of an L4 imaging volume and creates maps for ocular dominance, azimuth, elevation, preferred orientation, and several computed features

Created on Tuesday 8 Jan 2025

python make-example-maps.py

@author: pgoltstein
"""

# Global imports
import sys, os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import skimage.measure

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

settings = odcfunctions.generalsettings("L4")
settings.data.include_z_start = 370
settings.data.include_z_end = 431
settings.cluster.method = "od-geometry"
settings.map.smooth_sigma = 42
print("General settings:\n-----------------")
odcfunctions.print_dict(settings)
msettings = odcfunctions.mousesettings("O03")
print("\nMouse specific settings:\n------------------------")
odcfunctions.print_dict(msettings)
settings_ret = settings

# Colorbar for orientation preference scaled by resultant length
cmap = matplotlib.cm.get_cmap("hsv")
ori_bar_ang = np.zeros((180,50))
ori_bar_len = np.zeros((180,50))
for i in range(180):
    ori_bar_ang[i,:] = i
for i in range(50):
    ori_bar_len[:,i] = (i/50)
ori_bar_ang_rgba = cmap(ori_bar_ang/180)
ori_bar_ang_rgba[:,:,3] = ori_bar_len
gray_bg_bar_rgba = np.ones_like(ori_bar_ang_rgba)
gray_bg_bar_rgba[:,:,:3] = 0.5

# Continuous azimuth and elevation colorbars
azi_bar = np.zeros((256,50))
ele_bar = np.zeros((256,50))
for i in range(256):
    azi_bar[i,:] = i
    ele_bar[i,:] = i
azi_bar_rgba = settings.cmap.azi(azi_bar/256)
ele_bar_rgba = settings.cmap.ele(ele_bar/256)


#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Load the data

print("Loading OD imaging volume of mouse {}".format(msettings.name))
print("  << {} >>".format(msettings.datapath_od))
volume_od,parameter_names_od,aspect_ratio_od,_,tuningmatrix_od = odcfunctions.load_volume( settings, msettings, exp_type="od" )

print("Loading Ret imaging volume of mouse {}".format(msettings.name))
print("  << {} >>".format(msettings.datapath_ret))
volume_ret,parameter_names_ret,aspect_ratio_ret,_,tuningmatrix_ret = odcfunctions.load_volume( settings_ret, msettings, exp_type="ret" )

# Get data
params = odcfunctions.get_params(volume_od, parameter_names_od, msettings, volume_ret=volume_ret, parameter_names_ret=parameter_names_ret)
params = odcfunctions.add_ori_params(params, volume_od, parameter_names_od, tuningmatrix_od, msettings)
fparams = odcfunctions.get_fitted_ret_params(volume_ret, parameter_names_ret, tuningmatrix_ret, settings, msettings)

print("Azimuths: {}".format(np.unique(params.ret.AZI)))
print("Elevations: {}".format(np.unique(params.ret.ELE)))


#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Calculate the maps

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
ori_im_ang,ori_im_len,_,dir_mask = odcfunctions.circ_feature_map(np.mod(params.od.PD*2,360), params.od.XY, settings, msettings)
ori_im_ang = ori_im_ang / 2

# Get the gradient of PO
ori_grad_mag, ori_grad_ang_deg = odcfunctions.circ_calculate_gradient_magnitude_angle_unit_pix(ori_im_ang*2, settings)

# Get a map with the orientation of the iso-ODI lines
iso_odi_ori_deg,_ = odcfunctions.get_iso_odi_from_odi_angle_map(odi_grad_ang_deg)

# Calculate the angle between Rs and the iso-ODI lines
Rs_w_to_iso_odi = odcfunctions.angle_between_180deg_maps(Rs_w, iso_odi_ori_deg)

# Calculate the angle between ori and ODI gradient
ori_grad_ang_deg_180 = np.mod(ori_grad_ang_deg,180)
odi_grad_ang_deg_180 = np.mod(odi_grad_ang_deg,180)
ori_to_odi_angle = odcfunctions.angle_between_180deg_maps(ori_grad_ang_deg_180, odi_grad_ang_deg_180)

# Get V1 mask
v1_mask, v1_contour = odcfunctions.get_v1_mask(params, settings, msettings)

# Get mask across maps and v1
overall_msk = odcfunctions.mask_maps( None, [odi_mask,azi_mask,ele_mask,v1_mask])

# Mask maps
[odi_im_masked, azi_im_masked, ele_im_masked, ori_im_ang_masked, ori_im_len_masked, cmf_masked, \
 width_im_masked, Rs_r0_masked, Rs_w_masked, Rs_w_to_iso_odi_masked, ori_to_odi_angle_masked], _ = \
    odcfunctions.mask_maps([odi_im, azi_im, ele_im, ori_im_ang, ori_im_len, cmf, width_im, Rs_r0, Rs_w, Rs_w_to_iso_odi, ori_to_odi_angle], \
                  [odi_mask,azi_mask,ele_mask,dir_mask,dir_mask])
    
# Map with orientation preference scaled by resultant length
cmap = matplotlib.cm.get_cmap("hsv")
ori_im_ang_masked_rgba = cmap(ori_im_ang_masked/180)
ori_im_ang_masked_rgba[:,:,3] = ori_im_len_masked

ori_mask = np.ones_like(ori_im_ang_masked)
ori_mask[np.isnan(ori_im_ang_masked)] = np.NaN

gray_im_rgba = np.ones_like(ori_im_ang_masked_rgba)
gray_im_rgba[:,:,:3] = 0.5
for i in range(4):
    gray_im_rgba[:,:,i] = gray_im_rgba[:,:,i] * ori_mask
    
    
    
#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Find the clusters  
    
# Detect ipsi clusters
settings.cluster.type = "ipsi"
ipsi_clusters, ipsi_cluster_props, ipsi_cluster_contours = odcfunctions.find_clusters(params, settings, msettings, v1_mask=overall_msk )
print("Detected {} ipsi clusters".format(len(ipsi_clusters)))

# Detect contra clusters
settings.cluster.type = "contra"
contra_clusters, contra_cluster_props, contra_cluster_contours = odcfunctions.find_clusters(params, settings, msettings, v1_mask=overall_msk )
print("Detected {} contra clusters".format(len(contra_clusters)))


#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Get the contours

# Get contours
odi_contours = odcfunctions.get_iso_contours(odi_im_masked, settings.display.iso_odi_contour_range)
odi_contours_highres = odcfunctions.get_iso_contours(odi_im, np.arange(-1,1,0.05))
azi_contours = odcfunctions.get_iso_contours(azi_im_masked, settings.display.iso_azi_contour_range)
ele_contours = odcfunctions.get_iso_contours(ele_im_masked, settings.display.iso_ele_contour_range)
ori_contours, iso_ori_values = odcfunctions.circ_get_iso_contours(ori_im_ang_masked*2, settings.display.iso_ori_contour_range)


#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Get the information on clusters and V1 region

ipsi_cluster_data = odcfunctions.quantify_od_patterns(ipsi_cluster_props, params, fparams, odi_im, odi_steepness_im, azi_im, ele_im, cmf, cmf_azi, cmf_ele, 
                                        azi_angle, ele_angle, azi_ele_ang, azi_ele_ratio, width_im, Rs_r0, Mmax_Mmin, Rs_w_to_iso_odi,
                                        ori_im_ang, ori_im_len, ori_to_odi_angle, msettings, cluster_type=1)
for cnr,cdat in enumerate(ipsi_cluster_data):
    meancv = np.nanmean(cdat.cv_all)
    print("Ipsi cluster {}) length={:3.0f}, width={:3.0f}, wid_alo={:3.0f}, area={:5.0f}, angle={:3.0f}, ang_alo={:3.0f}, ecc={:0.2f}, odi={:5.2f}, cmf={:0.5f}, rfsize={:4.1f}, stpnss={:0.5f}, odi_skl={:4.2f}, odi_brd={:4.2f}, cv={:4.2f}, Rs_r0={:4.2f}, MM={:4.2f}, Rs_odi={:4.2f}".format(
        cnr, cdat.length, cdat.width, np.nanmean(cdat.wid_alo), cdat.area, cdat.angle, np.nanmean(cdat.ang_alo), cdat.ecc, 
                cdat.odi, cdat.cmf, cdat.rfsize, cdat.odi_steepness, cdat.odi_skeleton, cdat.odi_border, meancv,
                cdat.Rs_r0, cdat.Mmax_Mmin, cdat.Rs_w_to_iso_odi))

contra_cluster_data = odcfunctions.quantify_od_patterns(contra_cluster_props, params, fparams, odi_im, odi_steepness_im, azi_im, ele_im, cmf, cmf_azi, cmf_ele, 
                                        azi_angle, ele_angle, azi_ele_ang, azi_ele_ratio, width_im, Rs_r0, Mmax_Mmin, Rs_w_to_iso_odi,
                                        ori_im_ang, ori_im_len, ori_to_odi_angle, msettings, cluster_type=2)
for cnr,cdat in enumerate(contra_cluster_data):
    meancv = np.nanmean(cdat.cv_all)
    print("Contra cluster {}) length={:3.0f}, width={:3.0f}, wid_alo={:3.0f}, area={:5.0f}, angle={:3.0f}, ang_alo={:3.0f}, ecc={:0.2f}, odi={:5.2f}, cmf={:0.5f}, rfsize={:4.1f}, stpnss={:0.5f}, odi_skl={:4.2f}, odi_brd={:4.2f}, cv={:4.2f}, Rs_r0={:4.2f}, MM={:4.2f}, Rs_odi={:4.2f}".format(
        cnr, cdat.length, cdat.width, np.nanmean(cdat.wid_alo), cdat.area, cdat.angle, np.nanmean(cdat.ang_alo), cdat.ecc, 
                cdat.odi, cdat.cmf, cdat.rfsize, cdat.odi_steepness, cdat.odi_skeleton, cdat.odi_border, meancv,
                cdat.Rs_r0, cdat.Mmax_Mmin, cdat.Rs_w_to_iso_odi))

v1_props = odcfunctions.get_largest_region(overall_msk)
cluster_data_v1 = odcfunctions.quantify_od_patterns(v1_props, params, fparams, odi_im, odi_steepness_im, azi_im, ele_im, cmf, cmf_azi, cmf_ele, 
                                        azi_angle, ele_angle, azi_ele_ang, azi_ele_ratio, width_im, Rs_r0, Mmax_Mmin, Rs_w_to_iso_odi,
                                        ori_im_ang, ori_im_len, ori_to_odi_angle, msettings, cluster_type=3)
for cnr,cdat in enumerate(cluster_data_v1):
    meancv = np.nanmean(cdat.cv_all)
    print("V1 cluster {}) length={:3.0f}, width={:3.0f}, wid_alo={:3.0f}, area={:5.0f}, angle={:3.0f}, ang_alo={:3.0f}, ecc={:0.2f}, odi={:5.2f}, cmf={:0.5f}, rfsize={:4.1f}, stpnss={:0.5f}, odi_skl={:4.2f}, odi_brd={:4.2f}, cv={:4.2f}, Rs_r0={:4.2f}, MM={:4.2f}, Rs_odi={:4.2f}".format(
        cnr, cdat.length, cdat.width, np.nanmean(cdat.wid_alo), cdat.area, cdat.angle, np.nanmean(cdat.ang_alo), cdat.ecc, 
                cdat.odi, cdat.cmf, cdat.rfsize, cdat.odi_steepness, cdat.odi_skeleton, cdat.odi_border, meancv,
                cdat.Rs_r0, cdat.Mmax_Mmin, cdat.Rs_w_to_iso_odi))

intercluster_msk = odcfunctions.remove_regions_from_mask(overall_msk, ipsi_cluster_props)
intercluster_msk = odcfunctions.remove_regions_from_mask(overall_msk, contra_cluster_props)
intercluster_props = odcfunctions.get_largest_region(intercluster_msk)
inter_cluster_data = odcfunctions.quantify_od_patterns(intercluster_props, params, fparams, odi_im, odi_steepness_im, azi_im, ele_im, cmf, cmf_azi, cmf_ele, azi_angle, 
                                              ele_angle, azi_ele_ang, azi_ele_ratio, width_im, Rs_r0, Mmax_Mmin, Rs_w_to_iso_odi,
                                              ori_im_ang, ori_im_len, ori_to_odi_angle, msettings, cluster_type=3)
for cnr,cdat in enumerate(inter_cluster_data):
    meancv = np.nanmean(cdat.cv_all)
    print("Intercluster {}) length={:3.0f}, width={:3.0f}, wid_alo={:3.0f}, area={:5.0f}, angle={:3.0f}, ang_alo={:3.0f}, ecc={:0.2f}, odi={:5.2f}, cmf={:0.5f}, rfsize={:4.1f}, stpnss={:0.5f}, odi_skl={:4.2f}, odi_brd={:4.2f}, cv={:4.2f}, Rs_r0={:4.2f}, MM={:4.2f}, Rs_odi={:4.2f}".format(
        cnr, cdat.length, cdat.width, np.nanmean(cdat.wid_alo), cdat.area, cdat.angle, np.nanmean(cdat.ang_alo), cdat.ecc, 
                cdat.odi, cdat.cmf, cdat.rfsize, cdat.odi_steepness, cdat.odi_skeleton, cdat.odi_border, meancv,
                cdat.Rs_r0, cdat.Mmax_Mmin, cdat.Rs_w_to_iso_odi))


#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Make figure panel 3a

# Show overlaid ODI, retinotopy, ORI maps with annotations
fig = plottingtools.init_fig(fig_size=(17.6,17.6/4))

ax = plt.subplot(141)
im_handle = plt.imshow(odi_im_masked, cmap="seismic_r", vmin=-0.5, vmax=0.5, interpolation="None")
odcfunctions.plot_iso_contours(v1_contour, "V1", settings, msettings, linewidth=1.0)
odcfunctions.plot_clusters(clusters=None, cluster_contours=ipsi_cluster_contours, marker="^", markersize=3, contour_linestyle="-", contour_linewidth=0.5)
odcfunctions.plot_clusters(clusters=None, cluster_contours=contra_cluster_contours, marker="s", markersize=3, markercolor="#ffffff",
                  contour_linestyle="-", contour_linewidth=0.5, contour_color="#ffffff" )
for cnr,cdat in enumerate(contra_cluster_data):
    odcfunctions.plot_contours_no_jumps(cdat.skeleton, cdat.bbox[:2], max_jump=5, linestyle="-", color="#FFFFFF",  markersize=0, linewidth=0.25)
for cnr,cdat in enumerate(ipsi_cluster_data):
    odcfunctions.plot_contours_no_jumps(cdat.skeleton, cdat.bbox[:2], max_jump=5, linestyle="-", color="#000000",  markersize=0, linewidth=0.25)
plt.axis('off');

ax = plt.subplot(142)
im_handle = plt.imshow(azi_im_masked, cmap=settings.cmap.azi, vmin=min(msettings.azimuth_values), vmax=max(msettings.azimuth_values), interpolation="None")
odcfunctions.plot_iso_contours(azi_contours, "azimuth", settings, msettings, linewidth=0.25, iso_contour_linestyle=["-"]*200)
odcfunctions.plot_iso_contours(v1_contour, "V1", settings, msettings, linewidth=1.0)
odcfunctions.plot_clusters(clusters=None, cluster_contours=ipsi_cluster_contours, marker="^", markersize=3, contour_linestyle="-", contour_linewidth=0.5)
odcfunctions.plot_clusters(clusters=None, cluster_contours=contra_cluster_contours, marker="s", markersize=3, markercolor="#ffffff",
                  contour_linestyle="-", contour_linewidth=0.5, contour_color="#ffffff" )
plt.axis('off')

ax = plt.subplot(143)
im_handle = plt.imshow(ele_im_masked, cmap=settings.cmap.ele, vmin=min(msettings.elevation_values), vmax=max(msettings.elevation_values), interpolation="None")
odcfunctions.plot_iso_contours(ele_contours, "elevation", settings, msettings, linewidth=0.25, iso_contour_linestyle=["-"]*200)
odcfunctions.plot_iso_contours(v1_contour, "V1", settings, msettings, linewidth=1.0)
odcfunctions.plot_clusters(clusters=None, cluster_contours=ipsi_cluster_contours, marker="^", markersize=3, contour_linestyle="-", contour_linewidth=0.5)
odcfunctions.plot_clusters(clusters=None, cluster_contours=contra_cluster_contours, marker="s", markersize=3, markercolor="#ffffff",
                  contour_linestyle="-", contour_linewidth=0.5, contour_color="#ffffff" )
plt.axis('off')

ax = plt.subplot(144)
im_handle = plt.imshow(gray_im_rgba, interpolation="None")
im_handle = plt.imshow(ori_im_ang_masked_rgba, interpolation="None")
odcfunctions.plot_iso_contours(ori_contours, "orientation", settings, msettings, color="#000000", iso_contour_linestyle=["-"]*20, linewidth=0.25)
odcfunctions.plot_iso_contours(v1_contour, "V1", settings, msettings, linewidth=1.0)
odcfunctions.plot_clusters(clusters=None, cluster_contours=ipsi_cluster_contours, marker="^", markersize=3, contour_linestyle="-", contour_linewidth=0.5)
odcfunctions.plot_clusters(clusters=None, cluster_contours=contra_cluster_contours, marker="s", markersize=3, markercolor="#ffffff",
                  contour_linestyle="-", contour_linewidth=0.5, contour_color="#ffffff" )
plt.axis('off')

savename = os.path.join( msettings.savepath, "Fig-3a-feature-maps" )
plottingtools.finish_figure( filename=savename, wspace=0.2, hspace=0.2 )

# Colorbars for figure 3a
fig = plottingtools.init_fig(fig_size=(9,5))
ax = plt.subplot(131)
im_handle = plt.imshow(gray_bg_bar_rgba, interpolation="None")
im_handle = plt.imshow(ori_bar_ang_rgba, interpolation="None")
plt.yticks([0,45,90,135,180])
plt.xticks([0,50])
plt.gca().invert_yaxis()
ax = plt.subplot(132)
im_handle = plt.imshow(azi_bar_rgba, interpolation="None")
plt.yticks(np.arange(0,255,254.99/4),msettings.azimuth_values)
plt.xticks([0,50])
plt.gca().invert_yaxis()
ax = plt.subplot(133)
im_handle = plt.imshow(ele_bar_rgba, interpolation="None")
plt.yticks(np.arange(0,255,254.99/2),msettings.elevation_values)
plt.xticks([0,50])
plt.gca().invert_yaxis()
savename = os.path.join( msettings.savepath, "Fig-3a-feature-maps-colorbars" )
plottingtools.finish_figure( filename=savename, wspace=0.1, hspace=0.1 )


#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Make figure panel 3b

# Show CMF, RF-size, Rs and iso-ori to OD angle maps
fig = plottingtools.init_fig(fig_size=(17.6,17.6/4))

ax = plt.subplot(141)
print("CMF, min={}, max={}".format(np.nanmin(cmf_masked),np.nanmax(cmf_masked)))
im_handle = plt.imshow(cmf_masked, cmap="terrain", vmin=0, vmax=0.002, interpolation="None")
odcfunctions.plot_iso_contours(v1_contour, "V1", settings, msettings, linewidth=1.0)
odcfunctions.plot_clusters(clusters=None, cluster_contours=ipsi_cluster_contours, marker="^", markersize=3, contour_linestyle="-", contour_linewidth=0.5)
odcfunctions.plot_clusters(clusters=None, cluster_contours=contra_cluster_contours, marker="s", markersize=3, markercolor="#ffffff",
                  contour_linestyle="-", contour_linewidth=0.5, contour_color="#ffffff" )
plt.axis('off');

ax = plt.subplot(142)
print("Rs_r0, min={}, max={}".format(np.nanmin(Rs_r0_masked),np.nanmax(Rs_r0_masked)))
print("Rs_w, min={}, max={}".format(np.nanmin(Rs_w_masked),np.nanmax(Rs_w_masked)))
im_handle = plt.imshow(np.ones_like(width_im_masked), cmap="Greys_r", vmin=0, vmax=1, interpolation="None")
odcfunctions.plot_iso_contours(odi_contours_highres, "odi", settings, msettings, color="#888888", iso_contour_linestyle=['-',]*50)
for x in range(10,msettings.max_x,40):
    for y in range(10,msettings.max_y,40):
        c = 10 * Rs_r0_masked[y,x]*np.exp(complex(0,np.radians(Rs_w_masked[y,x])))
        x1 = x+c.real
        y1 = y+c.imag
        x2 = x-c.real
        y2 = y-c.imag
        plt.plot( [x1,x2], [y1,y2], linestyle="-", marker="None", color="#000000", linewidth=0.5)
odcfunctions.plot_iso_contours(v1_contour, "V1", settings, msettings, linewidth=1.0)
plt.axis('off');

ax = plt.subplot(143)
print("RF width, min={}, max={}".format(np.nanmin(width_im_masked),np.nanmax(width_im_masked)))
im_handle = plt.imshow(width_im_masked, cmap="inferno_r", vmin=10, vmax=25, interpolation="None")
odcfunctions.plot_iso_contours(v1_contour, "V1", settings, msettings, linewidth=1.0)
odcfunctions.plot_clusters(clusters=None, cluster_contours=ipsi_cluster_contours, marker="^", markersize=3, contour_linestyle="-", contour_linewidth=0.5)
odcfunctions.plot_clusters(clusters=None, cluster_contours=contra_cluster_contours, marker="s", markersize=3, markercolor="#ffffff",
                  contour_linestyle="-", contour_linewidth=0.5, contour_color="#ffffff" )
plt.axis('off');

ax = plt.subplot(144)
print("RF width, min={}, max={}".format(np.nanmin(ori_to_odi_angle_masked),np.nanmax(ori_to_odi_angle_masked)))
im_handle = plt.imshow(ori_to_odi_angle_masked, cmap="YlGnBu_r", vmin=0, vmax=90, interpolation="None")
odcfunctions.plot_iso_contours(v1_contour, "V1", settings, msettings, linewidth=1.0)
odcfunctions.plot_clusters(clusters=None, cluster_contours=ipsi_cluster_contours, marker="^", markersize=3, contour_linestyle="-", contour_linewidth=0.5)
odcfunctions.plot_clusters(clusters=None, cluster_contours=contra_cluster_contours, marker="s", markersize=3, markercolor="#ffffff",
                  contour_linestyle="-", contour_linewidth=0.5, contour_color="#ffffff" )
plt.axis('off');

savename = os.path.join( msettings.savepath, "Fig-3b-calculated-maps" )
plottingtools.finish_figure( filename=savename, wspace=0.2, hspace=0.2 )

# Continuous azimuth and elevation colorbars
terrain = matplotlib.cm.get_cmap('terrain', 256)
inferno_r = matplotlib.cm.get_cmap('inferno_r', 256)
YlGnBu_r = matplotlib.cm.get_cmap('YlGnBu_r', 256)

cmf_bar = np.zeros((256,50))
rf_bar = np.zeros((256,50))
ooa_bar = np.zeros((256,50))
for i in range(256):
    cmf_bar[i,:] = i
    rf_bar[i,:] = i
    ooa_bar[i,:] = i
cmf_bar_rgba = terrain(cmf_bar/256)
rf_bar_rgba = inferno_r(rf_bar/256)
ooa_bar_rgba = YlGnBu_r(ooa_bar/256)

fig = plottingtools.init_fig(fig_size=(9,5))
ax = plt.subplot(131)
im_handle = plt.imshow(cmf_bar_rgba, interpolation="None")
plt.yticks(np.arange(0,255,254.99/2),np.array([0,0.001,0.002]))
plt.xticks([0,50])
plt.gca().invert_yaxis()
ax = plt.subplot(132)
im_handle = plt.imshow(rf_bar_rgba, interpolation="None")
plt.yticks(np.arange(0,255,254.99/3),np.array([10,15,20,25]))
plt.xticks([0,50])
plt.gca().invert_yaxis()
ax = plt.subplot(133)
im_handle = plt.imshow(ooa_bar_rgba, interpolation="None")
plt.yticks(np.arange(0,255,254.99/2),np.array([0,45,90]))
plt.xticks([0,50])
plt.gca().invert_yaxis()
savename = os.path.join( msettings.savepath, "Fig-3b-calculated-maps-colorbars" )
plottingtools.finish_figure( filename=savename, wspace=0.1, hspace=0.1 )


#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Make figure panel 12a

# Get a normal, low and highpass filtered ODI map
odi_multiplier = 1.0
odi_im_low,_,_ = odcfunctions.feature_map(params.od.ODI*odi_multiplier, params.od.XY, settings, msettings,
                                 smooth_sigma=settings.map.smooth_sigma*2.5)
odi_im_high = odi_im-odi_im_low
odi_im_high[~v1_mask] = np.NaN
odi_im_v1_masked = np.array(odi_im)
odi_im_v1_masked[~v1_mask] = np.NaN

# Get n-th percentile contours in high-pass map as putative clusters
odi_threshold = np.nanpercentile(odi_im_high,settings.cluster.percentile_threshold)
cluster_countours = skimage.measure.find_contours(image=odi_im_high, level=odi_threshold)

# Get the individual patches with this threshold
binary_odi_im = odi_im_high<odi_threshold
labeled_odi_im = skimage.measure.label(binary_odi_im, connectivity=2)
n_clusters_all = np.max(labeled_odi_im)

# Show overlaid ODI, retinotopy, ORI maps with annotations
fig = plottingtools.init_fig(fig_size=(17.6,17.6/4))

ax = plt.subplot(141)
im_handle = plt.imshow(odi_im_masked, cmap="seismic_r", vmin=-0.5, vmax=0.5, interpolation="None")
odcfunctions.plot_iso_contours(v1_contour, "V1", settings, msettings, linewidth=1.0)
plt.title("Ocular dominance", size=5)
plt.axis('off');

ax = plt.subplot(142)
im_handle = plt.imshow(odi_im_low, cmap="seismic_r", vmin=-0.5, vmax=0.5, interpolation="None")
odcfunctions.plot_iso_contours(v1_contour, "V1", settings, msettings, linewidth=1.0)
plt.title("Low-pass filtered", size=5)
plt.axis('off');

ax = plt.subplot(143)
im_handle = plt.imshow(odi_im_high, cmap="seismic_r", vmin=-0.5, vmax=0.5, interpolation="None")
odcfunctions.plot_iso_contours(v1_contour, "V1", settings, msettings, linewidth=1.0)
plt.title("High-pass filtered, V1b only", size=5)
plt.axis('off');

ax = plt.subplot(144)
im_handle = plt.imshow(labeled_odi_im, cmap="tab10", interpolation="None")
plt.title("Thresholded areas and detected clusters ", size=5)

odcfunctions.plot_iso_contours(v1_contour, "V1", settings, msettings, linewidth=1.0)
odcfunctions.plot_clusters(clusters=None, cluster_contours=ipsi_cluster_contours, marker="^", markersize=3, contour_linestyle="-", contour_linewidth=0.5)
odcfunctions.plot_clusters(clusters=None, cluster_contours=contra_cluster_contours, marker="s", markersize=3, markercolor="#ffffff",
                  contour_linestyle="-", contour_linewidth=0.5, contour_color="#ffffff" )
for cnr,cdat in enumerate(contra_cluster_data):
    odcfunctions.plot_contours_no_jumps(cdat.skeleton, cdat.bbox[:2], max_jump=5, linestyle="-", color="#FFFFFF",  markersize=0, linewidth=0.25)
for cnr,cdat in enumerate(ipsi_cluster_data):
    odcfunctions.plot_contours_no_jumps(cdat.skeleton, cdat.bbox[:2], max_jump=5, linestyle="-", color="#000000",  markersize=0, linewidth=0.25)
plt.axis('off');

savename = os.path.join( msettings.savepath, "Fig-S12a-cluster-detection-example" )
plottingtools.finish_figure( filename=savename, wspace=0.2, hspace=0.2 )


#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Make figure panel 12b

# Get a normal, low and highpass filtered ODI map
odi_multiplier = -1.0
odi_im_low,_,_ = odcfunctions.feature_map(params.od.ODI*odi_multiplier, params.od.XY, settings, msettings,
                                 smooth_sigma=settings.map.smooth_sigma*2.5)
odi_im_high = (odi_im*odi_multiplier)-odi_im_low
odi_im_high[~v1_mask] = np.NaN
odi_im_v1_masked = np.array(odi_im)
odi_im_v1_masked[~v1_mask] = np.NaN

# Get n-th percentile contours in high-pass map as putative clusters
odi_threshold = np.nanpercentile(odi_im_high,settings.cluster.percentile_threshold)
cluster_countours = skimage.measure.find_contours(image=odi_im_high, level=odi_threshold)

# Get the individual patches with this threshold
binary_odi_im = odi_im_high<odi_threshold
labeled_odi_im = skimage.measure.label(binary_odi_im, connectivity=2)
n_clusters_all = np.max(labeled_odi_im)

# Show overlaid ODI, retinotopy, ORI maps with annotations
fig = plottingtools.init_fig(fig_size=(17.6,17.6/4))

ax = plt.subplot(141)
im_handle = plt.imshow(odi_im_masked*-1, cmap="seismic_r", vmin=-0.5, vmax=0.5, interpolation="None")
odcfunctions.plot_iso_contours(v1_contour, "V1", settings, msettings, linewidth=1.0)
plt.title("Inverted ocular dominance", size=5)
plt.axis('off');

ax = plt.subplot(142)
im_handle = plt.imshow(odi_im_low, cmap="seismic_r", vmin=-0.5, vmax=0.5, interpolation="None")
odcfunctions.plot_iso_contours(v1_contour, "V1", settings, msettings, linewidth=1.0)
plt.title("Low-pass filtered", size=5)
plt.axis('off');

ax = plt.subplot(143)
im_handle = plt.imshow(odi_im_high, cmap="seismic_r", vmin=-0.5, vmax=0.5, interpolation="None")
odcfunctions.plot_iso_contours(v1_contour, "V1", settings, msettings, linewidth=1.0)
plt.title("High-pass filtered, V1b only", size=5)
plt.axis('off');

ax = plt.subplot(144)
im_handle = plt.imshow(labeled_odi_im, cmap="tab10", interpolation="None")
plt.title("Thresholded areas and detected clusters ", size=5)

odcfunctions.plot_iso_contours(v1_contour, "V1", settings, msettings, linewidth=1.0)
odcfunctions.plot_clusters(clusters=None, cluster_contours=ipsi_cluster_contours, marker="^", markersize=3, contour_linestyle="-", contour_linewidth=0.5)
odcfunctions.plot_clusters(clusters=None, cluster_contours=contra_cluster_contours, marker="s", markersize=3, markercolor="#ffffff",
                  contour_linestyle="-", contour_linewidth=0.5, contour_color="#ffffff" )
for cnr,cdat in enumerate(contra_cluster_data):
    odcfunctions.plot_contours_no_jumps(cdat.skeleton, cdat.bbox[:2], max_jump=5, linestyle="-", color="#FFFFFF",  markersize=0, linewidth=0.25)
for cnr,cdat in enumerate(ipsi_cluster_data):
    odcfunctions.plot_contours_no_jumps(cdat.skeleton, cdat.bbox[:2], max_jump=5, linestyle="-", color="#000000",  markersize=0, linewidth=0.25)
plt.axis('off');

savename = os.path.join( msettings.savepath, "Fig-S12b-region-detection-contra-example" )
plottingtools.finish_figure( filename=savename, wspace=0.2, hspace=0.2 )


#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# That's all folks !!
print("\nDone.\n")
