#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Functions for analyzing OD clusters and columns in mice

Created on Monday 20 Nov 2023

@author: pgoltstein
"""

# ################################################
# Imports
# ################################################

import sys, os, glob
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.ndimage
import skimage.measure
import skimage.morphology
import sklearn.metrics
import densityclustering
from collections import namedtuple
from box import Box



# ################################################
# Variables
# ################################################

ODpatterns = namedtuple("ODpatterns", "mouse, no, type, lab_no, mask, y, x, peak, bbox, skeleton, outline, outl_perim, perim, length, len_alo, width, wid_alo, angle, ang_alo, area, ecc, odi, odi_cells, absodi_cells, odi_ipsi_cells, odi_contra_cells, cmf, cmf_azi, cmf_ele, azi, azi_cells, ele, ele_cells, azi_ang, ele_ang, ret_angle, ret_ratio, rfsize, rfsize_cells, odi_steepness, odi_skeleton, odi_border, fpd_all, fpd_ipsi, fpd_contra, cv_all, cv_ipsi, cv_contra, dsi_all, dsi_ipsi, dsi_contra, Rs_r0, Mmax_Mmin, Rs_w_to_iso_odi, Rs_w_to_iso_odi_hist, ori_len, ori_ang_hist, ori_len_hist, ori_to_od_grad_angle, ori_to_od_grad_angle_hist, n_ipsi_cells, n_contra_cells")


# ################################################
# Functions
# ################################################

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~ Function that returns the general settings ~~~~

def generalsettings(layer):
    settings = Box()

    # Data loading settings
    settings.data = Box()
    if layer == "L4":
        settings.data.start_depth=370
        settings.data.depth_increment=20
        settings.data.skip_first_plane=False
        settings.data.include_z_start = 370
        settings.data.include_z_end = 431
    elif layer == "L2345":
        settings.data.start_depth=170
        settings.data.depth_increment=10
        settings.data.skip_first_plane=True
        settings.data.z_L4_min = 370
        settings.data.z_L4_max = 431
        settings.data.include_z_start = 170
        settings.data.include_z_end = 531
    settings.data.include_very_first_plane=True
    settings.data.include_fovpos = False
    settings.data.include_sign = 0.05
    settings.data.exclude_double_xy=3 # micron
    settings.data.exclude_double_z=settings.data.depth_increment+5 # micron
    settings.data.shuffle = "None"

    # Cluster identification settings
    settings.cluster = Box()
    settings.cluster.method = "density"
    settings.cluster.type = "ipsi"
    settings.cluster.fraction = 0.05
    settings.cluster.rho_min = 0.2
    settings.cluster.delta_min = 0.2
    settings.cluster.percentile_threshold = 15
    settings.cluster.minimum_cluster_length = 100
    settings.cluster.minimum_percentile_odi = 25
    settings.cluster.rho_x_delta_min = None

    # V1 detection
    settings.v1 = Box()
    settings.v1.sm_sigma_v1_mask = 100
    settings.v1.v1_angle = 30
    settings.v1.exclude_azimuth_threshold = 40
    settings.v1.exclude_border_pixels = 42
    settings.v1.min_elevation = -99
    settings.v1.max_elevation = 99

    # Map settings
    settings.map = Box()
    settings.map.smooth_sigma = 42
    settings.map.blank_when_less_than_n_cells = 10

    # Retinotopy settings
    settings.retinotopy = Box()
    settings.retinotopy.kernel_radius = 1
    settings.retinotopy.excl_gradient_below_deg_px = 0.01
    settings.retinotopy.excl_gradient_above_deg_px = 0.25
    settings.retinotopy.excl_width_below_deg = 5
    settings.retinotopy.excl_width_above_deg = 50

    # Display settings
    settings.display = Box()
    settings.display.iso_odi_contour_range = [0, 0.2]
    settings.display.iso_odi_contour_linestyle = ["-", "--"]
    settings.display.iso_azi_contour_range = np.arange(-48,72,1)
    settings.display.iso_azi_contour_linestyle = [":"]*200
    settings.display.iso_ele_contour_range = np.arange(-24,24,1)
    settings.display.iso_ele_contour_linestyle = [":"]*200
    settings.display.iso_ori_contour_range = np.arange(0,180,22.5)
    settings.display.iso_ori_contour_linestyle = ["-"]*200
    settings.display.v1_contour_linestyle = ["-"]

    # Custom colormaps
    settings.cmap = Box()
    hsv = matplotlib.cm.get_cmap('hsv', 256)
    newcolors = hsv(np.linspace(0, 1, np.ceil(256 / (4/5)).astype(int)))
    settings.cmap.azi = matplotlib.colors.ListedColormap(newcolors[:256])
    hsv_r = matplotlib.cm.get_cmap('hsv_r', 256)
    newcolors = hsv_r(np.linspace(0, 1, np.ceil(256 / (2/3)).astype(int)))
    settings.cmap.ele = matplotlib.colors.ListedColormap(newcolors[:256])

    settings.cmap.terrain = matplotlib.cm.get_cmap('terrain', 256)
    settings.cmap.terrain.set_bad(color="black")

    settings.cmap.plasma = matplotlib.cm.get_cmap('plasma', 256)
    settings.cmap.plasma.set_bad(color="green")

    return settings


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~ Function that returns the mouse specific settings ~~~~

def mousesettings(mousename,
                  datapath_od="../../data/part1-planedata-od-layer4",
                  datapath_ret="../../data/part1-planedata-retinotopy-layer4",
                  savepath = "../../figureout"):

    msettings = Box()
    msettings.name = mousename
    msettings.datapath_od = datapath_od
    msettings.datapath_ret = datapath_ret
    msettings.savepath = savepath

    if msettings.datapath_od[-1] == "/":
        msettings.datapath_od = msettings.datapath_od[:-1]
    if msettings.datapath_ret[-1] == "/":
        msettings.datapath_ret = msettings.datapath_ret[:-1]

    if int(mousename[1:]) < 20:
        msettings.convert_to_micron_x = 1192/1024
        msettings.convert_to_micron_y = 1019/1024
        msettings.invert_odi_values = False
        msettings.azimuth_values = np.array([-24, 0, 24, 48, 72])
        msettings.elevation_values = np.array([24, 0, -24])
        msettings.v1_left = True
    else:
        msettings.convert_to_micron_x = 1180/1024
        msettings.convert_to_micron_y = 982/1024
        msettings.invert_odi_values = True
        msettings.azimuth_values = np.array([72,48,24,0,-24])-24
        msettings.elevation_values = np.array([24, 0, -24])
        msettings.v1_left = False

        msettings.datapath_od = msettings.datapath_od + "-scn1a"
        msettings.datapath_ret = msettings.datapath_ret + "-scn1a"

    msettings.max_x = np.ceil(1024 * msettings.convert_to_micron_x).astype(int)
    msettings.max_y = np.ceil(1024 * msettings.convert_to_micron_y).astype(int)

    return msettings


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~ Function that loads imaging volume data ~~~~

def load_volume( settings, msettings, exp_type ):

    # Find all plane files
    if exp_type == "od":
        plane_files = sorted(glob.glob(os.path.join(msettings.datapath_od, msettings.name+"*")))
        savepath = os.path.join(msettings.datapath_od, "last-loaded-volume-{}.npy".format(msettings.name))
    elif exp_type == "ret":
        plane_files = sorted(glob.glob(os.path.join(msettings.datapath_ret, msettings.name+"*")))
        savepath = os.path.join(msettings.datapath_ret, "last-loaded-volume-{}.npy".format(msettings.name))
    count_planes = 0
    plane_files_new = []
    if settings.data.skip_first_plane:
        for name in plane_files:
            if settings.data.include_very_first_plane and count_planes == 0:
                plane_files_new.append(name)
            elif "-plane0-" not in name:
                plane_files_new.append(name)
            count_planes += 1
        plane_files = plane_files_new

    # Prepare data containers
    all_planes = []
    all_tms = []
    all_ims = []

    # Loop & load all planes
    depth = settings.data.start_depth
    for nr,name in enumerate(plane_files):

        # Update depth
        if not settings.data.include_fovpos:
            if nr > 0:
                depth += settings.data.depth_increment
        else:
            # Use alternative depth method
            plane_no = int(name[name.find("plane")+5])
            depth = settings.data.start_depth + (plane_no*settings.data.depth_increment)

        if depth >= settings.data.include_z_start and depth < settings.data.include_z_end:
            print("{:>3d}: {}".format(depth,name))
        else:
            continue

        # Load data matrix
        plane_dict = np.load(name,allow_pickle=True).item()
        plane_data = plane_dict["tuning_parameters"]
        parameter_names = plane_dict["parameter_names"]
        aspect_ratio = plane_dict["aspect_ratio"]
        tm = plane_dict["tuningmatrix"]
        im = plane_dict["I"]

        # Add the field-of-view position to the neuron coordinates
        if settings.data.include_fovpos:
            fovpos_x = plane_dict["fovpos_x"]*-1
            fovpos_y = plane_dict["fovpos_y"]
            print("Shifting spatial coordinates with FOV position: {},{} (Y,X)".format(fovpos_y,fovpos_x))
            plane_data[:, parameter_names.index("x")] = plane_data[:, parameter_names.index("x")] + fovpos_x
            plane_data[:, parameter_names.index("y")] = plane_data[:, parameter_names.index("y")] + fovpos_y

        # Add column with depth
        parameter_names.append("z")
        plane_data = np.concatenate([ plane_data, np.zeros((plane_data.shape[0],1))+depth ], axis=1)

        # Gather plane data and increase depth increment
        all_planes.append(plane_data)
        all_tms.append(tm)
        all_ims.append(im)

    # Create volumes and tuning matrix concatenating all depths
    volume = np.concatenate(all_planes,axis=0)
    tuningmatrix = np.concatenate(all_tms,axis=0)
    print("Volume size, all neurons: {}".format(volume.shape))

    # Exclude non-significant neurons if requested
    if settings.data.include_sign is not False:
        sign = volume[:, parameter_names.index("Significance")]
        volume = volume[sign<settings.data.include_sign, :]
        tuningmatrix = tuningmatrix[sign<settings.data.include_sign, :, :, :]
        print("Volume size, only significant neurons: {}".format(volume.shape))

    # Convert xy coordinates from pixels to micron
    if msettings.convert_to_micron_x is not False and msettings.convert_to_micron_y is not False:
        print("Converting pixels to micron (x: {}, y: {})".format(msettings.convert_to_micron_x,msettings.convert_to_micron_y))
        volume[:, parameter_names.index("x")] = volume[:, parameter_names.index("x")] * msettings.convert_to_micron_x
        volume[:, parameter_names.index("y")] = volume[:, parameter_names.index("y")] * msettings.convert_to_micron_y

    # Exlude neurons that appear in multiple planes
    if settings.data.exclude_double_xy is not False and settings.data.exclude_double_z is not False:
        print("Looking for overlapping neurons")

        # Loop all neurons, from last to first
        ncnt = 0
        for n in range(volume.shape[0]-1,-1,-1):

            # Get parameters of this neuron
            x = volume[:, parameter_names.index("x")]
            y = volume[:, parameter_names.index("y")]
            z = volume[:, parameter_names.index("z")]
            sign = volume[:, parameter_names.index("Significance")]

            # Find overlapping neurons
            nearby_xy = np.argwhere( np.logical_and( np.logical_and( np.abs(x-x[n])<settings.data.exclude_double_xy, np.abs(y-y[n])<settings.data.exclude_double_xy ), np.abs(z-z[n])<settings.data.exclude_double_z) ).ravel()
            if len(nearby_xy) > 1:

                # Find most significant neuron of the list
                most_sign = np.argmin( sign[nearby_xy] )

                # Remove others
                nearby_xy_less_sign = np.delete( nearby_xy, most_sign )
                volume = np.delete( volume, nearby_xy_less_sign, axis=0 )
                tuningmatrix = np.delete( tuningmatrix, nearby_xy_less_sign, axis=0 )
                ncnt += len(nearby_xy_less_sign)
        print("Removed {} overlapping neurons".format(ncnt))
        print("Volume size, excluding overlapping neurons: {}".format(volume.shape))

    # Invert ODI, in case recording was from right hemisphere
    if msettings.invert_odi_values and exp_type == "od":
        print("Inverting ODI values, recording was from right hemisphere")
        volume[:,parameter_names.index("ODI")] = volume[:,parameter_names.index("ODI")]*-1.0

    # Set the lowest neuron coordinate to 0,0
    if settings.data.include_fovpos:
        min_x = np.min(volume[:,parameter_names.index("x")])
        min_y = np.min(volume[:,parameter_names.index("y")])
        print("Adjusting spatial coordinates to base 0,0 by subtracting the minimum along each axis (Y={}, X={})".format(min_y,min_x))
        volume[:,parameter_names.index("x")] = volume[:,parameter_names.index("x")] - min_x
        volume[:,parameter_names.index("y")] = volume[:,parameter_names.index("y")] - min_y

    # Return data
    return volume,parameter_names,aspect_ratio,all_ims,tuningmatrix


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~ Function that gets statistics of the individual patched ~~~~

def quantify_od_patterns(cluster_props, params, fparams, odi_im, odi_steepness_im, azi_im, ele_im, cmf, cmf_azi, cmf_ele,
                         azi_angle, ele_angle, azi_ele_ang, azi_ele_ratio, rf_width_im, Rs_r0_im, Mmax_Mmin_im, Rs_w_to_iso_odi_im,
                         ori_ang_im, ori_len_im, ori_to_od_grad_angle_im, msettings, cluster_type):

    # Loop patches
    cluster_data = []
    n_clusters = len(cluster_props)
    for c in range(n_clusters):
        # print("Cluster {}".format(c))

        # Cluster label number and mask
        cluster_label = cluster_props[c].label
        cluster_mask = cluster_props[c].image

        # Get the centroid and bounding box
        cluster_centroid = cluster_props[c].centroid # (y,x)
        cluster_bounding_box = cluster_props[c].bbox # (y1,x1,y2,x2)

        # Get the skeleton and medial distance to skeleton
        skeleton_im = skimage.morphology.skeletonize(cluster_mask, method="lee")
        labeled_skel_im = skimage.measure.label(skeleton_im)
        skeleton_props = skimage.measure.regionprops(labeled_skel_im)
        skeleton_coords = sort_coords(skeleton_props[0].coords, verbose=False)
        cluster_skeleton = skeleton_coords # (y,x)

        # Get the outline and perimeters of the cluster
        cluster_outline = calculate_outline_of_object( cluster_mask, cluster_bounding_box, msettings.max_y, msettings.max_x ) # (y,x)
        cluster_outline = sort_coords(cluster_outline, verbose=False)
        cluster_outline_perimeter = cluster_outline.shape[0]
        cluster_perimeter = cluster_props[c].perimeter

        # Calculate length parameters
        cluster_length = cluster_props[c].axis_major_length
        cluster_length_alonso = np.sum(skeleton_im>0)

        # Calculate width parameters
        cluster_width = cluster_props[c].axis_minor_length
        _, distance_medial_axis = skimage.morphology.medial_axis(cluster_mask, return_distance=True)
        cluster_widths_alonso = distance_medial_axis[skeleton_im>0]

        # Calculate angle parameter
        cluster_angles_alonso = []
        step = 10
        max_step_distance = np.sqrt(np.power(10,2)+np.power(10,2))
        for nr in range( 0, skeleton_coords.shape[0]-step, step ):
            dy = skeleton_coords[nr+step,0]-skeleton_coords[nr,0]
            dx = skeleton_coords[nr+step,1]-skeleton_coords[nr,1]
            # This is to only look at non-jumping parts of the skeleton
            if np.sqrt(np.power(dy,2)+np.power(dx,2)) <= max_step_distance:
                cluster_angles_alonso.append( np.rad2deg(np.arctan2(dy,dx)) )
        cluster_angles_alonso = np.abs(np.mod(np.array(cluster_angles_alonso),180)-180)

        # Calculate the angle of the major axis of the cluster
        cluster_angle = np.mod(np.mod((cluster_props[c].orientation/(2*np.pi))*360, 360)+90,180)

        # Calculate area
        cluster_area = cluster_props[c].area

        # Calculate the eccentricity of the cluster outline
        cluster_eccentricity = cluster_props[c].eccentricity

        # Bounding box coordinates
        y1,x1,y2,x2 = cluster_bounding_box

        # Get the full image with cluster masked
        full_cluster_mask = np.zeros((odi_im.shape[0],odi_im.shape[1]))
        full_cluster_mask[y1:y2,x1:x2] = cluster_mask*1.0
        full_cluster_mask = full_cluster_mask > 0.5

        # Calculate the average ODI and location of ODI peak
        cluster_odi = get_mean_value_from_im(odi_im[y1:y2,x1:x2], cluster_mask)
        odi_max_coords = get_peak_coords_from_im(-1*odi_im[y1:y2,x1:x2], cluster_mask)
        cluster_odi_cells = np.nanmean(get_param_within_mask(params.od.ODI, params.od.XY, full_cluster_mask))
        cluster_absodi_cells = np.nanmean(np.abs(get_param_within_mask(params.od.ODI, params.od.XY, full_cluster_mask)))
        cluster_odi_ipsicells = np.nanmean(get_param_within_mask(params.ipsi.ODI, params.ipsi.XY, full_cluster_mask))
        cluster_odi_contracells = np.nanmean(get_param_within_mask(params.contra.ODI, params.contra.XY, full_cluster_mask))
        cluster_n_ipsicells = len(get_param_within_mask(params.ipsi.ODI, params.ipsi.XY, full_cluster_mask).ravel())
        cluster_n_contracells = len(get_param_within_mask(params.contra.ODI, params.contra.XY, full_cluster_mask).ravel())

        # Calculate the steepness of the ODI gradient at the cluster outline
        odi_steepness = np.nanmean(odi_steepness_im[ cluster_outline[:,0]+y1, cluster_outline[:,1]+x1 ])
        odi_border = np.nanmean(odi_im[ cluster_outline[:,0]+y1, cluster_outline[:,1]+x1 ])
        odi_skeleton = np.nanmean(odi_im[ cluster_skeleton[:,0]+y1, cluster_skeleton[:,1]+x1 ])

        # Calculate the average cmf
        cluster_cmf = get_mean_value_from_im(cmf[y1:y2,x1:x2], cluster_mask)
        cluster_cmf_azi = get_mean_value_from_im(cmf_azi[y1:y2,x1:x2], cluster_mask)
        cluster_cmf_ele = get_mean_value_from_im(cmf_ele[y1:y2,x1:x2], cluster_mask)

        # Calculate the average azimuth and elevation
        cluster_azi = get_mean_value_from_im(azi_im[y1:y2,x1:x2], cluster_mask)
        cluster_ele = get_mean_value_from_im(ele_im[y1:y2,x1:x2], cluster_mask)
        cluster_azi_cells = np.nanmean(get_param_within_mask(params.ret.AZI, params.ret.XY, full_cluster_mask))
        cluster_ele_cells = np.nanmean(get_param_within_mask(params.ret.ELE, params.ret.XY, full_cluster_mask))

        # Calculate the average angle of the retinotopic gradient for azimuth and elevation
        cluster_azi_ang = get_circular_mean_value_from_im(azi_angle[y1:y2,x1:x2]*2, cluster_mask)/2
        cluster_ele_ang = get_circular_mean_value_from_im(ele_angle[y1:y2,x1:x2]*2, cluster_mask)/2

        # Calculate the average retinotopy angle and angle ratio
        cluster_azi_ele_ang = get_mean_value_from_im(azi_ele_ang[y1:y2,x1:x2], cluster_mask)
        cluster_azi_ele_ratio = get_mean_value_from_im(azi_ele_ratio[y1:y2,x1:x2], cluster_mask)

        # Calculate the receptive field width
        cluster_rf_width = get_mean_value_from_im(rf_width_im[y1:y2,x1:x2], cluster_mask)
        cluster_rf_width_cells = get_param_within_mask(fparams.ret.WDTH, fparams.ret.XY, full_cluster_mask)

        # Get the preferred directions
        cluster_fpd_all = get_param_within_mask(params.od.fPD, params.od.XY, full_cluster_mask)
        cluster_fpd_ipsi = get_param_within_mask(params.ipsi.fPD, params.ipsi.XY, full_cluster_mask)
        cluster_fpd_contra = get_param_within_mask(params.contra.fPD, params.contra.XY, full_cluster_mask)

        # Get the circular variance
        cluster_cv_all = get_param_within_mask(params.od.CV, params.od.XY, full_cluster_mask)
        cluster_cv_ipsi = get_param_within_mask(params.ipsi.CV, params.ipsi.XY, full_cluster_mask)
        cluster_cv_contra = get_param_within_mask(params.contra.CV, params.contra.XY, full_cluster_mask)

        # Get the direction selectivity index
        cluster_dsi_all = get_param_within_mask(params.od.DSI, params.od.XY, full_cluster_mask)
        cluster_dsi_ipsi = get_param_within_mask(params.ipsi.DSI, params.ipsi.XY, full_cluster_mask)
        cluster_dsi_contra = get_param_within_mask(params.contra.DSI, params.contra.XY, full_cluster_mask)

        # Calculate the average Rs_r0
        cluster_Rs_r0 = get_mean_value_from_im(Rs_r0_im[y1:y2,x1:x2], cluster_mask)

        # Calculate the average Mmax_Mmin
        cluster_Mmax_Mmin = get_mean_value_from_im(Mmax_Mmin_im[y1:y2,x1:x2], cluster_mask)

        # Calculate the average Rs_w to iso-ODI angle
        cluster_Rs_w_to_iso_odi_im = get_mean_value_from_im(Rs_w_to_iso_odi_im[y1:y2,x1:x2], cluster_mask)

        # Calculate the histogram of Rs_w to iso-ODI angle
        d_values = get_values_from_im(Rs_w_to_iso_odi_im[y1:y2,x1:x2], cluster_mask)
        cluster_Rs_w_to_iso_odi_im_hist,edges = np.histogram( d_values, bins=30, range=[0,90] )

        # Calculate the average orientation map resultant length
        cluster_ori_len_im = get_mean_value_from_im(ori_len_im[y1:y2,x1:x2], cluster_mask)

        # Calculate the histogram of the average orientation map resultant angle
        d_values = get_values_from_im(ori_ang_im[y1:y2,x1:x2], cluster_mask)
        cluster_ori_ang_im_hist,edges = np.histogram( d_values, bins=60, range=[0,180] )

        # Calculate the histogram of the average orientation map resultant length
        d_values = get_values_from_im(ori_len_im[y1:y2,x1:x2], cluster_mask)
        cluster_ori_len_im_hist,edges = np.histogram( d_values, bins=50, range=[0,1] )

        # Calculate the average orientation map gradient to ODI map gradient angle
        cluster_ori_to_od_grad_angle_im = get_mean_value_from_im(ori_to_od_grad_angle_im[y1:y2,x1:x2], cluster_mask)

        # Calculate the histogram of the average orientation map gradient to ODI map gradient angle
        d_values = get_values_from_im(ori_to_od_grad_angle_im[y1:y2,x1:x2], cluster_mask)
        cluster_ori_to_od_grad_angle_im_hist,edges = np.histogram( d_values, bins=30, range=[0,90] )

        # Store in named tuple and add to list
        cluster_data.append(ODpatterns(mouse=msettings.name, no=c, type=cluster_type, lab_no=cluster_label,
                                       mask=cluster_mask, y=cluster_centroid[0], x=cluster_centroid[1],
                                       peak=odi_max_coords, bbox=cluster_bounding_box, skeleton=cluster_skeleton,
                                       outline=cluster_outline, outl_perim=cluster_outline_perimeter, perim=cluster_perimeter,
                                       length=cluster_length, len_alo=cluster_length_alonso,
                                       width=cluster_width, wid_alo=cluster_widths_alonso,
                                       angle=cluster_angle, ang_alo=cluster_angles_alonso, area=cluster_area,
                                       ecc=cluster_eccentricity, odi=cluster_odi, odi_cells=cluster_odi_cells, absodi_cells=cluster_absodi_cells,
                                       odi_ipsi_cells=cluster_odi_ipsicells, odi_contra_cells=cluster_odi_contracells, cmf=cluster_cmf,
                                       cmf_azi=cluster_cmf_azi, cmf_ele=cluster_cmf_ele,
                                       azi=cluster_azi, azi_cells=cluster_azi_cells, ele_cells=cluster_ele, ele=cluster_ele_cells,
                                       azi_ang=cluster_azi_ang, ele_ang=cluster_ele_ang,
                                       ret_angle=cluster_azi_ele_ang, ret_ratio=cluster_azi_ele_ratio,
                                       rfsize=cluster_rf_width, rfsize_cells=cluster_rf_width_cells,
                                       odi_steepness=odi_steepness, odi_border=odi_border, odi_skeleton=odi_skeleton,
                                       fpd_all=cluster_fpd_all, fpd_ipsi=cluster_fpd_ipsi, fpd_contra=cluster_fpd_contra,
                                       cv_all=cluster_cv_all, cv_ipsi=cluster_cv_ipsi, cv_contra=cluster_cv_contra,
                                       dsi_all=cluster_dsi_all, dsi_ipsi=cluster_dsi_ipsi, dsi_contra=cluster_dsi_contra,
                                       Rs_r0=cluster_Rs_r0, Mmax_Mmin=cluster_Mmax_Mmin, Rs_w_to_iso_odi=cluster_Rs_w_to_iso_odi_im,
                                       Rs_w_to_iso_odi_hist=cluster_Rs_w_to_iso_odi_im_hist,
                                       ori_len=cluster_ori_len_im, ori_ang_hist=cluster_ori_ang_im_hist, ori_len_hist=cluster_ori_len_im_hist,
                                       ori_to_od_grad_angle=cluster_ori_to_od_grad_angle_im,
                                       ori_to_od_grad_angle_hist=cluster_ori_to_od_grad_angle_im_hist,
                                       n_ipsi_cells=cluster_n_ipsicells, n_contra_cells=cluster_n_contracells))

    return cluster_data


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~ Function that extracts the parameters ~~~~

def get_params(volume_od, parameter_names_od, msettings, volume_ret=None, parameter_names_ret=None):

    # Prep data
    params = Box()
    params.od = Box()
    params.ipsi = Box()
    params.contra = Box()

    # Get OD data
    params.od.XY = volume_od[:, [parameter_names_od.index("x"),parameter_names_od.index("y")]]
    params.od.Z = volume_od[:,parameter_names_od.index("z")]
    params.od.ODI = volume_od[:,parameter_names_od.index("ODI")]

    # Get Ipsi data
    params.ipsi.XY = params.od.XY[params.od.ODI<=0,:]
    params.ipsi.ODI = params.od.ODI[params.od.ODI<=0]

    # Get Contra data
    params.contra.XY = params.od.XY[params.od.ODI>0,:]
    params.contra.ODI = params.od.ODI[params.od.ODI>0]

    # Get retinotopy data (optional)
    if volume_ret is not None and parameter_names_ret is not None:
        params.ret = Box()
        params.ret.XY = volume_ret[:, [parameter_names_ret.index("x"),parameter_names_ret.index("y")]]
        AZI = volume_ret[:,parameter_names_ret.index("Pref azim")]
        params.ret.AZI = msettings.azimuth_values[AZI.astype(int)]
        ELE = volume_ret[:,parameter_names_ret.index("Pref elev")]
        params.ret.ELE = msettings.elevation_values[ELE.astype(int)]

    return params


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~ Function that adds orientation tuning parameters ~~~~

def add_ori_params(params, volume_od, parameter_names_od, tuningmatrix_od, msettings, volume_ret=None, parameter_names_ret=None):

    # Get N's and make data containers
    n_neurons,n_eyes,n_dirs,n_trials = tuningmatrix_od.shape
    PD = np.zeros((n_neurons,))
    PDamp = np.zeros((n_neurons,))
    fPD = np.zeros((n_neurons,))
    fPDamp = np.zeros((n_neurons,))
    fSIG = np.zeros((n_neurons,))
    CV = np.zeros((n_neurons,))
    OSI = np.zeros((n_neurons,))
    DSI = np.zeros((n_neurons,))
    error_count = 0

    # Loop neurons and calculate orientation tuning properties
    print("Calculating orientation tuning properties ", end="", flush=True)
    for n in range(n_neurons):

        # Get dominant-eye orientation tuning curve
        ipsi_tc = np.mean( tuningmatrix_od[n,0,:,:], axis=1 )
        contra_tc = np.mean( tuningmatrix_od[n,1,:,:], axis=1 )
        odi = calculate_odi(ipsi_tc, contra_tc, method=0)
        if odi < 0:
            dom_tc = ipsi_tc
        else:
            dom_tc = contra_tc

        # Fit the tuning curve using a two-peak Gaussian curve, params = (Rbaseline, Rpref, Rnull, thetapref, sigma)
        try:
            fittedcurve, fittedparams = twopeakgaussianfit( tuningcurve=dom_tc )
            fPDamp[n] = fittedparams[1]-fittedparams[0]
            fPD[n] = fittedparams[3]
            fSIG[n] = fittedparams[4]
        except RuntimeError as error:
            error_count += 1
            fPDamp[n] = np.NaN
            fPD[n] = np.NaN
            fSIG[n] = np.NaN

        # Pref dir
        PD[n],PD_ix = calculate_preferreddirection(dom_tc)
        PDamp[n] = dom_tc[PD_ix]

        # Circular variance
        reslen,_ = calculate_resultant(dom_tc, resultant_type="Orientation")
        CV[n] = 1-reslen

        # OSI
        OSI[n] = calculate_orientation_selectivity_index( dom_tc )

        # DSI
        DSI[n] = calculate_direction_selectivity_index( dom_tc )

        if np.mod(n,100) == 0:
            print('.', end="", flush=True)
    print(" done ({} tuningcurves failed to fit)".format(error_count))

    # Add to OD data
    params.od.PD = PD
    params.od.PDamp = PDamp
    params.od.fPD = fPD
    params.od.fPDamp = fPDamp
    params.od.fSIG = fSIG
    params.od.CV = CV
    params.od.OSI = OSI
    params.od.DSI = DSI

    # Add to ipsi data
    params.ipsi.PD = PD[params.od.ODI<=0]
    params.ipsi.PDamp = PDamp[params.od.ODI<=0]
    params.ipsi.fPD = fPD[params.od.ODI<=0]
    params.ipsi.fPDamp = fPDamp[params.od.ODI<=0]
    params.ipsi.fSIG = fSIG[params.od.ODI<=0]
    params.ipsi.CV = CV[params.od.ODI<=0]
    params.ipsi.OSI = OSI[params.od.ODI<=0]
    params.ipsi.DSI = DSI[params.od.ODI<=0]

    # Add to contra data
    params.contra.PD = PD[params.od.ODI>0]
    params.contra.PDamp = PDamp[params.od.ODI>0]
    params.contra.fPD = fPD[params.od.ODI>0]
    params.contra.fPDamp = fPDamp[params.od.ODI>0]
    params.contra.fSIG = fSIG[params.od.ODI>0]
    params.contra.CV = CV[params.od.ODI>0]
    params.contra.OSI = OSI[params.od.ODI>0]
    params.contra.DSI = DSI[params.od.ODI>0]

    return params


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~ Function that extracts fitted parameters ~~~~

def get_fitted_ret_params(volume_ret, parameter_names_ret, tuningmatrix_ret, settings, msettings):

    # Calculate tuning curves for azimuth
    tcs_ret = np.mean(tuningmatrix_ret,axis=3)

    # Get upsamples xvalues (helping against over-fitting)
    xvalues = np.array(msettings.azimuth_values)
    xvals_interp = np.linspace(np.min(xvalues), np.max(xvalues), 4*len(msettings.azimuth_values)) # Upsample by factor of 4

    # Make data containers
    n_neurons = tcs_ret.shape[0]
    AZI = np.zeros((tcs_ret.shape[0],))
    WDTH = np.zeros((tcs_ret.shape[0],))

    # Get all preferred azimuth and azimuth-width of all cells
    print("Fitting azimuth curves ", end="", flush=True)
    for n in range(n_neurons):

        # Mean tuning curve across elevations, only leaving azimuth
        tuningcurve_azi = np.mean(tcs_ret[n,:,:],axis=0)

        # Interpolate tuning curve to prevent overfitting
        if msettings.v1_left:
            tuningcurve_azi_interp = np.interp(xvals_interp, xvalues, tuningcurve_azi)
        else:
            tuningcurve_azi_interp = np.interp(xvals_interp, xvalues[::-1], tuningcurve_azi)[::-1]

        # Fit the interpolated curve
        fitted_params, mse = gaussianfit( tuningcurve_azi_interp, xvals_interp )

        # Store preferred azimuth and azimuth-width value
        AZI[n] = fitted_params[2]
        WDTH[n] = fitted_params[3]
        if np.mod(n,100) == 0:
            print('.', end="", flush=True)
    print(" done")

    # Get matching XY coordinates
    XY = volume_ret[:, [parameter_names_ret.index("x"),parameter_names_ret.index("y")]]

    # Remove NaN's and fits with awkward widths
    remove_ix_nan = np.logical_or(np.isnan(AZI), np.isnan(WDTH))
    remove_ix_width = np.logical_or( WDTH<settings.retinotopy.excl_width_below_deg,
                                     WDTH>settings.retinotopy.excl_width_above_deg )
    remove_ix = np.logical_or( remove_ix_nan, remove_ix_width )
    XY = XY[~remove_ix,:]
    AZI = AZI[~remove_ix]
    WDTH = WDTH[~remove_ix]

    # Store parameters in box
    params = Box()
    params.ret = Box()
    params.ret.XY = XY
    params.ret.AZI = AZI
    params.ret.WDTH = WDTH

    return params


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~ Function that returns parameter values only for neurons within masked area ~~~~

def get_param_within_mask(param, XY, mask, return_xy=False):

    # Get rounded XY coordinates
    rounded_XYs = np.round(XY).astype(int)

    # Number of neurons and data container for mask values per neuron
    n_neurons = rounded_XYs.shape[0]
    mask_vals = np.zeros((n_neurons,))

    # Get mask value per neuron
    for n in range(n_neurons):
        mask_vals[n] = mask[rounded_XYs[n,1],rounded_XYs[n,0]]*1.0
    remove_ix = mask_vals < 0.5

    # Return masked param and XY
    if return_xy:
        return param[~remove_ix], XY[~remove_ix,:]
    else:
        return param[~remove_ix]


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~ Function that returns the preferred direction ~~~~

def calculate_preferreddirection( tuningcurve, angles=None ):
    """ Returns the angle and index of the preferred direction
        - Inputs -
        tuningcurve: 1D array of neuronal responses per stimulus
        angles:      Array with angles (equal sampling across 360 degrees)
        returns tuple (angle, angle_ix)
    """
    # Calculate angles if not supplied
    if angles is None:
        angles = np.arange(0,360,360/tuningcurve.shape[0])

    # Find index of largest value
    pref_ix = np.argmax(tuningcurve)

    # Return angle and index of largest value
    return angles[pref_ix],pref_ix


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~ Function that returns the orientation selectivity index (OSI) ~~~~

def calculate_orientation_selectivity_index( tuningcurve ):
    """ Returns a ratio-like orientation selectivity index (osi). Assumes equal sampling across angles.
        - Inputs -
        tuningcurve: 1D array of neuronal responses per stimulus
        returns osi (np.float)
    """

    # Average across opposite directions to get orientation curve
    half_range = int(tuningcurve.shape[0]/2)
    orientationcurve = tuningcurve[:half_range]+tuningcurve[half_range:]

    # Find index of largest value
    pref_ix = np.argmax(orientationcurve)

    # Find index of orthogonal
    orth_ix = int(np.mod( pref_ix + orientationcurve.shape[0]/2, orientationcurve.shape[0] ))

    # Calulate and return osi
    osi = (orientationcurve[pref_ix]-orientationcurve[orth_ix]) / (orientationcurve[pref_ix]+orientationcurve[orth_ix])

    # Return osi, but bound between 0 and 1
    return np.max([0.0, np.min([1.0, osi]) ])


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~ Function that returns the direction selectivity index (DSI) ~~~~

def calculate_direction_selectivity_index( tuningcurve ):
    """ Returns a ratio-like direction selectivity index (dsi). Assumes equal sampling across angles.
        - Inputs -
        tuningcurve: 1D array of neuronal responses per stimulus
        returns dsi (np.float)
    """

    # Find index of largest value
    pref_ix = np.argmax(tuningcurve)

    # Find index of the null (opposite) direction
    null_ix = int(np.mod( pref_ix + tuningcurve.shape[0]/2, tuningcurve.shape[0] ))

    # Calulate dsi
    dsi = (tuningcurve[pref_ix]-tuningcurve[null_ix]) / (tuningcurve[pref_ix]+tuningcurve[null_ix])

    # Return dsi, but bound between 0 and 1
    return np.max([0.0, np.min([1.0, dsi]) ])


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~ Function that returns the ocular dominance indes (ODI) ~~~~

def calculate_odi(ipsi_tc, contra_tc, method=0):
    """ Calculates the ODI using the (ipsi and contra) responses to the preferred direction of the dominant eye
        - Inputs -
        ipsi_tc: 1D array of neuronal responses per stimulus, for ipsi eye
        contra_tc: 1D array of neuronal responses per stimulus, for contra eye
        method:
                0 (odi from ipsi and contra response at preferred direction of each eye individually)
                1 (odi from ipsi and contra response at preferred direction of dominant eye)
                2 (odi from ipsi and contra response averaged across all directions for each eye)
        returns ODI (ocular dominance index)
    """
    if method == 0:
        _, ipsi_pref_ix = calculate_preferreddirection(ipsi_tc)
        _, contra_pref_ix = calculate_preferreddirection(contra_tc)
        ipsi = ipsi_tc[ipsi_pref_ix]
        contra = contra_tc[contra_pref_ix]
    elif method == 1:
        _, ipsi_pref_ix = calculate_preferreddirection(ipsi_tc)
        _, contra_pref_ix = calculate_preferreddirection(contra_tc)
        ipsi_peak, contra_peak = ipsi_tc[ipsi_pref_ix], contra_tc[contra_pref_ix]
        if ipsi_peak > contra_peak:
            pref_ix = ipsi_pref_ix
        else:
            pref_ix = contra_pref_ix
        ipsi = ipsi_tc[pref_ix]
        contra = contra_tc[pref_ix]
    elif method == 2:
        ipsi = np.mean(ipsi_tc)
        contra = np.mean(contra_tc)
    odi = np.max([np.min([(contra-ipsi) / (contra+ipsi), 1]), -1])
    return odi


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~ Function that returns the resultant length and angle ~~~~

def calculate_resultant( tuningcurve, resultant_type, angles=None ):
    """ Calculates the resultant length and angle (using complex direction space, normalized to a range from 0.0 to 1.0, angle in degrees)
        - Inputs -
        tuningcurve: 1D array of neuronal responses per stimulus
        resultant_type: "direction" (default) or "orientation"
        angles:      Array with angles (equal sampling across 360 degrees)
        returns normalized resultant length,angle (np.float,np.float)
    """

    # Calculate angles if not supplied
    if angles is None:
        angles = np.arange(0,360,360/tuningcurve.shape[0])

    # Assign orientation multiplier
    ori_mult = 2.0 if resultant_type.lower()=="orientation" else 1.0

    # Set values below 0.0 to 0.0
    tuningcurve[tuningcurve<0.0] = 0.0

    # Initialize a list for our vector representation
    vector_representation = []

    # Iterate over response amplitudes and directions (in radians)
    for r,ang in zip(tuningcurve,np.radians(angles)):
        vector_representation.append( r * np.exp(ori_mult*complex(0,ang)) )

    # Convert the list to a numpy array
    vector_representation = np.array(vector_representation)

    # Mean resultant vector
    mean_vector = np.sum(vector_representation) / np.sum(tuningcurve)

    # Length of resultant
    res_length = np.abs(mean_vector)

    # Angle of resultant
    res_angle = np.mod(np.degrees(np.angle(mean_vector)), 360) / ori_mult

    # Return the length (absolute) of the resultant vector
    return res_length,res_angle


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~ Function that returns a single peak Gaussian ~~~~

def onepeakgaussian(x, Rbaseline, Rpref, stimpref, sigma):
    return Rbaseline + Rpref*np.exp(-(x-stimpref)**2/(2*sigma**2))


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~ Function that returns a two-peak Gaussian ~~~~

def twopeakgaussian(x, Rbaseline, Rpref, Rnull, thetapref, sigma):
    return Rbaseline + Rpref*np.exp(-wrap_x(x-thetapref)**2/(2*sigma**2)) + Rnull*np.exp(-wrap_x(x+180-thetapref)**2/(2*sigma**2))


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~ Function that wraps xvalues to 0-180 degrees ~~~~

def wrap_x(x):
    return np.abs(np.abs(np.mod(x,360)-180)-180)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~ Function that fits a single peak Gaussian to a curve ~~~~

def gaussianfit( tuningcurve, xvalues ):

    # Estimate baseline level of tuning curve
    Rbaseline = np.min(tuningcurve)

    # Estimate preferred stimulus
    pref_ix = np.argmax(tuningcurve)
    stimpref = xvalues[pref_ix]

    # Estimate response amplitude to preferred direction
    Rpref = tuningcurve[pref_ix]

    # Estimate estimate of tuning curve width
    sigma = 0.5 * (xvalues[1]-xvalues[0])

    # Merge all parameters in a tuple
    param_estimate = (Rbaseline, Rpref, stimpref, sigma),

    # Fit parameters
    try:
        fitted_params,pcov = scipy.optimize.curve_fit( onepeakgaussian, xvalues, tuningcurve, p0=param_estimate)
        fit_curve = onepeakgaussian(xvalues, *fitted_params)
        mse = np.mean(np.power(tuningcurve-fit_curve,2))

        # Return fitted parameters
        return fitted_params, mse
    except:
        return np.array([np.NaN,np.NaN,np.NaN,np.NaN]), np.NaN



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~ Function that fits a two-peak Gaussian to a curve ~~~~

def twopeakgaussianfit( tuningcurve, angles=None ):
    """ Returns an array of size (360,) with the fitted tuning curve at 1 degree resolution
        - Inputs -
        tuningcurve: 1D array of neuronal responses per stimulus
        angles:      Array with angles (equal sampling across 360 degrees)
        returns fitted tuning curve (array of np.float)
    """

    # Calculate angles if not supplied
    if angles is None:
        angles = np.arange(0,360,360/tuningcurve.shape[0])

    # Get x-value range to consider
    xvalues = np.arange(0,360,1)

    # -- Estimate parameters --

    # Baseline level of tuning curve
    Rbaseline = np.min(tuningcurve)

    # Estimate preferred stimulus
    pref_ix = np.argmax(tuningcurve)
    thetapref = angles[pref_ix]

    # Response amplitude to preferred direction
    Rpref = tuningcurve[pref_ix]

    # Response amplitude to null direction
    Rnull = tuningcurve[np.mod(pref_ix+int(angles.shape[0]/2),angles.shape[0])]

    # Estimate of tuning curve width
    # sigma = halfwidthhalfmax(tuningcurve)
    sigma = 0.5 * (angles[1]-angles[0])

    # Merge all parameters in a tuple
    param_estimate = (Rbaseline, Rpref, Rnull, thetapref, sigma),

    # Fit parameters
    fitted_params,pcov = scipy.optimize.curve_fit( twopeakgaussian, angles, tuningcurve, p0=param_estimate)

    # Return fitted curve
    return twopeakgaussian(xvalues,*fitted_params), fitted_params


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~ Function that calculates the cluster centers ~~~~

def find_clusters(params, settings, msettings, v1_mask=None ):

    if settings.cluster.method.lower() == "density" and settings.cluster.type.lower() == "ipsi":
        clusters = densityclustering.find_clusters(params.ipsi.XY, fraction=settings.cluster.fraction,
                        rho_min=settings.cluster.rho_min, delta_min=settings.cluster.delta_min,
                        rho_x_delta_min=settings.cluster.rho_x_delta_min, show_rho_vs_delta=False)
        return clusters

    if settings.cluster.method.lower() == "density" and settings.cluster.type.lower() == "od":
        clusters = densityclustering.find_clusters(params.od.XY, fraction=settings.cluster.fraction, weights=params.od.ODI*-1.0,
                        rho_min=settings.cluster.rho_min, delta_min=settings.cluster.delta_min,
                        rho_x_delta_min=settings.cluster.rho_x_delta_min, show_rho_vs_delta=False)
        return clusters

    if settings.cluster.method.lower() == "od-geometry":

        # Invert ODI if looking for contra clusters
        if settings.cluster.type.lower() == "ipsi":
            odi_multiplier = 1.0
        elif settings.cluster.type.lower() == "contra":
            odi_multiplier = -1.0

        # Get a normal, low and highpass filtered ODI map
        odi_im,_,odi_mask = feature_map(params.od.ODI*odi_multiplier, params.od.XY, settings, msettings)
        odi_im_low,_,_ = feature_map(params.od.ODI*odi_multiplier, params.od.XY, settings, msettings,
                                         smooth_sigma=settings.map.smooth_sigma*2.5)
        odi_im_high = odi_im-odi_im_low
        odi_im_high[~v1_mask] = np.NaN
        odi_im_masked = np.array(odi_im)
        odi_im_masked[~v1_mask] = np.NaN

        # Get n-th percentile contours in high-pass map as putative clusters
        odi_threshold = np.nanpercentile(odi_im_high,settings.cluster.percentile_threshold)
        cluster_countours = skimage.measure.find_contours(image=odi_im_high, level=odi_threshold)

        # Get the individual patches with this threshold
        binary_odi_im = odi_im_high<odi_threshold
        labeled_odi_im = skimage.measure.label(binary_odi_im, connectivity=2)
        n_clusters_all = np.max(labeled_odi_im)

        # Get the cluster properties
        cluster_props_all = skimage.measure.regionprops(labeled_odi_im)

        # Only keep clusters that match criteria:
        # 1) Minimum length
        # 2) Center ODI below x-th percentile
        # 3) Center in V1
        # 4) Center in area with enough cells
        clusters = []
        cluster_props = []
        cluster_contours = []
        for c in range(n_clusters_all):

            # Check minimun size of the cluster
            len1,len2 = cluster_props_all[c].axis_major_length, cluster_props_all[c].axis_minor_length
            is_min_len_exceeded = np.nanmax([len1,len2]) > settings.cluster.minimum_cluster_length

            # Get center coordinates
            y,x = cluster_props_all[c].centroid

            # Check actual ODI is smaller than x-th percentile
            is_odi_smaller_zero = odi_im[int(y),int(x)] < np.nanpercentile(odi_im_masked, settings.cluster.minimum_percentile_odi)

            # Check if center is in V1
            is_center_in_v1 = v1_mask[int(y),int(x)]

            # Check if odi_im has enough cells contributing
            is_valid_odi_im = odi_mask[int(y),int(x)]
            # print("{}) is_min_len_exceeded={}, is_odi_smaller_zero={}, is_center_in_v1={}, is_valid_odi_im={}".format(
                # c,is_min_len_exceeded,is_odi_smaller_zero,is_center_in_v1,is_valid_odi_im))

            # If critteria matched, keep clusters
            if is_min_len_exceeded and is_odi_smaller_zero and is_center_in_v1 and is_valid_odi_im:

                # Cluster centers and cluster properties
                clusters.append({'X': x, 'Y': y, 'rho': 0, 'delta': 0})
                cluster_props.append(cluster_props_all[c])

                # Contour of this patch
                this_cluster_contour = skimage.measure.find_contours((labeled_odi_im==c+1)*1.0, level=0.5)
                cluster_contours.append( this_cluster_contour[0] )

        return clusters, cluster_props, cluster_contours

    if settings.cluster.method.lower() == "crossings":

        # Invert ODI if looking for contra clusters
        if settings.cluster.type.lower() == "ipsi":
            odi_multiplier = 1.0
        elif settings.cluster.type.lower() == "contra":
            odi_multiplier = -1.0

        # Get a normal, low and highpass filtered ODI map
        odi_im,_,odi_mask = feature_map(params.od.ODI*odi_multiplier, params.od.XY, settings, msettings)
        odi_im_low,_,_ = feature_map(params.od.ODI*odi_multiplier, params.od.XY, settings, msettings,
                                         smooth_sigma=settings.map.smooth_sigma*2.5)
        odi_im_high = odi_im-odi_im_low
        odi_im_high[~v1_mask] = np.NaN
        odi_im_masked = np.array(odi_im)
        odi_im_masked[~v1_mask] = np.NaN

        # Get 50-th percentile contours in high-pass map as putative ipsi-regions
        odi_threshold = np.nanpercentile(odi_im_high,50)
        cluster_countours = skimage.measure.find_contours(image=odi_im_high, level=odi_threshold)

        # Get the individual patches with this threshold
        binary_odi_im = odi_im_high<odi_threshold
        labeled_odi_im = skimage.measure.label(binary_odi_im, connectivity=2)
        n_clusters_all = np.max(labeled_odi_im)

        # Get the cluster properties
        cluster_props_all = skimage.measure.regionprops(labeled_odi_im)

        # Get the data of the entire region beloning to the crossing
        clusters = []
        cluster_props = []
        cluster_contours = []
        for c in range(n_clusters_all):

            # Get center coordinates
            y,x = cluster_props_all[c].centroid

            # Cluster centers and cluster properties
            clusters.append({'X': x, 'Y': y, 'rho': 0, 'delta': 0})
            cluster_props.append(cluster_props_all[c])

            # Contour of this patch
            this_cluster_contour = skimage.measure.find_contours((labeled_odi_im==c+1)*1.0, level=0.5)
            cluster_contours.append( this_cluster_contour[0] )

        return clusters, cluster_props, cluster_contours


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~ Function that sorts coordinates ~~~~

def sort_coords( unsorted_coords, some_large_value=1000000, verbose=False ):
    """ Sorts coordinates by proximity """

    # Number of coordinates
    n_coords = unsorted_coords.shape[0]

    # Get pairwise distance matrix, set the diagonal to some large value
    D = sklearn.metrics.pairwise_distances(unsorted_coords, metric="euclidean")
    np.fill_diagonal(D, some_large_value)

    # Output variable
    sorted_coords = np.zeros_like(unsorted_coords)

    # First entry is the first one in the unsorted list
    sorted_coords[0,:] = unsorted_coords[0,:]

    # Remove distances to the first entry from the distance matrix
    D[:,0] = some_large_value

    # Set start coordinate index
    start_coordinate_ix = 0
    for c in range(1,n_coords):

        # Find the coordinate index with the smallest distance
        next_coordinate_ix = np.argmin(D[start_coordinate_ix,:])

        # Store this as the next coordinate
        sorted_coords[c,:] = unsorted_coords[next_coordinate_ix,:]

        # Set the distances to this next coordinate to some large value
        D[:,next_coordinate_ix] = some_large_value

        # Display process
        if verbose:
            print("{}->{} {}".format(start_coordinate_ix,next_coordinate_ix,sorted_coords[c,:]))

        # Now the next coordinate becomes the start coordinate and we iterate through the list
        start_coordinate_ix = next_coordinate_ix

    # Return the sorted coordinate list
    return sorted_coords


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~ Function that gets all values from an image using a mask ~~~~

def get_values_from_im(im, mask):
    return im[mask]


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~ Function that gets the mean value from an image using a mask ~~~~

def get_mean_value_from_im(im, mask):
    im_copy = np.array(im)
    im_copy[~mask] = np.NaN
    return np.nanmean(im_copy)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~ Function that gets the circular mean value from an array of angles (range 0-360) ~~~~

def circ_mean(values):
    # Convert to radians, split in x and y component and take the mean
    values_y = np.nanmean(np.sin(np.radians(values)))
    values_x = np.nanmean(np.cos(np.radians(values)))

    # Reconstruct the angle in degrees from the components and return
    return np.mod(np.degrees(np.arctan2(values_y,values_x)),360)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~ Function that gets the circular mean value from an array of angles (range 0-360) ~~~~

def circ_mean_im(values,axis=0):

    # Convert to radians, split in x and y component and take the mean
    values_y = np.nanmean(np.sin(np.radians(values)),axis=axis)
    values_x = np.nanmean(np.cos(np.radians(values)),axis=axis)

    # Reconstruct the angle in degrees from the components and return
    return np.mod(np.degrees(np.arctan2(values_y,values_x)),360)



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~ Function that gets the circular mean value from an image using a mask ~~~~

def get_circular_mean_value_from_im(im_deg, mask):
    # Works on the range 0--360 degrees
    im_copy = np.array(im_deg)
    im_copy[~mask] = np.NaN

    # Convert to radians, split in x and y component and take the mean
    angle_im_y = np.nanmean(np.sin(np.radians(im_copy)))
    angle_im_x = np.nanmean(np.cos(np.radians(im_copy)))

    # Reconstruct the angle in degrees from the components and return
    return np.mod(np.degrees(np.arctan2(angle_im_y,angle_im_x)),360)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~ Function that gets the coordinates of the peak value from a masked area in an image ~~~~

def get_peak_coords_from_im(im, mask):
    im_copy = np.array(im)
    im_copy[~mask] = np.NaN
    max_coords = np.argwhere( im_copy == np.nanmax(im_copy))
    if max_coords.shape[0] > 1:
        print("Warning: found multiple peaks {}, taking the mean ({})".format(max_coords, np.mean(max_coords,axis=0)))
    max_coords = np.mean(max_coords,axis=0)
    return max_coords


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~ Function that returns the properties of the largest object ~~~~

def get_largest_region(mask):
    labeled_im = skimage.measure.label(mask, connectivity=2)
    props = skimage.measure.regionprops(labeled_im)
    if len(props) > 1:
        area_sizes = [p.area for p in props]
        largest_nr = np.argmax(np.array(area_sizes))
        props = [props[largest_nr],]
    return props


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~ Function that returns the properties of the largest object ~~~~

def keep_largest_mask_area(mask):
    labeled_im = skimage.measure.label(mask, connectivity=2)
    props = skimage.measure.regionprops(labeled_im)
    area_sizes = [p.area for p in props]
    largest_nr = np.argmax(np.array(area_sizes))
    largest_label = props[largest_nr].label
    return labeled_im==largest_label


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~ Function that returns the mask with objects (regions) removed ~~~~

def remove_regions_from_mask(mask, props):
    all_obj_mask = np.zeros_like(mask)
    for p in props:
        obj_mask = np.zeros_like(mask)
        y1,x1,y2,x2 = p.bbox
        obj_mask[y1:y2,x1:x2] = p.image
        all_obj_mask = np.logical_or(all_obj_mask, obj_mask)
    mask_no_obj = np.logical_and(mask, ~all_obj_mask)
    return mask_no_obj


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~ Function that gets the outline of a masked object ~~~~

def calculate_outline_of_object( masked_object, object_bounding_box, max_y, max_x ):

    # Get the outline of the object
    object_mask = np.zeros((masked_object.shape[0]+2,masked_object.shape[1]+2))
    object_mask[1:-1,1:-1] = masked_object
    object_outline_im = (object_mask*1.0) - (skimage.morphology.binary_erosion(object_mask)*1.0)
    object_outline_im = object_outline_im[1:-1,1:-1]

    # Remove outline at the very border of full image
    y1,x1,y2,x2 = object_bounding_box
    if y1 == 0:
        object_outline_im[0,:] = 0
    if x1 == 0:
        object_outline_im[:,0] = 0
    if y2 == max_y:
        object_outline_im[-1,:] = 0
    if x2 == max_x:
        object_outline_im[:,-1] = 0

    # Return the outline coordinates
    return np.argwhere(object_outline_im != 0)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~ Function that sets outlier values in a map to nan ~~~~

def set_outliers_to_NaN(data,min_val,max_val,name, quiet=False):
    ixs_high = data>max_val
    ixs_low = data<min_val
    data[ixs_high] = np.NaN
    data[ixs_low] = np.NaN
    if not quiet:
        print("{}: Clipped {} datapoints below {} and {} datapoints above {} to np.NaN".format( name, np.nansum(ixs_low), min_val, np.nansum(ixs_high), max_val ))
    return data


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~ Function that performans circular smoothing of a map ~~~~

def circ_im_smooth(angle_im_deg, map_smooth_sigma_grad):

    # Convert to radians and split in x and y component
    angle_im_y = np.sin(np.radians(angle_im_deg))
    angle_im_x = np.cos(np.radians(angle_im_deg))

    # Smooth the component maps
    angle_im_y_sm = scipy.ndimage.gaussian_filter(angle_im_y, sigma=map_smooth_sigma_grad)
    angle_im_x_sm = scipy.ndimage.gaussian_filter(angle_im_x, sigma=map_smooth_sigma_grad)

    # Reconstruct the angle in degrees from the smoothed component maps and return
    angle_im_sm_deg = np.mod(np.degrees(np.arctan2(angle_im_y_sm,angle_im_x_sm)),360)

    return angle_im_sm_deg


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~ Function that calculates a feature map ~~~~

def feature_map(feature_values, feature_XYs, settings, msettings, smooth_sigma=None, max_x=None, max_y=None):

    if max_x is None:
        max_x = msettings.max_x
    if max_y is None:
        max_y = msettings.max_y

    # Prepare a matrix that represents the feature image
    feature_im = np.zeros((max_y,max_x))
    coverage_im = np.zeros((max_y,max_x))

    # Loop cells and add them to the image
    rounded_XYs = np.round(feature_XYs).astype(int)
    n_neurons = feature_XYs.shape[0]
    for n in range(n_neurons):

        # Sum feature values and coverage counts if the value is not np.NaN
        if ~np.isnan(feature_values[n]):
            feature_im[rounded_XYs[n,1],rounded_XYs[n,0]] = feature_im[rounded_XYs[n,1],rounded_XYs[n,0]] + feature_values[n]
            coverage_im[rounded_XYs[n,1],rounded_XYs[n,0]] = coverage_im[rounded_XYs[n,1],rounded_XYs[n,0]] + 1.0

    # Smooth maps
    if smooth_sigma is None:
        smooth_sigma = settings.map.smooth_sigma
    feature_im = scipy.ndimage.gaussian_filter(feature_im, sigma=smooth_sigma)
    coverage_im = scipy.ndimage.gaussian_filter(coverage_im, sigma=smooth_sigma)

    # Create a mask excludinig regions with low number of cells
    mask_im = np.ones_like(feature_im)
    if settings.map.blank_when_less_than_n_cells > 0:

        # Find out what coverage by 1 neuron means in terms of the smoothed map
        test_threshold_im = np.zeros((max_y,max_x))
        test_threshold_im[500,500] = 1.0
        test_threshold_im = scipy.ndimage.gaussian_filter(test_threshold_im, sigma=smooth_sigma)
        one_cell_at_3micron = test_threshold_im[500,503]

        # Set areas with no coverage to NaN
        mask_im[coverage_im<(one_cell_at_3micron*settings.map.blank_when_less_than_n_cells)] = 0
    mask_im = mask_im > 0.5

    # Normalize feature map by cell count
    feature_im = feature_im / coverage_im

    return feature_im, coverage_im, mask_im


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~ Function that calculates a feature map based on circular features ~~~~

def circ_feature_map(feature_values, feature_XYs, settings, msettings, smooth_sigma=None, max_x=None, max_y=None):

    if max_x is None:
        max_x = msettings.max_x
    if max_y is None:
        max_y = msettings.max_y

    # Prepare a matrix that represents the feature image
    feature_im_x = np.zeros((max_y,max_x))
    feature_im_y = np.zeros((max_y,max_x))
    coverage_im = np.zeros((max_y,max_x))

    # Loop cells and add them to the image
    rounded_XYs = np.round(feature_XYs).astype(int)
    n_neurons = feature_XYs.shape[0]
    for n in range(n_neurons):

        # Sum feature values and coverage counts if the value is not np.NaN
        if ~np.isnan(feature_values[n]):
            feature_im_x[rounded_XYs[n,1],rounded_XYs[n,0]] = feature_im_x[rounded_XYs[n,1],rounded_XYs[n,0]] + np.cos(np.radians(feature_values[n]))
            feature_im_y[rounded_XYs[n,1],rounded_XYs[n,0]] = feature_im_y[rounded_XYs[n,1],rounded_XYs[n,0]] + np.sin(np.radians(feature_values[n]))
            coverage_im[rounded_XYs[n,1],rounded_XYs[n,0]] = coverage_im[rounded_XYs[n,1],rounded_XYs[n,0]] + 1.0

    # Smooth maps
    if smooth_sigma is None:
        smooth_sigma = settings.map.smooth_sigma
    feature_im_x = scipy.ndimage.gaussian_filter(feature_im_x, sigma=smooth_sigma)
    feature_im_y = scipy.ndimage.gaussian_filter(feature_im_y, sigma=smooth_sigma)
    coverage_im = scipy.ndimage.gaussian_filter(coverage_im, sigma=smooth_sigma)

    # Create a mask excludinig regions with low number of cells
    mask_im = np.ones_like(feature_im_x)
    if settings.map.blank_when_less_than_n_cells > 0:

        # Find out what coverage by 1 neuron means in terms of the smoothed map
        test_threshold_im = np.zeros((max_y,max_x))
        test_threshold_im[500,500] = 1.0
        test_threshold_im = scipy.ndimage.gaussian_filter(test_threshold_im, sigma=smooth_sigma)
        one_cell_at_3micron = test_threshold_im[500,503]

        # Set areas with no coverage to NaN
        mask_im[coverage_im<(one_cell_at_3micron*settings.map.blank_when_less_than_n_cells)] = 0
    mask_im = mask_im > 0.5

    # Normalize feature map by cell count
    feature_im_x = feature_im_x / coverage_im
    feature_im_y = feature_im_y / coverage_im

    # Reconstruct the angle in degrees from the smoothed component maps and return
    feature_im_ang = np.mod(np.degrees(np.arctan2(feature_im_y,feature_im_x)),360)
    feature_im_len = np.sqrt( np.power(feature_im_y,2) + np.power(feature_im_x,2) )

    return feature_im_ang, feature_im_len, coverage_im, mask_im


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~ Function that calculates the magnitude and angle of gradients in a feature map ~~~~

def calculate_gradient_magnitude_angle_unit_pix(im, settings, kernel_radius=None):
    if kernel_radius is None:
        kernel_radius = settings.retinotopy.kernel_radius

    # Kernels that calculate the amount of retinotopy covered in x and y direction
    kern_values = np.linspace(0,1,kernel_radius+1)
    kern_values = np.concatenate([ kern_values[::-1], -1*kern_values[1:] ])
    kern_ver = np.array([kern_values,]).T
    kern_hor = np.array([kern_values,])
    n_kern_vals = kern_values.shape[0]

    # Normalize kernels
    kern_ver = kern_ver / np.abs(np.sum(np.arange(n_kern_vals)*kern_values))
    kern_hor = kern_hor / np.abs(np.sum(np.arange(n_kern_vals)*kern_values))

    # Convolve the kernels, returning gradients in map-unit per pixel
    gradient_ver = scipy.ndimage.convolve(im, kern_ver)
    gradient_hor = scipy.ndimage.convolve(im, kern_hor)

    # Calculate the number of degrees covered by each pixel (using Pythagoras)
    gradient_magnitude = np.sqrt( np.power(gradient_ver,2) + np.power(gradient_hor,2) )

    # Calculate gradient angle (using trigonometry), in degrees
    gradient_angle_deg = np.mod(np.arctan2( gradient_ver, gradient_hor ) * (180/np.pi), 360)

    # Return
    return gradient_magnitude, gradient_angle_deg


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~ Function that calculates the magnitude and angle of gradients of circular values in a feature map ~~~~

def circ_calculate_gradient_magnitude_angle_unit_pix(im, settings, kernel_radius=None):
    # Note: im should contain values on a range of 0-360 degree
    if kernel_radius is None:
        kernel_radius = settings.retinotopy.kernel_radius

    # Kernels that calculate the amount of retinotopy covered in x and y direction
    kern_values = np.linspace(0,1,kernel_radius+1)
    kern_values = np.concatenate([ kern_values[::-1], -1*kern_values[1:] ])
    kern_ver = np.array([kern_values,]).T
    kern_hor = np.array([kern_values,])
    n_kern_vals = kern_values.shape[0]

    # Normalize kernels
    kern_ver = kern_ver / np.abs(np.sum(np.arange(n_kern_vals)*kern_values))
    kern_hor = kern_hor / np.abs(np.sum(np.arange(n_kern_vals)*kern_values))

    # Convolve the kernels, returning gradients in map-unit per pixel
    gradient_ver_x = scipy.ndimage.convolve( np.cos(np.radians(im)), kern_ver)
    gradient_ver_y = scipy.ndimage.convolve( np.sin(np.radians(im)), kern_ver)
    gradient_hor_x = scipy.ndimage.convolve( np.cos(np.radians(im)), kern_hor)
    gradient_hor_y = scipy.ndimage.convolve( np.sin(np.radians(im)), kern_hor)

    # Merge to single gradient magnitude per horiz and vert direction. Angle (pref ori) can be ignored, is not revelant here
    gradient_ver = np.sqrt( np.power(gradient_ver_y,2) + np.power(gradient_ver_x,2) )
    gradient_hor = np.sqrt( np.power(gradient_hor_y,2) + np.power(gradient_hor_x,2) )

    # However, this makes the gradient_ver and _hor only be positive, so that should be fixed
    gradient_ver = gradient_ver * ( ((gradient_ver_y<0)*-1.0) + ((gradient_ver_y>=0)*1.0) )
    gradient_hor = gradient_hor * ( ((gradient_hor_y<0)*-1.0) + ((gradient_hor_y>=0)*1.0) )
    # Note: this is an experimental and not fully verified fix !!

    # Calculate the number of degrees covered by each pixel (using Pythagoras)
    gradient_magnitude = np.sqrt( np.power(gradient_ver,2) + np.power(gradient_hor,2) )

    # Calculate gradient angle (using trigonometry), in degrees
    gradient_angle_deg = np.mod(np.arctan2( gradient_ver, gradient_hor ) * (180/np.pi), 360)

    # Return
    return gradient_magnitude, gradient_angle_deg


def calculate_cmf(azi_im, ele_im, settings):

    # Calculate coverage
    azi_degr_pix, azi_angle = calculate_gradient_magnitude_angle_unit_pix(azi_im, settings)
    ele_degr_pix, ele_angle = calculate_gradient_magnitude_angle_unit_pix(ele_im, settings)

    # Smooth coverage maps
    azi_degr_pix = scipy.ndimage.gaussian_filter(azi_degr_pix, sigma=settings.map.smooth_sigma)
    ele_degr_pix = scipy.ndimage.gaussian_filter(ele_degr_pix, sigma=settings.map.smooth_sigma)

    # Smooth angle maps
    azi_angle_reor = (azi_angle*-1) +360
    azi_angle_ori = np.mod(azi_angle_reor,180)
    azi_angle_ori_sm = circ_im_smooth(azi_angle_ori*2, settings.map.smooth_sigma)/2
    ele_angle_reor = (ele_angle*-1) +360
    ele_angle_ori = np.mod(ele_angle_reor,180)
    ele_angle_ori_sm = circ_im_smooth(ele_angle_ori*2, settings.map.smooth_sigma)/2

    # Exclude data out of range
    azi_degr_pix = set_outliers_to_NaN(azi_degr_pix, settings.retinotopy.excl_gradient_below_deg_px,
                                       settings.retinotopy.excl_gradient_above_deg_px, "azi_degr_pix")
    ele_degr_pix = set_outliers_to_NaN(ele_degr_pix, settings.retinotopy.excl_gradient_below_deg_px,
                                       settings.retinotopy.excl_gradient_above_deg_px, "ele_degr_pix")

    # Cortical magnification
    cmf_mm2_deg2 = 1/( (azi_degr_pix * ele_degr_pix) * (1000*1000) )
    cmf_azi_mm2_deg2 = 1/( (azi_degr_pix * azi_degr_pix) * (1000*1000) )
    cmf_ele_mm2_deg2 = 1/( (ele_degr_pix * ele_degr_pix) * (1000*1000) )

    # Angle and map distortion values
    aziele_ang = azi_angle_ori_sm-ele_angle_ori_sm
    ang_diff = np.mod(aziele_ang,180) # 0 and 180 are close, 90 is far
    ang_diff = np.abs(ang_diff-90) # now 0 is far, 90 is close
    ang_diff = np.abs(ang_diff-90) # now 0 is close, 90 is far
    aziele_ratio = azi_degr_pix/ele_degr_pix

    return cmf_mm2_deg2, cmf_azi_mm2_deg2, cmf_ele_mm2_deg2, azi_angle_ori_sm, ele_angle_ori_sm, ang_diff, aziele_ratio


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~ Function that calculates the sum of two vectors defined by angle and amplitude ~~~~

def sum_vecs(lens,angs):
    lens = np.stack(lens,axis=2)
    angs = np.stack(angs,axis=2)
    c_arr = np.zeros_like(lens).astype(complex)
    c_arr.imag = np.radians(angs)
    c_arr = lens*np.exp(c_arr)
    sumvec = np.sum(c_arr,axis=-1)
    x = np.real(sumvec)
    y = np.imag(sumvec)
    sum_len = np.sqrt( np.power(y,2) + np.power(x,2) )
    sum_ang = np.degrees(np.arctan2( y, x ))
    return sum_len, sum_ang


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~ Function that calculates the inner axes of a piece of retinotopic map, defined ~~~~
# ~~~~   by CMF vectors (angle, amplitude)                                            ~~~~

def calculate_inner_axes( a_len, a_ang, b_len, b_ang ):
    ax1_ang = np.mod(b_ang+90,360)
    ax2_ang = np.mod(a_ang+90,360)
    ang_a_to_ax1 = np.mod(ax1_ang-a_ang,360)
    ang_b_to_ax2 = np.mod(ax2_ang-b_ang,360)
    ax1_len = np.abs(a_len / np.cos(np.radians(ang_a_to_ax1)))
    ax2_len = np.abs(b_len / np.cos(np.radians(ang_b_to_ax2)))
    corner_vecs = [ sum_vecs([ax1_len,ax2_len], [ax1_ang, ax2_ang]),
                    sum_vecs([ax1_len,ax2_len], [np.mod(ax1_ang+180,360), ax2_ang]), 
                    sum_vecs([ax1_len,ax2_len], [ax1_ang, np.mod(ax2_ang+180,360)]), 
                    sum_vecs([ax1_len,ax2_len], [np.mod(ax1_ang+180,360), np.mod(ax2_ang+180,360)]) ]
    return [ax1_len,ax1_ang], [ax2_len,ax2_ang], corner_vecs


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~ Function that calculates Rs on full image data (matrix X x Y x corner) ~~~~

def calc_Rs_im(amplitudes_ims, angles_deg_ims):
    c_arr = np.zeros_like(angles_deg_ims).astype(complex)
    c_arr.imag = np.radians(angles_deg_ims)
    c_arr = amplitudes_ims*np.exp(c_arr)

    top_div = np.sum(np.power(c_arr,2),axis=2)
    bot_div = 4*np.power(np.mean(np.abs(c_arr),axis=2),2)
    Rs_xy = np.sqrt(top_div / bot_div)

    x = np.real(Rs_xy)
    y = np.imag(Rs_xy)
    amp_im = np.sqrt( np.power(y,2) + np.power(x,2) )
    ang_im = np.mod(np.arctan2( y, x ) * (180/np.pi), 180)
    return amp_im, ang_im


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~ Function that calculates Mmax_Mmin on full image data with R_0 ~~~~

def calc_Mmax_Mmin_im(R_0):
    return np.sqrt( (1-np.power(R_0,2)) / (1+np.power(R_0,2)) )


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~ Function that calculates Rs and Mmax_Mmin on retinotopic map data (CMF and gradientr angle) ~~~~

def calculate_Rs_and_Mmax_Mmin_per_pixel( cmf_azi, cmf_ele, azi_angle, ele_angle ):

    # Get 'virtual' vertices of each pixel
    [x_len,x_ang], [y_len,y_ang], crnrs = calculate_inner_axes( a_len=cmf_azi, a_ang=azi_angle, b_len=cmf_ele, b_ang=ele_angle )

    # Stack vertices into single matrix
    vec_amps = np.stack([crnrs[0][0],crnrs[1][0],crnrs[2][0],crnrs[3][0]],axis=2)
    vec_angs = np.stack([crnrs[0][1],crnrs[1][1],crnrs[2][1],crnrs[3][1]],axis=2)

    # This is to make it work, otherwise the vertices are upside down ¯\_(ツ)_/¯
    vec_angs = np.mod(vec_angs * -1.0,360)

    # Calculate Rs image
    Rs_r0_im, Rs_w_im = calc_Rs_im(vec_amps, vec_angs)

    # Calculate Mmax / Mmin image
    Mmax_Mmin_im = calc_Mmax_Mmin_im(Rs_r0_im)

    # Return values
    all_coords = [[x_len,x_ang], [y_len,y_ang], [vec_amps,vec_angs]]
    return Rs_r0_im, Rs_w_im, Mmax_Mmin_im, all_coords


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~ Function that calculates the angle between two maps (on 180 degree basis, returns max 90 degree angle) ~~~~

def angle_between_180deg_maps(ang_map1, ang_map2):
    intersect_angles = ang_map1-ang_map2
    intersect_angles = np.mod(intersect_angles,180) # 0 and 180 are close, 90 is far
    intersect_angles = np.abs(intersect_angles-90) # now 0 is far, 90 is close
    intersect_angles = np.abs(intersect_angles-90) # now 0 is close, 90 is far
    return intersect_angles


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~ Function that calculates the angle of the iso-odi lines from a odi-gradient-angle-map ~~~~

def get_iso_odi_from_odi_angle_map(od_grad_ang_deg):
    od_grad_ang_deg = np.mod(od_grad_ang_deg,180)
    iso_od_ori_deg = np.mod(od_grad_ang_deg-90,180)
    return iso_od_ori_deg, od_grad_ang_deg


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~ Function that calculates a mask for V1b ~~~~

def get_v1_mask(params, settings, msettings):

    # Calculate V1b mask
    azi_im_sm, _, _ = feature_map( params.ret.AZI, params.ret.XY, settings, msettings, smooth_sigma=settings.v1.sm_sigma_v1_mask)
    ele_im_sm, _, _ = feature_map( params.ret.ELE, params.ret.XY, settings, msettings, smooth_sigma=settings.v1.sm_sigma_v1_mask)
    _, ang_azi_sm = calculate_gradient_magnitude_angle_unit_pix(azi_im_sm, settings, kernel_radius=3)
    if msettings.v1_left:
        binarized_gradient = np.logical_or(ang_azi_sm<(90-settings.v1.v1_angle), ang_azi_sm>(270-settings.v1.v1_angle))
    else:
        binarized_gradient = np.logical_and(ang_azi_sm>(90-settings.v1.v1_angle), ang_azi_sm<(270-settings.v1.v1_angle))
    labeled_gradient = skimage.measure.label(binarized_gradient, connectivity=2)

    # The largest region with a gradient is V1 (should be)
    n_labels = np.max(labeled_gradient)+1
    area_size = []
    for label_nr in range(n_labels):
        mean_bin_gradient = np.mean(binarized_gradient[labeled_gradient==label_nr])

        # Only include labeled regions with right-ward gradients, so set size of left wards gradients to zero
        if mean_bin_gradient > 0.5:
            area_size.append(np.sum(labeled_gradient==label_nr))
        else:
            area_size.append(0)

    # V1 is the largest area with a rightwards gradient
    v1_label_nr = np.argmax(area_size)

    # Now make the V1 mask
    v1_mask = labeled_gradient==v1_label_nr

    # Get mask for binoc area
    binarized_azi_binoc = azi_im_sm < settings.v1.exclude_azimuth_threshold

    # Mask borders
    if settings.v1.exclude_border_pixels > 0:
        binarized_azi_binoc[:settings.v1.exclude_border_pixels,:] = False
        binarized_azi_binoc[-settings.v1.exclude_border_pixels:,:] = False
        binarized_azi_binoc[:,:settings.v1.exclude_border_pixels] = False
        binarized_azi_binoc[:,-settings.v1.exclude_border_pixels:] = False

    # Include only elevations within range
    v1_ele_mask = np.logical_and( ele_im_sm > settings.v1.min_elevation, ele_im_sm < settings.v1.max_elevation )

    # Create merged v1b mask
    v1_mask = np.logical_and(v1_mask, binarized_azi_binoc)
    v1_mask = np.logical_and(v1_mask, v1_ele_mask)

    # Get contour
    v1_contour = skimage.measure.find_contours(image=v1_mask, level=0.5)

    # Return mask
    return v1_mask, v1_contour


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~ Function that masks multiple maps using multiple masks ~~~~

def mask_maps(maps, masks):

    # Get a single mask, based on the "AND" of all supplied masks
    if type(masks) in [list,tuple]:
        joint_mask = np.ones_like(masks[0])
        for mask in masks:
            joint_mask = np.logical_and(joint_mask, mask)
    else:
        joint_mask = masks

    if maps is None:
        return joint_mask

    # Mask all maps and return
    if type(maps) in [list,tuple]:
        masked_maps = []
        for map_ in maps:
            masked_map = np.array(map_)
            masked_map[~joint_mask] = np.NaN
            masked_maps.append(masked_map)
        return masked_maps, joint_mask
    else:
        masked_map = np.array(maps)
        masked_map[~joint_mask] = np.NaN
        return masked_map, joint_mask


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~ Function that returns the coordinates of intersections of iso-curves ~~~~

def get_contour_intersectionpoints( im1, im2, range1, range2 ):

    # output variable
    intersections = []
    im1_values = []
    im2_values = []

    # Get contours
    im1_contours = get_iso_contours(im1, range1)
    im2_contours = get_iso_contours(im2, range2)

    # Loop across contours to find intersection coordinates
    print("Finding intersections",end="")
    for im1_cntrs in im1_contours:
        print('.', end="")
        for im2_cntrs in im2_contours:
            if len(im1_cntrs)>0 and len(im2_cntrs)>0:

                # Create an image with summed contours
                contour_im1 = np.zeros_like(im1)
                contour_im2 = np.zeros_like(im2)
                for c in range(len(im1_cntrs)):
                    contour_im1[ np.ceil(im1_cntrs[c][:,0]).astype(int), np.ceil(im1_cntrs[c][:,1]).astype(int) ] = 1
                for c in range(len(im2_cntrs)):
                    contour_im2[ np.ceil(im2_cntrs[c][:,0]).astype(int), np.ceil(im2_cntrs[c][:,1]).astype(int) ] = 1
                contour_im = contour_im1+contour_im2

                # Intersections will have the value 2
                contour_im_bin = contour_im==2

                # Use standard region finding to get x and y coordinate of intersection
                labeled_im = skimage.measure.label(contour_im_bin, connectivity=2)
                reg_props = skimage.measure.regionprops(labeled_im)
                for prop in reg_props:
                    y,x = prop.centroid
                    intersections.append( np.array([y,x]) )
                    im1_values.append(im1[int(y),int(x)])
                    im2_values.append(im2[int(y),int(x)])
    # Concatenate into a nice numpy matrix
    intersections = np.stack(intersections,axis=1).T
    n_intersections = intersections.shape[0]
    print("done (n={})".format(n_intersections))
    return intersections, np.array(im1_values), np.array(im2_values)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~ Function that returns the coordinates of intersections of iso-curves ~~~~

def circ_get_contour_intersectionpoints( im1, im2, range1, range2 ):

    # output variable
    intersections = []
    im1_values = []
    im2_values = []

    # Get contours
    im1_contours,_ = circ_get_iso_contours(im1, range1)
    im2_contours,_ = circ_get_iso_contours(im2, range2)

    # Masked area border
    masked_area = np.logical_or( ~np.isnan(im1), ~np.isnan(im2) )
    masked_area_no_border = skimage.morphology.binary_erosion(masked_area)

    # Loop across contours to find intersection coordinates
    print("Finding intersections",end="")
    for im1_cntrs in im1_contours:
        print('.', end="")
        for im2_cntrs in im2_contours:
            if len(im1_cntrs)>0 and len(im2_cntrs)>0:

                # Create an image with summed contours
                contour_im1 = np.zeros_like(im1)
                contour_im2 = np.zeros_like(im2)
                for c in range(len(im1_cntrs)):
                    contour_im1[ np.ceil(im1_cntrs[c][:,0]).astype(int), np.ceil(im1_cntrs[c][:,1]).astype(int) ] = 1
                for c in range(len(im2_cntrs)):
                    contour_im2[ np.ceil(im2_cntrs[c][:,0]).astype(int), np.ceil(im2_cntrs[c][:,1]).astype(int) ] = 1
                contour_im = contour_im1+contour_im2

                # Intersections will have the value 2
                contour_im_bin = contour_im==2

                # Remove points on the exact border of the mask
                contour_im_bin[~masked_area_no_border] = False

                # Use standard region finding to get x and y coordinate of intersection
                labeled_im = skimage.measure.label(contour_im_bin, connectivity=2)
                reg_props = skimage.measure.regionprops(labeled_im)
                for prop in reg_props:
                    y,x = prop.centroid
                    intersections.append( np.array([y,x]) )
                    im1_values.append(im1[int(y),int(x)])
                    im2_values.append(im2[int(y),int(x)])
    # Concatenate into a nice numpy matrix
    intersections = np.stack(intersections,axis=1).T
    n_intersections = intersections.shape[0]
    print("done (n={})".format(n_intersections))
    return intersections, np.array(im1_values), np.array(im2_values)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~ Function that gets the contours of iso-lines ~~~~

def get_iso_contours(im, iso_contour_range, mask=None):
    im_mskd = np.array(im)
    if mask is not None:
        im_mskd[mask] = np.NaN
    iso_contours = []
    for val in iso_contour_range:
        iso_contours.append( skimage.measure.find_contours(image=im_mskd, level=val) )
    return iso_contours


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~ Function that gets the contours of iso-lines on circular data ~~~~

def circ_get_iso_contours(im, iso_contour_range, mask=None):
    # iso_contour_range: values will be doubled with 180 degrees, so 45 gives lines on 45 and 225 degrees
    im_mskd = np.array(im)
    if mask is not None:
        im_mskd[mask] = np.NaN
    iso_contours = []
    iso_values = []
    for val in iso_contour_range:

        val_low = np.min( [val, np.mod(val+180,360)])
        val_high = np.max( [val, np.mod(val+180,360)])
        im1 = im > val_low
        im2 = im < val_high
        im_thresh = np.logical_and( im1, im2 ) * 1.0

        iso_contours.append( skimage.measure.find_contours(image=im_thresh, level=0.5) )
        iso_values.append(val_low)
        iso_values.append(val_high)
    iso_values.sort()
    return iso_contours, np.array(iso_values)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~ Function that plots the centers of clusters ~~~~

def plot_clusters(clusters=None, cluster_contours=None, markersize=20, markercolor="#000000",
                  marker="o", contour_linestyle="-", contour_color="#000000", contour_linewidth=0.5, ax=None):

    # PLot contours if supplied
    if cluster_contours is not None:
        for c in range(len(cluster_contours)):
            if ax is None:
                plt.plot(cluster_contours[c][:,1],cluster_contours[c][:,0], contour_linestyle,
                         color=contour_color,  markersize=0, linewidth=contour_linewidth)
            else:
                ax.plot(cluster_contours[c][:,1],cluster_contours[c][:,0], contour_linestyle,
                         color=contour_color,  markersize=0, linewidth=contour_linewidth)

    # Plot cluster centers
    if clusters is not None:
        for c in clusters:
            if ax is None:
                plt.plot( c["X"], c["Y"], marker=marker, markersize=markersize, markeredgewidth=1.0,
                         markeredgecolor=markercolor, markerfacecolor='None')
            else:
                ax.plot( c["X"], c["Y"], marker=marker, markersize=markersize, markeredgewidth=1.0,
                         markeredgecolor=markercolor, markerfacecolor='None')


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~ Function that plots the contours of iso-lines ~~~~

def plot_iso_contours(contours, contour_name, settings, msettings, color="#000000",  markersize=0, linewidth=0.5, iso_contour_linestyle=None, ax=None):

    # Get contour specific settings
    if iso_contour_linestyle is None:
        if contour_name.lower() == "odi":
            iso_contour_linestyle = settings.display.iso_odi_contour_linestyle
        if contour_name.lower() == "v1":
            iso_contour_linestyle = settings.display.v1_contour_linestyle
            contours = [contours,]
        if contour_name.lower() == "azimuth":
            iso_contour_linestyle = settings.display.iso_azi_contour_linestyle
        if contour_name.lower() == "elevation":
            iso_contour_linestyle = settings.display.iso_ele_contour_linestyle

    # Plot contours
    for ix in range(len(contours)):
        if isinstance(contours[ix], list):
            for c in range(len(contours[ix])):
                if ax is None:
                    plt.plot(contours[ix][c][:,1],contours[ix][c][:,0], iso_contour_linestyle[ix], color=color,  markersize=markersize, linewidth=linewidth)
                else:
                    ax.plot(contours[ix][c][:,1],contours[ix][c][:,0], iso_contour_linestyle[ix], color=color,  markersize=markersize, linewidth=linewidth)
        else:
            if ax is None:
                plt.plot(contours[ix][:,1],contours[ix][:,0], iso_contour_linestyle[ix], color=color,  markersize=markersize, linewidth=linewidth)
            else:
                ax.plot(contours[ix][:,1],contours[ix][:,0], iso_contour_linestyle[ix], color=color,  markersize=markersize, linewidth=linewidth)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~ Function that plots contours excluding large jumps ~~~~

def plot_contours_no_jumps(contour_coords_yx, offset_yx=(0,0), max_jump=5, linestyle="-", color="#000000",  markersize=0, linewidth=0.5):
    first_coords = contour_coords_yx[0,:]
    contour_coords_yx = np.concatenate([contour_coords_yx, first_coords[np.newaxis,:]], axis=0)
    for ix in range(contour_coords_yx.shape[0]-1):
        y1,y2 = contour_coords_yx[ix,0]+offset_yx[0], contour_coords_yx[ix+1,0]+offset_yx[0]
        x1,x2 = contour_coords_yx[ix,1]+offset_yx[1], contour_coords_yx[ix+1,1]+offset_yx[1]
        if np.sqrt(np.power(y2-y1,2)+np.power(x2-x1,2)) <= max_jump:
            plt.plot([x1,x2],[y1,y2], linestyle, color=color,  markersize=markersize, linewidth=linewidth)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~ Function that redraws the markers in e.g. a swarm plot ~~~~

def redraw_markers( ax, marker_list, color_list, size=5, reduce_x_width=1, markeredgewidth=0.25, x_offset=None ):
    collections = ax.collections
    col_offsets = []
    for nr,col in enumerate(collections):
        type(col.get_offsets())
        col_offsets.append( list(col.get_offsets()) )
    if x_offset is None:
        x_offset = np.zeros((len(collections)))
    ax.cla()
    for nr,col in enumerate(col_offsets):
        for x,y in col:
            plt.plot( nr + (reduce_x_width*(x-nr)) + x_offset[nr], y, color="None", marker=marker_list[nr],
                     markerfacecolor=None, markersize=size, markeredgewidth=0.5, markeredgecolor=color_list[nr] )


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~ Function that calculates the nearest neighbor histogram for preferred orientation ~~~~

def nn_hist_PD( PD, XY, dist_bins, dist_step, param_name):
    # Make PD into PO on 360 range
    PO_360 = np.mod(PD*2,360)

    # Prep output
    n_neurons = len(PO_360)
    n_bins = len(dist_bins)-1
    nn_bins = np.full((n_neurons,n_bins), fill_value=np.NaN)
    nn_counts = np.full((n_neurons,n_bins), fill_value=np.NaN)
    large_number_outside_of_bin_range = dist_bins[-1] + 100

    # Loop neurons
    print("Finding nearest neighbors ({}) ".format(param_name), end="", flush=True)
    for n1 in range(n_neurons):
        if np.mod(n1,100) == 0:
            print(".",end="", flush=True)

        # Get all spatial distances to this neuron (n1)
        d_XY = np.sqrt( (XY[n1,0]-XY[:,0])**2 + (XY[n1,1]-XY[:,1])**2 )

        # Set distance to itself to a large number outside of the bin range
        d_XY[n1] = large_number_outside_of_bin_range

        # Get all orientation differences with this neuron (n1) (divide by two to go to range 0--90)
        d_PO = np.abs(np.abs(np.mod(PO_360[n1]-PO_360[:],360)-180)-180) / 2

        # Get values per bin
        for b in range(n_bins):
            b1 = dist_bins[b]
            b2 = dist_bins[b]+dist_step
            incl_n = np.logical_and(d_XY>=b1,d_XY<b2)
            nn_bins[n1,b] = np.nansum(d_PO[incl_n])
            nn_counts[n1,b] = np.sum(~np.isnan(d_PO[incl_n]*1.0))
    nn_bins = np.nansum(nn_bins,axis=0) / np.nansum(nn_counts,axis=0)
    print(" done (n={})".format(n_neurons))
    return nn_bins


def nn_hist_PD_matrix( PD, XY, dist_bins, dist_step, param_name):
    # Make PD into PO on 360 range
    PO_360 = np.mod(PD*2,360)

    # Prep output
    n_neurons = len(PO_360)
    n_bins = len(dist_bins)-1
    nn_bins = np.full((n_bins,), fill_value=np.NaN)
    large_number_outside_of_bin_range = dist_bins[-1] + 100

    # Loop neurons
    print("Finding nearest neighbors ({}), matrix method ".format(param_name), end="", flush=True)

    # Create a matrix with rows of y coordinates, and columns of x coordinates
    X = XY[:,0]
    Y = XY[:,1]
    Y1 = np.tile(Y[:,np.newaxis],[1,n_neurons])
    Y2 = np.tile(Y[np.newaxis,:],[n_neurons,1])
    X1 = np.tile(X[np.newaxis,:],[n_neurons,1])
    X2 = np.tile(X[:,np.newaxis],[1,n_neurons])
    print(".",end="", flush=True)

    # Create a matrix with all neuron-to-neuron spatial distances
    d_XY = np.sqrt( (X1-X2)**2 + (Y1-Y2)**2  )
    print(".",end="", flush=True)

    # Set the diagonal out of range of the bins
    np.fill_diagonal(d_XY, large_number_outside_of_bin_range)

    # Create a matrix with all neuron-to-neuron angle distances
    PO1 = np.tile(PO_360[:,np.newaxis],[1,n_neurons])
    PO2 = np.tile(PO_360[np.newaxis,:],[n_neurons,1])
    d_PO = np.abs(np.abs(np.mod(PO1-PO2,360)-180)-180) / 2

    # Make 1D arrays
    d_XY = d_XY.ravel()
    d_PO = d_PO.ravel()
    print(".",end="", flush=True)

    # Get values per bin
    for b in range(n_bins):
        print(".",end="", flush=True)
        b1 = dist_bins[b]
        b2 = dist_bins[b]+dist_step
        incl_n = np.logical_and(d_XY>=b1,d_XY<b2)
        nn_bins[b] = np.nanmean(d_PO[incl_n])

    # Return binned data
    print(" done (n={})".format(n_neurons))
    return nn_bins


def nn_deltaorihist_PD_matrix( PD, XY, dist_bins, ori_bins):
    # Make PD into PO on 360 range
    PO_360 = np.mod(PD*2,360)

    # Prep output
    n_neurons = len(PO_360)
    n_dist_bins = len(dist_bins)-1
    n_ori_bins = len(ori_bins)-1
    nn_hists = np.full((n_dist_bins,n_ori_bins), fill_value=np.NaN)
    large_number_outside_of_bin_range = dist_bins[-1] + 100

    # Loop neurons
    print("Finding nearest neighbors for ori hist, matrix method ", end="", flush=True)

    # Create a matrix with rows of y coordinates, and columns of x coordinates
    X = XY[:,0]
    Y = XY[:,1]
    Y1 = np.tile(Y[:,np.newaxis],[1,n_neurons])
    Y2 = np.tile(Y[np.newaxis,:],[n_neurons,1])
    X1 = np.tile(X[np.newaxis,:],[n_neurons,1])
    X2 = np.tile(X[:,np.newaxis],[1,n_neurons])
    print(".",end="", flush=True)

    # Create a matrix with all neuron-to-neuron spatial distances
    d_XY = np.sqrt( (X1-X2)**2 + (Y1-Y2)**2  )
    print(".",end="", flush=True)

    # Set the diagonal out of range of the bins
    np.fill_diagonal(d_XY, large_number_outside_of_bin_range)

    # Create a matrix with all neuron-to-neuron angle distances
    PO1 = np.tile(PO_360[:,np.newaxis],[1,n_neurons])
    PO2 = np.tile(PO_360[np.newaxis,:],[n_neurons,1])
    d_PO = np.abs(np.abs(np.mod(PO1-PO2,360)-180)-180) / 2

    # Make 1D arrays
    d_XY = d_XY.ravel()
    d_PO = d_PO.ravel()
    print(".",end="", flush=True)

    # Get values per bin
    for b in range(n_dist_bins):
        print(".",end="", flush=True)
        b1 = dist_bins[b]
        b2 = dist_bins[b+1]
        incl_n = np.logical_and(d_XY>=b1,d_XY<b2)
        nn_hists[b,:],_ = np.histogram( d_PO[incl_n], bins=n_ori_bins, range=[ori_bins[0],ori_bins[-1]] )

    # Return binned data
    print(" done (n={})".format(n_neurons))
    return nn_hists



def nn_hist_reslen_matrix( PD, PDamp, XY, dist_bins, dist_step):
    # Make PD into PO on 360 range
    PO_360 = np.mod(PD*2,360)

    # Prep output
    n_neurons = len(PO_360)
    n_bins = len(dist_bins)-1
    nn_bins_reslen = np.full((n_bins,), fill_value=np.NaN)
    large_number_outside_of_bin_range = dist_bins[-1] + 100

    # Loop neurons
    print("Finding resultant of nearest neighbors, matrix method ", end="", flush=True)

    # Create a matrix with rows of y coordinates, and columns of x coordinates
    X = XY[:,0]
    Y = XY[:,1]
    Y1 = np.tile(Y[:,np.newaxis],[1,n_neurons])
    Y2 = np.tile(Y[np.newaxis,:],[n_neurons,1])
    X1 = np.tile(X[np.newaxis,:],[n_neurons,1])
    X2 = np.tile(X[:,np.newaxis],[1,n_neurons])
    print(".",end="", flush=True)

    # Create a matrix with all neuron-to-neuron spatial distances
    d_XY = np.sqrt( (X1-X2)**2 + (Y1-Y2)**2  )
    print(".",end="", flush=True)

    # Set the diagonal out of range of the bins
    np.fill_diagonal(d_XY, large_number_outside_of_bin_range)

    # Create a matrix with all neuron angles on one axis
    PO1 = np.tile(PO_360[:,np.newaxis],[1,n_neurons])
    PO2 = np.tile(PO_360[np.newaxis,:],[n_neurons,1])
    PO_stack = np.stack([PO1,PO2],axis=2)

    # Create a matrix with all neuron amplitudes on one axis
    POamp1 = np.tile(PDamp[:,np.newaxis],[1,n_neurons])
    POamp2 = np.tile(PDamp[np.newaxis,:],[n_neurons,1])
    POamp_stack = np.stack([POamp1,POamp2],axis=2)

    # Calculate resultant for all neuron-to-neuron combinations
    c_arr = np.zeros_like(PO_stack).astype(complex)
    c_arr.imag = np.radians(PO_stack)
    c_arr = POamp_stack*np.exp(c_arr)

    # Mean resultant vector
    mean_c_arr = np.sum(c_arr,axis=2) / np.sum(POamp_stack,axis=2)

    # Get resultant angle and resultant length
    res_length = np.abs(mean_c_arr)

    # Make 1D arrays
    d_XY = d_XY.ravel()
    res_length = res_length.ravel()
    print(".",end="", flush=True)

    # Get values per bin
    for b in range(n_bins):
        print(".",end="", flush=True)
        b1 = dist_bins[b]
        b2 = dist_bins[b]+dist_step
        incl_n = np.logical_and(d_XY>=b1,d_XY<b2)
        nn_bins_reslen[b] = np.nanmean(res_length[incl_n])

    # Return binned data
    print(" done (n={})".format(n_neurons))
    return nn_bins_reslen


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~ Function that calculates the nearest neighbor histogram for a linear parameter ~~~~

def nn_hist( param, XY, dist_bins, dist_step, param_name):

    # Prep output
    n_neurons = len(param)
    n_bins = len(dist_bins)-1
    nn_bins = np.full((n_neurons,n_bins), fill_value=np.NaN)
    nn_counts = np.full((n_neurons,n_bins), fill_value=np.NaN)
    large_number_outside_of_bin_range = dist_bins[-1] + 100

    # Loop neurons
    print("Finding nearest neighbors ({})".format(param_name), end="", flush=True)
    for n1 in range(n_neurons):
        if np.mod(n1,100) == 0:
            print(".",end="", flush=True)

        # Get all spatial distances to this neuron (n1)
        d_XY = np.sqrt( (XY[n1,0]-XY[:,0])**2 + (XY[n1,1]-XY[:,1])**2 )

        # Set distance to itself to a large number outside of the bin range
        d_XY[n1] = large_number_outside_of_bin_range

        # Get all orientation differences with this neuron (n1) (divide by two to go to range 0--90)
        d_param = np.abs(param[n1]-param[:])

        # Get values per bin
        for b in range(n_bins):
            b1 = dist_bins[b]
            b2 = dist_bins[b]+dist_step
            incl_n = np.logical_and(d_XY>=b1,d_XY<b2)
            nn_bins[n1,b] = np.nansum(d_param[incl_n])
            nn_counts[n1,b] = np.sum(~np.isnan(d_param[incl_n]*1.0))
    nn_bins = np.nansum(nn_bins,axis=0) / np.nansum(nn_counts,axis=0)
    print("done (n={})".format(n_neurons))
    return nn_bins


def nn_hist_matrix( param, XY, dist_bins, dist_step, param_name):

    # Prep output
    n_neurons = len(param)
    n_bins = len(dist_bins)-1
    nn_bins = np.full((n_bins,), fill_value=np.NaN)
    large_number_outside_of_bin_range = dist_bins[-1] + 100

    # Loop neurons
    print("Finding nearest neighbors ({}), matrix method ".format(param_name), end="", flush=True)

    # Create a matrix with rows of y coordinates, and columns of x coordinates
    X = XY[:,0]
    Y = XY[:,1]
    Y1 = np.tile(Y[:,np.newaxis],[1,n_neurons])
    Y2 = np.tile(Y[np.newaxis,:],[n_neurons,1])
    X1 = np.tile(X[np.newaxis,:],[n_neurons,1])
    X2 = np.tile(X[:,np.newaxis],[1,n_neurons])
    print(".",end="", flush=True)

    # Create a matrix with all neuron-to-neuron spatial distances
    d_XY = np.sqrt( (X1-X2)**2 + (Y1-Y2)**2  )
    print(".",end="", flush=True)

    # Set the diagonal out of range of the bins
    np.fill_diagonal(d_XY, large_number_outside_of_bin_range)

    # Create a matrix with all neuron-to-neuron angle distances
    P1 = np.tile(param[:,np.newaxis],[1,n_neurons])
    P2 = np.tile(param[np.newaxis,:],[n_neurons,1])
    d_param = np.abs(P1-P2)

    # Make 1D arrays
    d_XY = d_XY.ravel()
    d_param = d_param.ravel()
    print(".",end="", flush=True)

    # Get values per bin
    for b in range(n_bins):
        print(".",end="", flush=True)
        b1 = dist_bins[b]
        b2 = dist_bins[b]+dist_step
        incl_n = np.logical_and(d_XY>=b1,d_XY<b2)
        nn_bins[b] = np.nanmean(d_param[incl_n])

    # Return binned data
    print(" done (n={})".format(n_neurons))
    return nn_bins


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~ Function that prints the contents of a dictionary in a compehensive way 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def print_dict( d, indent=0, max_items=5, top_level=True ):
    """ Functions prints the contents of a dictionary in a hierarchical way.
        Uses a recursive procedure. For numpy arrays it only displays the size
        of the array. For lists, only the length if the list is longer than 5 items.
        - Inputs -
        d:        dictionary
        indent:   current indent (mostly for internal use)
    """
    numpy_mem = 0
    for k,v in d.items():
        if isinstance(v,dict):
            print("{}{}:".format("  "*indent,k))
            print_dict(v, indent+1, top_level=False)
        else:
            if isinstance(v,np.ndarray):
                numpy_mem += (1.0*v.nbytes)
                if len(v)>max_items:
                    v = "Numpy ndarray: {} >> {} (mem={:0.2f}MB)".format(v.shape, v.dtype, (1.0*v.nbytes)/(1024*1024))
                else:
                    v = "Numpy ndarray: {} >> {} (mem={:0.2f}MB)".format(v, v.dtype, (1.0*v.nbytes)/(1024*1024))
            elif isinstance(v,list):
                if len(v) > max_items:
                    v = "List: {} (mem={:0.2f}MB)".format(len(v), (1.0*sys.getsizeof(v))/(1024*1024))
                else:
                    v = "List {} (mem={:0.2f}MB)".format(v, (1.0*sys.getsizeof(v))/(1024*1024))
            print("{}{}: {}".format("  "*indent,k,v))
            continue
    if top_level:
        print("Total size of numpy ndarrays in memory: {:0.2f}MB".format(numpy_mem/(1024*1024)))
        print("Total size in memory: {:0.2f}MB".format((numpy_mem+sys.getsizeof(d)*1.0)/(1024*1024)))


