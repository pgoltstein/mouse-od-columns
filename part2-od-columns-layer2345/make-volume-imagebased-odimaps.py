#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This script loads responsemaps of a L23-L4-L5 experiment and makes ODI maps across depth

python make-volume-imagebased-odimaps.py O03

Created on Wednesday 5 May 2022

@author: pgoltstein
"""


# Imports
import sys, os
import numpy as np
import matplotlib.pyplot as plt
import skimage.morphology
import skimage.filters
from skimage.transform import resize as imresize
from tqdm import tqdm

# Local imports
sys.path.append('../xx_analysissupport')
import plottingtools

# Module settings
plottingtools.font_size = { "title": 6, "label": 5, "tick": 5, "text": 5, "legend": 5 }

# Probably shouldn't do this, but got tired of "mean of empty slice" warnings
import warnings
warnings.filterwarnings('ignore')

# Arguments
import argparse


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Arguments

parser = argparse.ArgumentParser( description = "This script loads responsemaps of a L23-L4-L5 experiment and makes ODI maps across depth.\n (written by Pieter Goltstein - May 2022)")
parser.add_argument('mousename', type=str, help='name of the mouse to analyze')
parser.add_argument('-n', '--nbins', type=int, default=4, help='manually set the number of depth bins in which to average the odimaps (default=4).')
args = parser.parse_args()


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Settings

# Path settings
savepath = "../../figureout"
datapath_maps = os.path.join("../../data/part2-responsemaps-od-layer2345")
print(f"{datapath_maps=}")

figname = {"O02": "Fig-S8a-",
           "O03": "Fig-S8c-",
           "O06": "Fig-S8e-",
           "O09": "Fig-S8g-",
           "O10": "Fig-S8i-"}

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

# Map settings
median_filter_images = False
disk_kernel = skimage.morphology.disk(radius=1)
if args.nbins == 4 and args.mousename=="O03":
    downsample_by = 1
else:
    downsample_by = 4

# Depth bins
if args.nbins == 4:
    depth_bins = [170,260,350,440,540] # last bin should include extra plane step
if args.nbins == 12:
    depth_bins = np.arange(170,541,30)
    depth_bins = depth_bins + 10
    depth_bins[0] = 170
n_bins = len(depth_bins)-1


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Load data

# Actual data
datafile = np.load(os.path.join(datapath_maps,"{}-responsemaps-volume.npz".format(args.mousename)))
lefteye_response_maps, lefteye_baseline_maps = datafile["lefteye_response_maps"], datafile["lefteye_baseline_maps"]
righteye_response_maps, righteye_baseline_maps = datafile["righteye_response_maps"], datafile["righteye_baseline_maps"]
aspect_ratio = datafile["aspect_ratio"]
y_res, x_res, n_planes = lefteye_response_maps.shape

if downsample_by > 1:
    x_res = int(np.round(x_res / downsample_by))
    y_res = int(np.round(y_res / downsample_by))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Loop planes and calculate dF/F maps

# Prepare data containers
df_images = np.zeros( (x_res, y_res, n_stimuli, n_planes) )

# # Loop images to filter (smoothing)
if median_filter_images:
    with tqdm(total=n_planes, desc="Filtering", unit="plane") as bar:
        for plane_nr in range(n_planes):
            lefteye_baseline_maps[:,:,plane_nr] = skimage.filters.rank.median(lefteye_baseline_maps[:,:,plane_nr], selem=disk_kernel)
            righteye_baseline_maps[:,:,plane_nr] = skimage.filters.rank.median(righteye_baseline_maps[:,:,plane_nr], selem=disk_kernel)
            lefteye_response_maps[:,:,plane_nr] = skimage.filters.rank.median(lefteye_response_maps[:,:,plane_nr], selem=disk_kernel)
            righteye_response_maps[:,:,plane_nr] = skimage.filters.rank.median(righteye_response_maps[:,:,plane_nr], selem=disk_kernel)
            bar.update(1)

# Loop image planes to calculate dF/F
depths = []
current_depth = start_depth
with tqdm(total=start_depth+(n_planes*depth_increment), initial=start_depth, desc="Calculating dF/F", unit="um") as bar:
    for plane_nr in range(n_planes):
        depths.append(current_depth)

        # Get one average baseline image and set minimum value to 1
        bs_im = (lefteye_baseline_maps[:,:,plane_nr].astype(float) + righteye_baseline_maps[:,:,plane_nr].astype(float)) / 2
        bs_im[bs_im<1] = 1.0

        # # Smooth images
        right_eye_im = righteye_response_maps[:,:,plane_nr].astype(float)
        left_eye_im = lefteye_response_maps[:,:,plane_nr].astype(float)

        # Downsample if requested
        if downsample_by > 1:
            bs_im = imresize(bs_im, (y_res, x_res))
            right_eye_im = imresize(right_eye_im, (y_res, x_res))
            left_eye_im = imresize(left_eye_im, (y_res, x_res))

        # Mean, and df/f
        df_images[:,:,0,plane_nr] = (right_eye_im - bs_im) / bs_im
        df_images[:,:,1,plane_nr] = (left_eye_im - bs_im) / bs_im

        # increase depth step
        current_depth += depth_increment
        bar.update(depth_increment)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Main >> Average planes into x bins, create ODI map and save as image

odi_maps = np.zeros((y_res,int(x_res * aspect_ratio),n_bins))

fig,ax = plottingtools.init_figure(fig_size=(24*aspect_ratio,8))
for bin_nr,(depth1,depth2) in enumerate(zip(depth_bins[:-1],depth_bins[1:])):

    # Calulate bin nrs
    start_bin = int((depth1-start_depth)/depth_increment)
    end_bin = int((depth2-start_depth)/depth_increment)
    print("{}-{}: {}-{}".format(depth1,depth2,start_bin,end_bin))

    # Get data from depth bin
    avg_map = np.nanmean(df_images[:,:,:,start_bin:end_bin], axis=3)

    # Create ODI map
    odi_map = (avg_map[:,:,0]-avg_map[:,:,1]) / (avg_map[:,:,0]+avg_map[:,:,1])

    # Change to correct aspect ratio
    new_x = int(x_res * aspect_ratio)
    odi_map = imresize(odi_map, (y_res,new_x), order=0)

    # Save as image
    if args.nbins == 4 and args.mousename=="O03":
        odimap_file = os.path.join(savepath, "Fig-2b-{}-odimap-d{}-d{}.png".format(args.mousename, depth1, depth2-10))
    else:
        odimap_file = os.path.join(savepath, figname[args.mousename]+"{}-odimap-d{}-d{}.png".format(args.mousename, depth1, depth2-10))

    print("Saving odi map to file: {}".format(odimap_file))
    plt.imsave(fname=odimap_file, arr=odi_map, vmin=-0.5, vmax=0.5, cmap="RdBu_r")


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Done

print("\nDone.. that's all folks!")
