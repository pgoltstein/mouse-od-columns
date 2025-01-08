#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Functions to make hls maps

Created on Mon Jan 4, 2022

@author: pgoltstein
"""

# Imports
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import skimage.morphology
import skimage.filters
from skimage.transform import resize as imresize
from scipy.ndimage.filters import gaussian_filter
from tqdm import tqdm


def load_trialimages(plane_no, datapath):
    """ Loads a saved dfof responsemap of each trial into a 3d matrix [y_res,x_res,trials]
    """

    # Prepare filenames
    savedatapath = os.path.join(datapath,"hls")
    bsimages_file = os.path.join(savedatapath, "trialimages-bs-plane{}.npy".format(plane_no))
    stimimages_file = os.path.join(savedatapath, "trialimages-stim-plane{}.npy".format(plane_no))

    # Load df_images
    print("Loading baseline images from: {}".format(bsimages_file))
    bs_images = np.load(bsimages_file)
    print("Loading stimulus images from: {}".format(stimimages_file))
    stim_images = np.load(stimimages_file)

    # Return data
    return bs_images, stim_images


def hlsmap( hlsname, stimuli, bs_images, stim_images, max_dfof=0, colormap="hsv", aspect_ratio=1.0, scale_by_im=None, show_colorbar=True ):
    """ Creates an HLS map as RGB array [yres,xres,rgb]
    """

    print("Creating hlsmap for: {}".format(hlsname))

    # Get info on dimensions etc
    x_res, y_res, n_trials = bs_images.shape
    unique_stimuli = np.unique(stimuli)
    n_stimuli = len(unique_stimuli)

    # Prepare colorspace
    colors = np.zeros((n_stimuli,3))
    if colormap.lower() == "eye":
        colors[0,:] = [1.0, 0.0, 0.0]
        colors[1,:] = [0.0, 0.0, 1.0]
    elif colormap.lower() == "eye-reverse":
        colors[0,:] = [0.0, 0.0, 1.0]
        colors[1,:] = [1.0, 0.0, 0.0]
    elif colormap.lower() == "category":
        colors[0,:] = [0.0, 0.8, 1.0]
        colors[1,:] = [1.0, 0.0, 0.4]
    elif colormap.lower() == "category-reverse":
        colors[0,:] = [1.0, 0.0, 0.4]
        colors[1,:] = [0.0, 0.8, 1.0]
    elif colormap.lower() == "3colors":
        colors[0,:] = [1.0, 0.0, 0.0]
        colors[1,:] = [0.0, 1.0, 0.0]
        colors[2,:] = [0.0, 0.0, 1.0]
    elif colormap.lower() == "3colors-reverse":
        colors[2,:] = [1.0, 0.0, 0.0]
        colors[1,:] = [0.0, 1.0, 0.0]
        colors[0,:] = [0.0, 0.0, 1.0]
    elif colormap.lower() == "5colors":
        colors[0,:] = [1.0, 0.0, 0.0]
        colors[1,:] = [1.0, 1.0, 0.0]
        colors[2,:] = [0.0, 1.0, 0.0]
        colors[3,:] = [0.0, 1.0, 1.0]
        colors[4,:] = [0.0, 0.0, 1.0]
    elif colormap.lower() == "5colors-reverse":
        colors[4,:] = [1.0, 0.0, 0.0]
        colors[3,:] = [1.0, 1.0, 0.0]
        colors[2,:] = [0.0, 1.0, 0.0]
        colors[1,:] = [0.0, 1.0, 1.0]
        colors[0,:] = [0.0, 0.0, 1.0]
    else:
        cmap = matplotlib.cm.get_cmap(colormap)
        if colormap == "hsv":
            for s in range(n_stimuli):
                colors[s,:] = cmap(float(s)/n_stimuli)[:3]
        else:
            for s in range(n_stimuli):
                colors[s,:] = cmap(float(s)/(n_stimuli-1))[:3]

    print("bs_images intensities: {} (min)  {} (max)".format(np.nanmin(bs_images),np.nanmax(bs_images)))
    print("stim_images intensities: {} (min)  {} (max)".format(np.nanmin(stim_images),np.nanmax(stim_images)))

    # Filter images (smoothing)
    disk_kernel = skimage.morphology.disk(radius=1)
    with tqdm(total=n_trials, desc="Filtering", unit="trial") as bar:
        for t in range(n_trials):
            bs_images[:,:,t] = skimage.filters.rank.median(bs_images[:,:,t], footprint=disk_kernel)
            stim_images[:,:,t] = skimage.filters.rank.median(stim_images[:,:,t], footprint=disk_kernel)
            bar.update(1)

    print("bs_images intensities: {} (min)  {} (max)".format(np.nanmin(bs_images),np.nanmax(bs_images)))
    print("stim_images intensities: {} (min)  {} (max)".format(np.nanmin(stim_images),np.nanmax(stim_images)))

    # Prepare data containers
    df_images = np.zeros( (x_res, y_res, n_stimuli) )

    # Get one average baseline image and set minimum value to 1
    bs_im = np.mean(bs_images, axis=2)
    bs_im[bs_im<1] = 1.0
    print("bs_im intensities: {} (min)  {} (max)".format(np.nanmin(bs_im),np.nanmax(bs_im)))

    # Get one df/f image per stimulus
    for stim_nr,stim_id in enumerate(unique_stimuli):

        # Get indices for this stimulus id
        stim_indices = stimuli==stim_id

        # Mean, and df/f
        df_images[:,:,stim_nr] = (np.mean(stim_images[:,:,stim_indices], axis=2) - bs_im) / bs_im
    print("df_images intensities: {} (min)  {} (max)".format(np.nanmin(df_images),np.nanmax(df_images)))

    # Change to correct aspect ratio
    if aspect_ratio != 1.0:
        print("Correcting to aspect ratio {}".format(aspect_ratio))
        new_x = int(x_res * aspect_ratio)
        df_images_new = np.zeros((y_res,new_x,n_stimuli))
        for s in range(n_stimuli):
            df_images_new[:,:,s] = imresize(df_images[:,:,s], (y_res,new_x), order=0)
        x_res = new_x
        df_images = df_images_new

    # Create H map (hue=stim preference)
    H = np.argmax(df_images, axis=2)
    # print("H: {}  {}".format(np.nanmin(H),np.nanmax(H)))

    # Create L map (lightness=response amplitude) and clip responses to range
    if max_dfof == 0:
        max_dfof = np.round(np.percentile(df_images.ravel(),99) * 100)
        if scale_by_im is not None:
            max_dfof = np.round(max_dfof * 0.75)
    L = np.max(df_images, axis=2) * (100/float(max_dfof))
    L[L<0] = 0.0
    # print("L: {}  {}".format(np.nanmin(L),np.nanmax(L)))

    # Calculate resultant and circular variance for 'S' (saturation=selectivity)
    phasor = np.zeros( (y_res, x_res, n_stimuli), dtype=np.complex_)
    amp_sorted = np.sort(df_images,axis=2)
    amp_sorted[amp_sorted<0.0001] = 0.0001
    min_as = np.min(amp_sorted)
    max_as = np.max(amp_sorted)
    amp_sorted = (amp_sorted-min_as) / (max_as-min_as)
    for s in range(n_stimuli):
        phasor[:,:,s] = amp_sorted[:,:,s] * np.exp( 1j * (s/n_stimuli) * 2 * np.pi)
    resultant = np.abs(np.sum(phasor,axis=2)) / np.sum(np.abs(phasor),axis=2)
    circvar = 1-resultant

    # Convert H map to RGB
    H = colors[H]

    # Stack L, circvar and resultant to 3d arrays
    L = np.stack([L,L,L], axis=2)
    resultant = np.stack([resultant,resultant,resultant], axis=2)
    circvar = np.stack([circvar,circvar,circvar], axis=2)

    # Add amplitude and saturarion by elementwise multiplications
    HLS = (H * L * resultant) + L * circvar
    # print("HLS {}  {}".format(np.nanmin(HLS),np.nanmax(HLS)))
    
    # If scale_by_im is set, scale the brightness by the supplied image
    if scale_by_im is not None:
        scale_max = np.percentile(scale_by_im.ravel(),99)
        scale_min = np.percentile(scale_by_im.ravel(),1)
        scale_by_im = (scale_by_im - scale_min) / (scale_max - scale_min)
        if aspect_ratio != 1.0:
            scale_by_im = imresize(scale_by_im, (y_res,x_res), order=0)
        scale_by_im[scale_by_im>1.0] = 1.0
        scale_by_im[scale_by_im<0.0] = 0.0
        scale_by_im = np.stack([scale_by_im,scale_by_im,scale_by_im], axis=2)
        HLS = HLS * scale_by_im

    HLSmax = np.nanmax(HLS,axis=2)
    HLSmax[HLSmax<1.0] = 1.0
    HLSmax = np.stack([HLSmax,HLSmax,HLSmax], axis=2)
    HLS = HLS / HLSmax
    # print("HLS after scaling {}  {}".format(np.nanmin(HLS),np.nanmax(HLS)))

    if show_colorbar:
        color_bar = np.zeros((10,x_res)).astype(int)
        x_offset = int(0.1 * x_res)
        x_res_offsetted = int(x_res - (2*x_offset))
        # Create full bar
        for stim_nr in range(n_stimuli):
            x_start = int( (stim_nr/n_stimuli) * x_res_offsetted ) + x_offset
            x_end = int( ((stim_nr+1)/n_stimuli) * x_res_offsetted ) + x_offset
            color_bar[:,x_start:x_end] = stim_nr
        color_bar = colors[color_bar]
        # Now black out edges
        color_bar[:,:x_offset,:] = 0
        color_bar[:,-x_offset:,:] = 0
        for stim_nr in range(n_stimuli):
            x_start = int( ((stim_nr/n_stimuli) * x_res_offsetted) ) + x_offset
            x_end   = int( ((stim_nr/n_stimuli) * x_res_offsetted) + (0.2*(x_res_offsetted/n_stimuli)) ) + x_offset
            color_bar[:,x_start:x_end,:] = 0
            x_start = int( ( ((stim_nr+1)/n_stimuli) * x_res_offsetted) - (0.2*(x_res_offsetted/n_stimuli)) ) + x_offset
            x_end   = int( ( ((stim_nr+1)/n_stimuli) * x_res_offsetted) ) + x_offset
            color_bar[:,x_start:x_end,:] = 0
        HLS = np.concatenate([HLS,color_bar], axis=0)

    return HLS
