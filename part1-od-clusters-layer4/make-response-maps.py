#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This script loads data of a single plane of a 4-plane recording, shows the FOV, makes the HLS map and plots PSTH's and tuning curves of example data

Created on Friday 13 Dec 2024

python make-response-maps.py O03 1

@author: pgoltstein
"""

# Imports
import sys, os, glob
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize as imresize
from skimage.exposure import rescale_intensity

# Local imports
sys.path.append('../xx_analysissupport')
import scanimagestack
import auxrecorder
import matlabstimulus
import suite2pdata
import responsemap
import analysistools
import plottingtools
import singleplaneodfunctions

# Arguments
import argparse


#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Arguments

parser = argparse.ArgumentParser( description = "This script loads data of a single plane of a 4-plane recording, shows the FOV, makes the HLS map and plots PSTH's and tuning curves of example data.\n (written by Pieter Goltstein - Dec 2024)")
parser.add_argument('mouse', type=str, help= 'name of the mouse to analyze')
parser.add_argument('imagingplane', type=str, help= 'number of the imaging plane to process')
parser.add_argument('-m', '--maxhls', type=int, default=20, help='manually set the maximum response amplitude in the HLS map')
parser.add_argument('-r', '--reverseodi',  action="store_true", default=False, help='Flag flips the odi value for right hemisphere recordings')
parser.add_argument('-e', '--examplecells',  type=int, nargs="+", default=[], help='List of example cells to show')
args = parser.parse_args()


#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Settings

settingspath = "../settings"
savepath = "../../figureout"
datapath = os.path.join("../../data/part1-responsemapdata", "{}-L4-OD".format(args.mouse))
print(f"{datapath=}")

# Find image and aux settings files
imagesettingsfile = glob.glob( os.path.join( settingspath, "*.imagesettings.py" ) )[0]
auxsettingsfile = glob.glob( os.path.join( settingspath, "*.auxsettings.py" ) )[0]
if args.mouse == "O03":
    figname = "Fig-1"
elif int(args.mouse[1:]) < 20:
    figname = "Fig-S1"
else:
    figname = "Fig-S7"



#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Local functions

def adjust_intensity_aspect( I, percentile_range=(1,99), aspect_ratio=1.0):
    # adjust intensity
    p_low, p_high = np.percentile(I, percentile_range)
    I = rescale_intensity(I, in_range=(p_low, p_high))

    # Spatial rescale to aspect ratio
    new_y = int(I.shape[0])
    new_x = int(I.shape[0] * aspect_ratio)
    I = imresize(I, (new_y,new_x), order=None)

    return I


#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Step 1: Load session data

# Open imagestack
Im = scanimagestack.XYT( filepath=datapath, imagesettingsfile=imagesettingsfile )
print(Im)
aspect_ratio = Im.fovsize["x"]/Im.fovsize["y"]
print("Aspect ratio = {} X/Y".format(aspect_ratio))

# Load Aux data
print("\nLoading aux data:")
Aux = auxrecorder.LvdAuxRecorder( filepath=datapath, filename="*.lvd", auxsettingsfile=auxsettingsfile, nimagingplanes=Im.nplanes )
print(Aux)

# Load stimulus data
print("\nLoading stimulus data:")
Stim = matlabstimulus.StimulusData(datapath)
print(Stim)

# Load suite2pdata
print("\nLoading suite2pdata:")
S2p = suite2pdata.Suite2pData(datapath)
print(S2p)

# Select plane
S2p.plane = int(args.imagingplane)
S2p.select_neurons("iscell")

#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Step 2: Plot overview images of field of view

# Get and adjust fov image
I = adjust_intensity_aspect( S2p.image, percentile_range=(1,99), aspect_ratio=aspect_ratio )
I = np.stack([I,I,I], axis=2)

# Save to image
savefile = os.path.join(savepath, figname+"a-{}-im-plane{}".format(args.mouse,args.imagingplane) + '.png')
print("Saving image to file: {}".format(savefile))
plt.imsave(savefile, I)


#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Step 3: Create and plot HLS map of field of view

# Create dFoF maps
bs_images, stim_images = responsemap.load_trialimages(plane_no=args.imagingplane, datapath=datapath)

# Get HLS map
if args.reverseodi:
    HLS = responsemap.hlsmap( hlsname="eye-plane{}".format(args.imagingplane), stimuli=Stim.eye, bs_images=bs_images, stim_images=stim_images, max_dfof=args.maxhls, colormap="eye-reverse", aspect_ratio=aspect_ratio, scale_by_im=None, show_colorbar=False )
else:
    HLS = responsemap.hlsmap( hlsname="eye-plane{}".format(args.imagingplane), stimuli=Stim.eye, bs_images=bs_images, stim_images=stim_images, max_dfof=args.maxhls, colormap="eye", aspect_ratio=aspect_ratio, scale_by_im=None, show_colorbar=False )

# Save to image
savefile = os.path.join(savepath, figname+"a-{}-hls-od-plane{}".format(args.mouse,args.imagingplane) + "-max{}".format(args.maxhls) + '.png')
print("Saving hlsmap to file: {}".format(savefile))
plt.imsave(savefile, HLS)


#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Step 4: If example cells specified, continue

if len(args.examplecells) > 0:

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Step 4a: Calculate tuning curve / psth data

    print("\nStimulus locked analysis settings:".format())
    bs_frame_range = [ int(-0.9*Stim.iti_duration*Aux.imagingsf), 0 ]
    tc_frame_range = [ 0, int(Stim.stimulus_duration*Aux.imagingsf) ]
    psth_frame_range = [int(-0.5*Stim.iti_duration*Aux.imagingsf), int((Stim.stimulus_duration+(0.5*Stim.iti_duration))*Aux.imagingsf) ]
    print("Baseline range: {}".format(bs_frame_range))
    print("Stimulus range: {}".format(tc_frame_range))
    print("PSTH range: {}".format(psth_frame_range))
    xvalues = np.arange(psth_frame_range[0],psth_frame_range[-1])

    # Load psth's and tuning curves in spike data
    datamat_sp = S2p.spikes
    bs_sp = analysistools.tm(datamat_sp, Aux.stimulus_onsets, bs_frame_range, Stim.eye, stimuli2=Stim.direction)
    print(bs_sp.shape)
    tm_sp = analysistools.tm(datamat_sp, Aux.stimulus_onsets, tc_frame_range, Stim.eye, stimuli2=Stim.direction)
    print(tm_sp.shape)
    psth_sp = analysistools.psth(datamat_sp, Aux.stimulus_onsets, psth_frame_range, Stim.eye, stimuli2=Stim.direction)
    print(psth_sp.shape)

    # Flip eyes if other hemisphere
    if args.reverseodi:
        bs_sp = bs_sp[:,::-1,:,:]
        tm_sp = tm_sp[:,::-1,:,:]
        psth_sp = psth_sp[:,::-1,:,:]

    # Baselined tuning matrix
    tm_spbs = tm_sp - bs_sp

    # Get tuning: ROInr, ODI, Pref dir, Pref ori, Circ var, Bandwidth, Significance
    tuning_parameters_sp,parameter_names_sp = singleplaneodfunctions.calculate_OD_orientation_tuning( tm_spbs )

    # Add coordinates
    x,y = S2p.x,S2p.y
    parameter_names_sp.extend(["x","y"])
    tuning_parameters_sp = np.concatenate([tuning_parameters_sp,x[:,np.newaxis]],axis=1)
    tuning_parameters_sp = np.concatenate([tuning_parameters_sp,y[:,np.newaxis]],axis=1)

    # Select significant neurons on spike data
    print("# neurons (total): {}".format(tuning_parameters_sp.shape[0]))
    sign_nrns = tuning_parameters_sp[:, parameter_names_sp.index("Significance")]<0.05
    bs_sp = bs_sp[sign_nrns,:,:,:]
    tm_sp = tm_sp[sign_nrns,:,:,:]
    psth_sp = psth_sp[sign_nrns,:,:,:,:]
    tm_spbs = tm_spbs[sign_nrns,:,:,:]
    tuning_parameters_sp = tuning_parameters_sp[sign_nrns,:]
    print("# neurons (tuned): {}".format(tuning_parameters_sp.shape[0]))


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Step 4b: Display example neurons

    # variables of relevance
    odi = tuning_parameters_sp[:, parameter_names_sp.index("ODI")]
    prefeye = ((odi >= 0) * 1.0).astype(int) # 0=ipsi, 1=contra
    prefdir = (tuning_parameters_sp[:, parameter_names_sp.index("Pref dir")]/45).astype(int)
    tc = np.mean(tm_sp,axis=3)
    respamp = np.zeros((tc.shape[0]))
    for n in range(tc.shape[0]):
        respamp[n] = tc[n,prefeye[n],prefdir[n]]

    # print some info of best neurons and show in image
    x = tuning_parameters_sp[:, parameter_names_sp.index("x")] * aspect_ratio
    y = tuning_parameters_sp[:, parameter_names_sp.index("y")]

    # Plot circles at location of neurons on response map
    fig,ax = plottingtools.init_figure(fig_size=(20,20))
    plt.imshow(I)
    for n in args.examplecells:
        print("neuron {}: odi={}, respamp={}".format( n, odi[n], respamp[n] ))
        cc = plt.Circle(( x[n], y[n]), radius=10, fill=False, color="#00FF00" )
        ax.add_artist( cc )
        plt.text( x[n], y[n], "{}".format(n), color="#00FF00" )

    example_cell_str = "-".join(str(num) for num in args.examplecells)
    savefile = os.path.join(savepath, "{}-plane{}-labeledneurons{}-in-image".format(args.mouse, args.imagingplane, example_cell_str) + '.png')
    print("Saving neuron-annotated image to file: {}".format(savefile))
    plottingtools.finish_figure( filename=savefile, wspace=0.2, hspace=0.2 )


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Step 4c: Plot psth's of example cells

    # Plot psths
    singleplaneodfunctions.generic_psth_overview_OD( psth_sp, tm_spbs, xvalues, include_neurons=args.examplecells, x_labels=np.arange(0,316,45), y_labels=["I","C"], savepath=savepath, scalebar=100, n_rows=3, n_cols=1, savename=figname+"b-{}-plane{}-psths-od-neurons{}".format(args.mouse, args.imagingplane, example_cell_str) )


    #<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
    # Step 5: Plot traces in one big matrix

    # variables of relevance
    odi = tuning_parameters_sp[:, parameter_names_sp.index("ODI")]
    best_neurons = np.argsort(odi)[::-1]
    prefdir = (tuning_parameters_sp[:, parameter_names_sp.index("Pref dir")]/45).astype(int)
    n_psth_frames = psth_frame_range[1]-psth_frame_range[0]

    # Now compile these neurons in a 2d matrix
    psth_lin = np.zeros((len(best_neurons), psth_sp.shape[1]*psth_sp.shape[2]*psth_sp.shape[4]))
    for nr,n in enumerate(best_neurons):

        # Get preferred stimulus per eye of neuron
        _, ipsi_pref_ix = analysistools.preferreddirection(tc[n,0,:])
        _, contra_pref_ix = analysistools.preferreddirection(tc[n,1,:])

        # Get psth and baseline of neuron
        neuron_psth = psth_sp[n,:,:,:,:]
        neuron_bs = bs_sp[n,:,:,:,np.newaxis]
        neuron_bs = np.tile(neuron_bs, [1,1,1,neuron_psth.shape[3]] )

        # Baseline correct
        neuron_psth = neuron_psth-neuron_bs

        # Average trials and split ipsi/contra
        neuron_psth = np.mean(neuron_psth,axis=2).ravel()
        neuron_psth_ipsi = neuron_psth[:(psth_sp.shape[2]*n_psth_frames)]
        neuron_psth_contra = neuron_psth[(psth_sp.shape[2]*n_psth_frames):]

        # Get number of frames to shift for aligning to pref dir
        pref_dir_shift_ipsi = np.mod( (np.arange(neuron_psth_ipsi.shape[0]) - (neuron_psth_ipsi.shape[0]-((ipsi_pref_ix-1)*n_psth_frames))), neuron_psth_ipsi.shape[0]).astype(int)
        pref_dir_shift_contra = np.mod( (np.arange(neuron_psth_ipsi.shape[0]) - (neuron_psth_ipsi.shape[0]-((contra_pref_ix-1)*n_psth_frames))), neuron_psth_ipsi.shape[0]).astype(int)

        # Store in plot matrix, contra left, ipsi right
        psth_lin[nr,:(psth_sp.shape[2]*n_psth_frames)] =  neuron_psth_contra[pref_dir_shift_contra]
        psth_lin[nr,(psth_sp.shape[2]*n_psth_frames):] =  neuron_psth_ipsi[pref_dir_shift_ipsi]

    # Save to image
    max_int = np.percentile(psth_lin, 95)
    savefile = os.path.join(savepath, figname+"b-{}-plane{}-traces2d".format(args.mouse,args.imagingplane) + "-max{:3.0f}".format(max_int*100) +  '.png')
    print("Saving traces (2d) to file: {}".format(savefile))
    plt.imsave(savefile, psth_lin, cmap="Greys", vmin=0.0, vmax=max_int)


#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# That's all folks !!
print("\nDone.\n")
