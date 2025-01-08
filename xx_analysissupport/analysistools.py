#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Functions to analyze imaging datasets

Created on Fri Dec 4, 2020

@author: pgoltstein
"""

import numpy as np
import scipy.optimize as optimize


def tm(data_mat, frame_ixs, frame_range, stimuli1, stimuli2=None):
    """ Returns a tuning matrux (frame-averaged psth) (neuron,stimulus1,trial) or (neuron,stimulus1,stimulus2,trial)
        Note: Assumes equal number of trials per stimulus
    """
    n_neurons = data_mat.shape[0]
    unique_stimuli1 = np.unique(stimuli1)
    n_stimuli1 = len(unique_stimuli1)
    frame_indices = np.arange( frame_range[0], frame_range[1] ).astype(int)

    if stimuli2 is None:
        n_trials = np.sum(stimuli1==unique_stimuli1[0])
        tm_mat = np.zeros((n_neurons,n_stimuli1,n_trials))
        for s1_ix,s1 in enumerate(unique_stimuli1):
            stimulus_trials = np.argwhere(stimuli1==s1)
            stimulus_trial_frame_ixs = frame_ixs[stimulus_trials]
            for t_ix,t_frame in enumerate(stimulus_trial_frame_ixs):
                tm_mat[:,s1_ix,t_ix] = np.nanmean(data_mat[:,t_frame+frame_indices],axis=1)
    else:
        unique_stimuli2 = np.unique(stimuli2)
        n_stimuli2 = len(unique_stimuli2)
        n_trials = np.sum(np.logical_and(stimuli1==unique_stimuli1[0], stimuli2==unique_stimuli2[0]))
        tm_mat = np.zeros((n_neurons,n_stimuli1,n_stimuli2,n_trials))
        for s1_ix,s1 in enumerate(unique_stimuli1):
            for s2_ix,s2 in enumerate(unique_stimuli2):
                stimulus_trials = np.argwhere(np.logical_and(stimuli1==s1,stimuli2==s2))
                stimulus_trial_frame_ixs = frame_ixs[stimulus_trials]
                for t_ix,t_frame in enumerate(stimulus_trial_frame_ixs):
                    tm_mat[:,s1_ix,s2_ix,t_ix] = np.nanmean(data_mat[:,t_frame+frame_indices],axis=1)
    return tm_mat


def psth(data_mat, frame_ixs, frame_range, stimuli1, stimuli2=None):
    """ Returns a peri-stimulus time histogram (neuron,stimulus1,trial,frame) or (neuron,stimulus1,stimulus2,trial,frame)
        Note: Assumes equal number of trials per stimulus
    """
    n_neurons = data_mat.shape[0]
    unique_stimuli1 = np.unique(stimuli1)
    n_stimuli1 = len(unique_stimuli1)
    frame_indices = np.arange( frame_range[0], frame_range[1] ).astype(int)
    n_frames = frame_indices.shape[0]

    if stimuli2 is None:
        n_trials = np.sum(stimuli1==unique_stimuli1[0])
        psth_mat = np.zeros((n_neurons,n_stimuli1,n_trials,n_frames))
        for s1_ix,s1 in enumerate(unique_stimuli1):
            stimulus_trials = np.argwhere(stimuli1==s1)
            stimulus_trial_frame_ixs = frame_ixs[stimulus_trials]
            for t_ix,t_frame in enumerate(stimulus_trial_frame_ixs):
                psth_mat[:,s1_ix,t_ix,:] = data_mat[:,t_frame+frame_indices]
    else:
        unique_stimuli2 = np.unique(stimuli2)
        n_stimuli2 = len(unique_stimuli2)
        n_trials = np.sum(np.logical_and(stimuli1==unique_stimuli1[0], stimuli2==unique_stimuli2[0]))
        psth_mat = np.zeros((n_neurons,n_stimuli1,n_stimuli2,n_trials,n_frames))
        for s1_ix,s1 in enumerate(unique_stimuli1):
            for s2_ix,s2 in enumerate(unique_stimuli2):
                stimulus_trials = np.argwhere(np.logical_and(stimuli1==s1,stimuli2==s2))
                stimulus_trial_frame_ixs = frame_ixs[stimulus_trials]
                for t_ix,t_frame in enumerate(stimulus_trial_frame_ixs):
                    psth_mat[:,s1_ix,s2_ix,t_ix,:] = data_mat[:,t_frame+frame_indices]
    return psth_mat


def preferreddirection( tuningcurve, angles=None ):
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

def preferredorientation( tuningcurve, angles=None ):
    """ Returns the angle and index of the preferred orientation
        - Inputs -
        tuningcurve: 1D array of neuronal responses per stimulus
        angles:      Array with angles (equal sampling across 360 degrees)
        returns tuple (angle, angle_ix)
    """
    # Calculate angles if not supplied
    if angles is None:
        angles = np.arange(0,360,360/tuningcurve.shape[0])

    # Average across opposite directions to get orientation curve
    half_range = int(tuningcurve.shape[0]/2)
    orientationcurve = tuningcurve[:half_range]+tuningcurve[half_range:]

    # Find index of largest value
    pref_ix = np.argmax(orientationcurve)

    # Return angle and index of largest value
    return angles[pref_ix],pref_ix

def odi(ipsi_tc, contra_tc, method=0):
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
        _, ipsi_pref_ix = preferreddirection(ipsi_tc)
        _, contra_pref_ix = preferreddirection(contra_tc)
        ipsi = ipsi_tc[ipsi_pref_ix]
        contra = contra_tc[contra_pref_ix]
    elif method == 1:
        _, ipsi_pref_ix = preferreddirection(ipsi_tc)
        _, contra_pref_ix = preferreddirection(contra_tc)
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

def resultant( tuningcurve, resultant_type, angles=None ):
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

def halfwidthhalfmax( tuningcurve ):
    """ Returns the half width of the tuning curve at half height. Assumes equal sampling across angles.
        - Inputs -
        tuningcurve: 1D array of neuronal responses per stimulus
        returns halfwidthhalfmax in degrees (np.float)
    """
    # Get number of angles
    n_angles = tuningcurve.shape[0]

    # Calculate angles
    angles = np.arange(0,360,360/n_angles)

    # Find index of the maximum value of the tuning curve
    pref_ix = np.argmax(tuningcurve)

    # Get half-maximum value
    halfmax = tuningcurve[pref_ix]/2

    # Shift tuning curve to have only right-side values (including peak)
    tcr = tuningcurve[ np.mod( np.arange(0,n_angles,1)+pref_ix, n_angles ) ]

    # Find the first index below halfmax
    right_edge = np.argmax(tcr<halfmax)

    # Interpolate the exact width between the right_edge and right_edge-1 point
    exact_w = (tcr[right_edge-1]-halfmax) / (tcr[right_edge-1]-tcr[right_edge])

    # Calculate right-handed width in degrees
    width_right = (360/n_angles) * ((right_edge-1)+exact_w)

    # Flip tuning curve to have only left-side values (including peak)
    tcl = tuningcurve[ np.mod( (np.arange(0,n_angles,1)+pref_ix)[::-1]+1, n_angles ) ]

    # Find the first index below halfmax
    left_edge = np.argmax(tcl<halfmax)

    # Interpolate the exact width between the left_edge and left_edge-1 point
    exact_w = (tcl[left_edge-1]-halfmax) / (tcl[left_edge-1]-tcl[left_edge])

    # Calculate left-handed width in degrees
    width_left = (360/n_angles) * ((left_edge-1)+exact_w)

    # Return bandwidth value
    return (width_right + width_left) / 2

def twopeakgaussianfit( tuningcurve, angles ):
    """ Returns an array of size (360,) with the fitted tuning curve at 1 degree resolution
        - Inputs -
        tuningcurve: 1D array of neuronal responses per stimulus
        angles:      Array with angles (equal sampling across 360 degrees)
        returns fitted tuning curve (array of np.float)
    """

    # Function that wraps the xvalues in the Gaussian function to 0-180 degrees
    def wrap_x(x):
        return np.abs(np.abs(np.mod(x,360)-180)-180)

    # Function that returns the two peaked Gaussian
    def twopeakgaussian(x, Rbaseline, Rpref, Rnull, thetapref, sigma):
        return Rbaseline + Rpref*np.exp(-wrap_x(x-thetapref)**2/(2*sigma**2)) + Rnull*np.exp(-wrap_x(x+180-thetapref)**2/(2*sigma**2))

    # Calculate angles if not supplied
    if angles is None:
        angles = np.arange(0,360,360/tuningcurve.shape[0])

    # Get x-value range to consider
    x_values = np.arange(0,360,1)

    # -- Estimate parameters --

    # Baseline level of tuning curve
    Rbaseline = np.min(tuningcurve)

    # Preferred direction
    thetapref,pref_ix = preferreddirection(tuningcurve,angles)

    # Response amplitude to preferred direction
    Rpref = tuningcurve[pref_ix]

    # Response amplitude to null direction
    Rnull = tuningcurve[np.mod(pref_ix+int(angles.shape[0]/2),angles.shape[0])]

    # Estimate of tuning curve width
    sigma = halfwidthhalfmax(tuningcurve)

    # Merge all parameters in a tuple
    param_estimate = (Rbaseline, Rpref, Rnull, thetapref, sigma),

    # Fit parameters
    fitted_params,pcov = optimize.curve_fit( twopeakgaussian, angles, tuningcurve, p0=param_estimate)

    # Return fitted curve
    return twopeakgaussian(x_values,*fitted_params)
