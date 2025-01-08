#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This module holds functions that support suite 2p processed data to be integrated with image stacks

Requires suite2p

Created on Thu May 7, 2020

@author: pgoltstein
"""

#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Imports
import os.path, glob
import numpy as np
from suite2p.registration import rigid, nonrigid, bidiphase


#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Functions

def load_suite2p_ops( filepath ):
    """ Load the multiplane ops file
    """

    # Find planes
    ops = []
    plane_query = os.path.join( filepath, "suite2p", "plane*" )
    plane_folders = glob.glob(plane_query)
    for plane_folder in plane_folders:
        opsfile = os.path.join( plane_folder, "ops.npy" )
        ops_plane = np.load( opsfile, allow_pickle=True)

        # Dealing with zero-length numpy array and adding to ops
        if not ops_plane.shape:
            ops.append( ops_plane[()] )
        else:
            ops.append( ops_plane[0] )

    # Convert to numpy array to be compatible with suite2p
    ops = np.array(ops)
    return ops


def shift_imagedata( imagedata, plane_no, frames, suite2p_ops ):
    """ Realignes image data to parameters in the ops dictionary
    """

    # Display ops
    # print('Displaying ops')
    # for k,v in suite2p_ops[plane_no].items():
    #     print("{}: {}".format(k,v))

    # Get parameters
    bidiphase_par = int(suite2p_ops[plane_no]['bidiphase'])
    xmax = suite2p_ops[plane_no]['xoff'][frames].astype(np.int)
    ymax = suite2p_ops[plane_no]['yoff'][frames].astype(np.int)
    if suite2p_ops[plane_no]['nonrigid']:
        nblocks = suite2p_ops[plane_no]['nblocks']
        xblock = suite2p_ops[plane_no]['xblock']
        yblock = suite2p_ops[plane_no]['yblock']
        xmax1 = suite2p_ops[plane_no]['xoff1'][frames,:]
        ymax1 = suite2p_ops[plane_no]['yoff1'][frames,:]

    # for suite2p, data : int16 or float32, 3D array (size [nimg x Ly x Lx])
    imagedata = imagedata.transpose(2,0,1)

    # old suite2p
    # imagedata = register.apply_shifts(imagedata, suite2p_ops[plane_no], ymax, xmax, ymax1, xmax1)

    # Correct phase shift
    if bidiphase_par != 0:
        bidiphase.shift(imagedata, bidiphase_par)

    # New suite2p -> rigid registration step
    for frame_no, (dy, dx) in enumerate(zip(ymax, xmax)):
        imagedata[frame_no,:,:] = rigid.shift_frame( frame=imagedata[frame_no,:,:], dy=dy, dx=dx )

    # New suite2p -> non-rigid registration step
    if suite2p_ops[plane_no]['nonrigid']:
        imagedata = nonrigid.transform_data(
            data=imagedata,
            nblocks=nblocks,
            xblock=xblock,
            yblock=yblock,
            ymax1=ymax1,
            xmax1=xmax1,
        )

    return imagedata.transpose(1,2,0)
