#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Script to test analysis of suite2p processed data.

Created on Tue Jan 28, 2020

@author: pgoltstein
"""

import sys, os, glob
import numpy as np

# Local imports
sys.path.append('../xx_analysissupport')
import auxrecorder
import scanimagestack

class Suite2pData():
    """ Loads and represents the preprocessed suite2p data.
    """

    def __init__(self, datapath=".", imagefilestem="", imagefileextention="tif"):
        """ - filepath: Path to where the image stack is located, will find the suite2p files from there.
        """
        super(Suite2pData, self).__init__()

        # Find image stack
        # self._ImStack = si_stack.XYT(filestem=imagefilestem, filepath=self._datapath, extention=imagefileextention)

        # Find suite2p data folder
        self._datapath = datapath
        self._s2proot = os.path.join(self._datapath,"suite2p")

        # Find planes
        self._plane_paths = sorted(glob.glob( os.path.join( self._s2proot, 'plane*' ) ))
        self._nplanes = len(self._plane_paths)

        # Load options file -> do this now in a plane loop
        # filename = os.path.join(self._s2proot,'ops1.npy')
        # self._ops = np.load(filename,allow_pickle=True)
        self._ops = []
        for plane in range(self._nplanes):
            filename = os.path.join(self._plane_paths[plane],'ops.npy')
            self._ops.append( np.load(filename,allow_pickle=True).item() )

        # Load ROI info
        self._roiinfo = []
        self._nrois_total = []
        self._iscell = []
        for plane in range(self._nplanes):
            filename = os.path.join(self._plane_paths[plane],'stat.npy')
            self._roiinfo.append( np.load(filename,allow_pickle=True) )
            self._nrois_total.append(len(self._roiinfo[plane]))
            filename = os.path.join(self._plane_paths[plane],'iscell.npy')
            self._iscell.append( np.load(filename,allow_pickle=True) )

        # Process basic ROI info
        self._x = []
        self._y = []
        self._npixels = []
        self._radius = []
        self._aspect_ratio_cell = []
        self._compact = []
        self._footprint = []
        self._selected_id = []
        for plane in range(self._nplanes):
            self._x.append(np.zeros((self._nrois_total[plane],), dtype=int))
            self._y.append(np.zeros((self._nrois_total[plane],), dtype=int))
            self._npixels.append(np.zeros((self._nrois_total[plane],), dtype=float))
            self._radius.append(np.zeros((self._nrois_total[plane],), dtype=float))
            self._aspect_ratio_cell.append(np.zeros((self._nrois_total[plane],), dtype=float))
            self._compact.append(np.zeros((self._nrois_total[plane],), dtype=float))
            self._footprint.append(np.zeros((self._nrois_total[plane],), dtype=float))
            self._selected_id.append(np.zeros((self._nrois_total[plane],), dtype=int))
            for roi in range(self._nrois_total[plane]):
                self._y[plane][roi] = self._roiinfo[plane][roi]["med"][0]
                self._x[plane][roi] = self._roiinfo[plane][roi]["med"][1]
                self._npixels[plane][roi] = self._roiinfo[plane][roi]["npix"]
                self._radius[plane][roi] = self._roiinfo[plane][roi]["radius"]
                self._aspect_ratio_cell[plane][roi] = self._roiinfo[plane][roi]["aspect_ratio"]
                self._compact[plane][roi] = self._roiinfo[plane][roi]["compact"]
                self._footprint[plane][roi] = self._roiinfo[plane][roi]["footprint"]
                self._selected_id[plane][roi] = roi

        # Set internal variables
        self._plane = 0
        self._shuffletraces = False
        self._spikes = [None for _ in range(self._nplanes)]
        self._F = [None for _ in range(self._nplanes)]

    # Built-in properties
    def __str__(self):
        """ Returns a printable string with summary output """
        return "Suite2pData of {}\n* # of planes: {}".format( self._datapath, self._nplanes )

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Handling of imaging planes
    @property
    def nplanes(self):
        """ Number of fastz-piezo imaging planes in stack """
        return self._nplanes

    @property
    def plane(self):
        """ Returns the currently selected image plane number """
        return self._plane

    @plane.setter
    def plane(self,plane_nr):
        """ Sets the plane """
        self._plane = int(plane_nr)
        print("Selected plane {}".format(self._plane))

    @property
    def image(self):
        """ Returns the average image of the currently selected plane """
        return np.array(self._ops[self._plane]["meanImg"])

    @property
    def aspect_ratio(self):
        """ Returns the aspect ratio of the image """
        return np.array(self._ops[self._plane]["aspect"])

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Handling of ROI data
    @property
    def neurons(self):
        """ Returns the list of selected neurons for the selected plane, or overall """
        return np.array(self._selected_id[self._plane], dtype=int).ravel()

    @neurons.setter
    def neurons(self, selection):
        """ Manually sets the list of selected neurons """
        self._selected_id[self._plane] = selection.ravel()

    @property
    def nrois(self):
        """ returns the number of ROI's in the selected plane, or the sum across all planes if selected plane is None
        """
        return int(self._selected_id[self._plane].shape[0])

    @property
    def x(self):
        """ Returns the position of rois along the x-axis """
        return np.array(self._x[self._plane][self._selected_id[self._plane]], dtype=int)

    @property
    def y(self):
        """ Returns the position of rois along the y-axis """
        return np.array(self._y[self._plane][self._selected_id[self._plane]], dtype=int)

    @property
    def npixels(self):
        """ Returns npix: number of pixels in ROI (from suite2p stat.npy) """
        return np.array(self._npixels[self._plane][self._selected_id[self._plane]], dtype=float)

    @property
    def radius(self):
        """ Returns radius: estimated radius of cell from 2D Gaussian fit to mask (from suite2p stat.npy) """
        return np.array(self._radius[self._plane][self._selected_id[self._plane]], dtype=float)

    @property
    def aspect_ratio_cell(self):
        """ Returns aspect_ratio: ratio between major and minor axes of a 2D Gaussian fit to mask (from suite2p stat.npy) """
        return np.array(self._aspect_ratio_cell[self._plane][self._selected_id[self._plane]], dtype=float)

    @property
    def compact(self):
        """ Returns compact: how compact the ROI is (1 is a disk, >1 means less compact) """
        return np.array(self._compact[self._plane][self._selected_id[self._plane]], dtype=float)

    @property
    def footprint(self):
        """ Returns footprint: spatial extent of an ROI’s functional signal, including pixels not assigned to the ROI; a threshold of 1/5 of the max is used as a threshold, and the average distance of these pixels from the center is defined as the footprint (from suite2p stat.npy) """
        return np.array(self._footprint[self._plane][self._selected_id[self._plane]], dtype=float)

    def select_neurons(self, selector, threshold=None):
        """ Selects the neurons as indicated by the variable "selector"
            "iscell": that suite2p classified as 'cell'
        """
        if isinstance(selector,str):
            if selector == "iscell":
                self.neurons = np.argwhere(self._iscell[self._plane][:,0]==1)
                print("selecting {} neurons that Suite2p classified as iscell".format(self.neurons.shape[0]))
            if selector == "isnotcell":
                self.neurons = np.argwhere(self._iscell[self._plane][:,0]==0)
                print("selecting {} neurons that Suite2p classified as NOT iscell".format(self.neurons.shape[0]))
            if selector == "iscell_p_larger_than":
                self.neurons = np.argwhere(self._iscell[self._plane][:,1]>threshold)
                print("selecting {} neurons that Suite2p classified as iscell with a probability > {}".format(self.neurons.shape[0], threshold))
            if selector == "iscell_p_smaller_than":
                self.neurons = np.argwhere(self._iscell[self._plane][:,1]<threshold)
                print("selecting {} neurons that Suite2p classified as iscell with a probability < {}".format(self.neurons.shape[0], threshold))

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Handling of shuffling settings
    @property
    def shuffletraces(self):
        """ If set to True, the F/spike trace of each cell will be individually temporally shuffled """
        return self._shuffletraces

    @shuffletraces.setter
    def shuffletraces(self, shuffle_setting):
        """ Sets value of shuffle variable True/False """
        self._shuffletraces = shuffle_setting

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Data access functions
    @property
    def spikes(self):
        """ Returns the [ROI x Spiketrain] matrix """

        # Load the data of the requested plane and make a copy to return
        if self._spikes[self._plane] is None:
            print("Loading spike matrix for plane {}".format(self._plane))
            filename = os.path.join(self._plane_paths[self._plane],'spks.npy')
            self._spikes[self._plane] = np.load(filename)
        spike_matrix = np.array(self._spikes[self._plane][ self._selected_id[self._plane], : ])

        # Shuffle spike matrix if flag was set
        if self.shuffletraces is True:
            print("Warning: Spike data is shuffled (per neuron) in time domain")
            for ix in range(spike_matrix.shape[0]):
                np.random.shuffle(spike_matrix[ix,:])

        return spike_matrix

    @property
    def F(self):
        """ Returns the [ROI x Fluorescence] matrix """

        # Load the data of the requested plane and make a copy to return
        if self._F[self._plane] is None:
            print("Loading F matrix for plane {}".format(self._plane))
            filename = os.path.join(self._plane_paths[self._plane],'F.npy')
            self._F[self._plane] = np.load(filename)
        F_matrix = np.array(self._F[self._plane][ self._selected_id[self._plane], : ])

        # Shuffle F matrix if flag was set
        if self.shuffletraces is True:
            print("Warning: F data is shuffled (per neuron) in time domain")
            for ix in range(F_matrix.shape[0]):
                np.random.shuffle(F_matrix[ix,:])

        return F_matrix
