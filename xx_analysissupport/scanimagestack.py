#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This module holds functions to import a scanimage tiff stack that consist of multiple tiff files, and gather the meta info stored in the headers. With thanks to Tobias Rose for some of the regular expressions and others that worked on the various dependencies.

Requires ScanImageTiffReader and suite2p
https://vidriotech.gitlab.io/scanimagetiffreader-python/

Created on Thu Jan 30, 2020

@author: pgoltstein
"""

#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Imports

import os, glob
import re
import numpy as np
from ScanImageTiffReader import ScanImageTiffReader
from tqdm import tqdm
import argparse


#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Functions

def parseheader(header):
    """ This function reads the most relevant information from the tiff header of a scanimage tiff.
        Inputs
        - header: The can be obtained from ScanImageTiffReader, header (tifffile.description(0))
        Returns
        - si_info: A dictionary holding the scanimage named variables in python format (int, float, list[floats])
    """

    # Define the regular expressions needed to extract the information
    rx_dict = {
        "stackNumSlices": re.compile(r'stackNumSlices = (?P<stackNumSlices>\d+)'),
        "scanZoomFactor": re.compile(r'scanZoomFactor = (?P<scanZoomFactor>\d+.?\d*)'),
        "scanFrameRate": re.compile(r'scanFrameRate = (?P<scanFrameRate>\d+.?\d*)'),
        "channelsSave": re.compile(r'channelsSave = (?P<channelsSave>.*)'),
        "fastZNumVolumes": re.compile(r'fastZNumVolumes = (?P<fastZNumVolumes>\d+)'),
        "acqNumFrames": re.compile(r'acqNumFrames = (?P<acqNumFrames>\d+)'),
        "fastZEnable": re.compile(r'fastZEnable = (?P<fastZEnable>\d+)'),
        "stackZStepSize": re.compile(r'stackZStepSize = (?P<stackZStepSize>\d+)'),
        "triggerClockTimeFirst": re.compile(r'triggerClockTimeFirst = (?P<triggerClockTimeFirst>\'\d+-\d+-\d+ \d+:\d+:\d+.\d+\')'),
        "loggingFramesPerFile": re.compile(r'loggingFramesPerFile = (?P<loggingFramesPerFile>\d+)'),
        "beamPowers": re.compile(r'beamPowers = (?P<beamPowers>\d+.?\d*)'),
        "loggingFileStem": re.compile(r'loggingFileStem = (?P<loggingFileStem>\'.+\')'),
        "motorPosition": re.compile(r'motorPosition = (?P<motorPosition>\[.+\])'),
        "pmtGain": re.compile(r'pmtGain = (?P<pmtGain>\[.+\])'),
        "scanLinesPerFrame": re.compile(r'scanLinesPerFrame = (?P<scanLinesPerFrame>\d+)'),
        "scanPixelsPerLine": re.compile(r'scanPixelsPerLine = (?P<scanPixelsPerLine>\d+)'),
        "stackZEndPos": re.compile(r'stackZEndPos = (?P<stackZEndPos>\d+.?\d*)'),
        "stackZStartPos": re.compile(r'stackZStartPos = (?P<stackZStartPos>\d+.?\d*)'),
        "stackZStepSize": re.compile(r'stackZStepSize = (?P<stackZStepSize>\d+.?\d*)'),
    }

    # Now step through the dictionary and extract the information as int, float, string or a list of floats
    si_info = {}
    for key, rx in rx_dict.items():
        # Find match using reg-ex
        match = rx_dict[key].search(header)
        if match:

            # floating point numbers
            if key in ["scanZoomFactor", "scanFrameRate", "beamPowers", "stackZEndPos", "stackZStartPos", "stackZStepSize"]:
                si_info[key] = float(match.group(key))

            # strings
            elif key in ["triggerClockTimeFirst","loggingFileStem"]:
                si_info[key] = str(match.group(key)).strip('\'')

            # list of floats
            elif key in ["motorPosition","pmtGain"]:
                positions = str(match.group(key)).strip('[]')
                si_info[key] = []
                for pos in positions.split(' '):
                    si_info[key].append(float(pos))

            # channelsSave can be both int or list of ints, return as list always
            elif key in ["channelsSave",]:
                if match.group(key).isdigit():
                    si_info[key] = [int(match.group(key)),]
                else:
                    chan_list = match.group(key).strip("[]").split(";")
                    chan_list = [int(ch) for ch in chan_list]
                    si_info[key] = chan_list

            # otherwise integer
            else:
                si_info[key] = int(match.group(key))
        # if no reasonable match found, assign None
        else:
            si_info[key] = None

    # Return the dict
    return si_info


#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Classes

class XYT(object):
    """ This class represents an entire (multi-tiff) scanimage timeseries stack, either a single plane, or multi-plane aquired using fast z controls. Image channel and image plane should be set manually (defaults are 0).

        The class can load the image data using standard np.ndarray indexing:
         * data = imagestack[:] returns all the data
         * data = imagestack[1] returns the second frame (zero based slice)
         * data = imagestack[[5,8,10]] returns frames 5,8 and 10
         * data = imagestack[::2] returns every second frame.

         In addition, the class provides access to the meta data as properties. For instance:
         * res = XYT.resolution returns the [y,x] image resolution
         * nchannels = XYT.nchannels returns number of image channels
    """

    def __init__(self, filestem='', filepath='.', extention="tif", imagesettingsfile=None, do_reg = False, imregfunc=None, imregparams=[], verbose=False):
        """ Initializes the image stack and gathers the meta data
            Inputs
            - filestem: Part of the file name that is shared among all tiffs belonging to the stack (optional, if left out all tiffs in filepath will be included)
            - filepath: Full directory path to the tiffs
            - extention: file extention of the stack
            - imagesettingsfile: manually stored image settings
            - do_reg: Whether or not to perform registration on the images
            - imregfunc: Function to use for image registration
            - imregparams: List of parameters to supply to imregfunc
            - verbose: print warnings
        """
        super(XYT, self).__init__()

        # Set the filepath
        self._filepath = filepath
        self._extention = extention
        self._verbose = verbose

        # Find the tiff files
        self._block_files = sorted( glob.glob( os.path.join( self._filepath, filestem+'*.'+extention ) ) )
        self._nblocks = len(self._block_files)

        # Load and parse the header
        with ScanImageTiffReader( self._block_files[0] ) as tifffile:
            header = (tifffile.description(0))
            self.si_info = parseheader(header)
        self._nframesperblock = self.si_info["loggingFramesPerFile"] * self.nchannels

        # Get number of frames or volumes depending on whether stack or not
        if self.nplanes > 1:
            self._nframes = int(self.si_info["fastZNumVolumes"])
        else:
            self._nframes = int(self.si_info["acqNumFrames"])

        # correct number of frames if discrepancy detected between header and block files
        n_frames_from_blocks = int((self._nframesperblock * self._nblocks) / (self.nplanes * self.nchannels))
        if n_frames_from_blocks != self.nframes:
            if self._verbose:
                print("#frames/volumes in header ({}); inferred different number from tiff blocks ({})".format( self.nframes, n_frames_from_blocks ))
            self._nframes = n_frames_from_blocks

        # Load default settings and internal variables
        if imagesettingsfile is None:
            self_path = os.path.dirname(os.path.realpath(__file__))
            settings_path = os.path.join( os.path.sep.join(  self_path.split(os.path.sep)[:-1] ), "settings" )
            imagesettingsfile = os.path.join(settings_path,"default.imagesettings.py")
        self._imagesettingsfile = imagesettingsfile
        settings = {}
        with open(imagesettingsfile) as f:
            exec(f.read(), settings)
            self._fovsize_for_zoom = settings["fovsize_for_zoom"]
            self._laserpowers_for_wavelength = settings["laserpowers_for_wavelength"]

        self.register = do_reg
        self._imregfunc = None
        if imregfunc is not None:
            self.imregfunc = imregfunc
        self.imregparams = imregparams

        self._datatype = np.int16
        self.channel = 0
        self.plane = 0

    # properties
    def __str__(self):
        """ Returns a printable string with summary output """
        first_file = self._block_files[0].split(os.path.sep)[-1]
        return "Imagestack of {} {} files, first file: {}\n* Image settings: {}\n* {} frames, {} planes, {} channels, {} x {} pixels".format( self._nblocks, self._extention, first_file, self._imagesettingsfile, self.nframes, self.nplanes, self.nchannels, self.yres, self.xres )


    @property
    def filepath(self):
        """ Path where to find image files """
        return self._filepath

    @property
    def xres(self):
        """ Number of pixels along the x-axis """
        return int(self.si_info["scanPixelsPerLine"])

    @property
    def yres(self):
        """ Number of pixels along the y-axis """
        return int(self.si_info["scanLinesPerFrame"])

    @property
    def resolution(self):
        """ Number of pixels along the y- and x-axis """
        return int(self.si_info["scanLinesPerFrame"]), int(self.si_info["scanPixelsPerLine"])

    @property
    def nframes(self):
        """ Number of frames per slice and channel """
        return self._nframes

    @property
    def nplanes(self):
        """ Number of planes """
        if self.si_info["stackNumSlices"] > 0:
            return int(self.si_info["stackNumSlices"])
        else:
            return int(1)

    @property
    def nchannels(self):
        """ Number of channels """
        return len(self.si_info["channelsSave"])

    @property
    def zoom(self):
        """ Imaging zoom factor """
        return float(self.si_info["scanZoomFactor"])

    @property
    def zstep(self):
        """ Piezo step along the z-axis. Returns 0 if no (fast) z volume) """
        if self.nplanes > 1:
            return float(self.si_info["stackZStepSize"])
        else:
            return 0

    @property
    def rawposition(self):
        """ the raw position parameters of the microscope """
        return self.si_info["motorPosition"]

    @property
    def x(self):
        """ the x position of the microscope fov """
        return self.si_info["motorPosition"][0]

    @property
    def y(self):
        """ the y position of the microscope fov """
        return self.si_info["motorPosition"][1]

    @property
    def z(self):
        """ the z position of the microscope fov, includes the piezo plane """
        z = self.si_info["motorPosition"][2]
        piezo = self.plane * self.zstep
        return z - piezo

    @property
    def z_base(self):
        """ the base z position of the microscope fov, in cm?? """
        return self.si_info["motorPosition"][3]

    @property
    def fovsize(self):
        """ Size of the field of view in micron"""
        if self.zoom in self._fovsize_for_zoom.keys():
            x = self._fovsize_for_zoom[self.zoom]["x"]
            y = self._fovsize_for_zoom[self.zoom]["y"]
            # Correct for piezo step tilting y plane
            if self.nplanes > 1:
                plane_angle = np.arctan( self.si_info["stackZStepSize"] / y )
                y = y / np.cos(plane_angle)
            return { "x": x, "y": y }
        else:
            print("FOV has not been calibrated for zoom {}x, returning np.NaNs".format(self.zoom))
            return { "x": np.NaN, "y": np.NaN }

    @property
    def pixelsize(self):
        """ Size of a single pixel in micron """
        if self.zoom in self._fovsize_for_zoom.keys():
            x = self._fovsize_for_zoom[self.zoom]["x"]
            y = self._fovsize_for_zoom[self.zoom]["y"]
        else:
            # Try and interpolate from known values
            # --> still needs to be implemented
            print("FOV has not been calibrated for zoom {}x, returning np.NaNs".format(self.zoom))
            return { "x": np.NaN, "y": np.NaN }
        # Correct for piezo step tilting y plane
        if self.nplanes > 1:
            plane_angle = np.arctan( self.si_info["stackZStepSize"] / y )
            y = y / np.cos(plane_angle)
        return { "x": x/self.xres, "y": y/self.yres }

    @property
    def laserpower(self):
        """ Laser power in percentage """
        return float(self.si_info["beamPowers"])

    def get_laserpower(self, wavelength):
        """ Laser power for the used wavelength """
        if wavelength not in self._laserpowers_for_wavelength.keys():
            print("Laserpower has not been calibrated for wavelength {}nm, returning np.NaN".format(wavelength))
            return np.NaN
        if self.laserpower not in self._laserpowers_for_wavelength[wavelength].keys():
            print("Laserpower at wavelength {} nm, has not been calibrated for value {}%, returning np.NaN".format( wavelength, self.laserpower ))
            return np.NaN
        return float( self._laserpowers_for_wavelength[wavelength][self.laserpower] )

    @property
    def verbose(self):
        """ Returns the verbose flag """
        return self._verbose

    @verbose.setter
    def verbose(self,verbose_bool):
        """ Sets the verbose flag """
        self._verbose = bool(verbose_bool)

    @property
    def channel(self):
        """ Returns the currently selected channel """
        return self._channel

    @channel.setter
    def channel(self,chan_nr):
        """ Sets the channel """
        self._channel = int(chan_nr)

    @property
    def plane(self):
        """ Returns the currently selected image plane, zero-based """
        return self._plane

    @plane.setter
    def plane(self,plane_nr):
        """ Sets the plane, zero-based """
        if plane_nr < self.nplanes:
            self._plane = int(plane_nr)
        else:
            print("!!! Warning, tried to set plane to {} (zero-based), but stack has only {} planes !!!\nCurrent plane remains set to {} ".format(plane_nr,self.nplanes,self._plane))

    @property
    def register(self):
        """ Returns whether or not to register the image data """
        return self._do_register

    @register.setter
    def register(self,do_register):
        """ Tries to load/initialize registration data from suite2p folders when set to true """
        if do_register:
            if not hasattr(self._imregfunc, '__call__'):
                print("!!! Cannot register images because no image-registration function has been set. !!!")
                self._do_register = False
                return
        self._do_register = do_register

    @property
    def imregparams(self):
        """ Returns the parameters supplied to the image registration function """
        return self._imregparams

    @imregparams.setter
    def imregparams(self,imregparams):
        """ Sets the parameters supplied to the image registration function """
        if not isinstance(imregparams,list):
            imregparams = [imregparams,]
        self._imregparams = imregparams

    @property
    def imregfunc(self):
        """ Returns the function that will perform the image registration """
        return self._imregfunc

    @imregfunc.setter
    def imregfunc(self,imregfunc):
        """ Sets the function that will perform the image registration, the function should take three fixed inputs and further params
            1. The image stack
            2. The image plane number
            3. The frames that will be realigned
            4. Params to be set by imregparams
        """
        if not hasattr(imregfunc, '__call__'):
            print("Cannot set image registration function because the supplied function is not 'callable' (i.e. is not a function).")
            return
        self._imregfunc = imregfunc

    # Internal function to load the imaging data using slicing
    def __getitem__(self, indices):
        """ Loads and returns the image data directly from disk """

        # Use the provided slice object to get the requested frames
        n_frames_exceeded = False
        if isinstance(indices, slice):
            start = 0 if indices.start is None else indices.start
            stop = self.nframes if indices.stop is None else indices.stop
            step = 1 if indices.step is None else indices.step
            if start > self.nframes or stop > self.nframes:
                n_frames_exceeded = True
                n_frames_requested = len(range(start,stop,step))
            frames = np.arange(self.nframes)[indices]
        elif isinstance(indices, list) or isinstance(indices, tuple):
            if max(indices) > self.nframes:
                n_frames_exceeded = True
                n_frames_requested = len(indices)
            frames = np.array(indices)
        else:
            if indices > self.nframes:
                n_frames_exceeded = True
                n_frames_requested = 1
            frames = np.array([indices,])

        # Check if the requested frames do not exceed the stack
        if n_frames_exceeded:
            print("!!! Requested frames {}, but stack has only {} frames, returning {} 'zero' frames !!!".format(indices,self.nframes,n_frames_requested))
            return np.zeros((self.yres,self.xres,n_frames_requested),dtype=self._datatype)
            # raise IndexError("Requested frames {}, but stack has only {} frames".format(indices,self.nframes))

        # Define the indices of the requested frames
        # tiffs are stored as [ch0-sl0, ch1-sl0, ch0-sl1, ch2-sl1, ch0-sl2 etc]
        start_frame = (self._plane * self.nchannels) + self._channel
        frame_jump = self.nchannels * self.nplanes
        frame_ixs = start_frame + (frames * frame_jump)
        n_frame_ixs = len(frame_ixs)
        frame_ids = np.arange(n_frame_ixs)

        # Identify the block files to open, and which frames to load
        block_ixs_per_frame = np.floor(frame_ixs / self._nframesperblock).astype(int)
        frame_ixs_in_block = np.mod(frame_ixs, self._nframesperblock)
        block_numbers,block_inverse = np.unique(block_ixs_per_frame, return_inverse=True)
        block_indexes = list(range(len(block_numbers)))

        # Split the list of frames into a per-block list
        frame_ixs_per_block = []
        frame_ids_per_block = []
        for b in block_indexes:
            frame_ixs_per_block.append( list(frame_ixs_in_block[block_inverse==b]) )
            frame_ids_per_block.append( list(frame_ids[block_inverse==b]) )

        # Loop block files, and frame indices to load all requested frames
        imagedata = np.zeros((self.yres,self.xres,n_frame_ixs),dtype=self._datatype)
        if self._verbose:
            with tqdm(total=n_frame_ixs, desc="Reading", unit="Fr") as bar:
                for bnr,bix in zip(block_numbers,block_indexes):
                    with ScanImageTiffReader(self._block_files[bnr]) as tifffile:
                        for ix,id_ in zip( frame_ixs_per_block[bix], frame_ids_per_block[bix] ):
                            imagedata[:,:,id_] = tifffile.data(beg=ix,end=ix+1)
                            bar.update(1)
        else:
            for bnr,bix in zip(block_numbers,block_indexes):
                with ScanImageTiffReader(self._block_files[bnr]) as tifffile:
                    for ix,id_ in zip( frame_ixs_per_block[bix], frame_ids_per_block[bix] ):
                        imagedata[:,:,id_] = tifffile.data(beg=ix,end=ix+1)

        # Register the stack and return
        if self._do_register:
            imagedata = self._imregfunc(imagedata, self._plane, frames, *self._imregparams)

        # Return the stack
        return imagedata
