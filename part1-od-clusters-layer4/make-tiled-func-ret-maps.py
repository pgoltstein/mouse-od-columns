#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This script loads data of a single tiled-FOV imaging volume and displays maps for retinotopy

python make-tiled-func-ret-maps.py O10

Created on Monday 2 May 2022

@author: pgoltstein
"""

# Imports
import sys, os
import numpy as np

# Local imports
sys.path.append('../xx_analysissupport')
import plottingtools
import singlevolumeodfunctions

# Module settings
plottingtools.font_size = { "title": 6, "label": 5, "tick": 5, "text": 5, "legend": 5 }

# Arguments
import argparse


#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Arguments

parser = argparse.ArgumentParser( description = "This script loads data of a single tiled-FOV imaging volume and displays maps for retinotopy.\n (written by Pieter Goltstein - Oct 2021)")
parser.add_argument('mousename', type=str, help='name of the mouse to analyze')
args = parser.parse_args()


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Settings

# Path settings
settingspath = "../settings"
savepath = "../../figureout"
datapath = os.path.join("../../data/part1-tileddata-ret-layer4")
print(f"{datapath=}")

# Data
start_depth=370
depth_increment=20
skip_first_plane=False
include_sign = 0.05
exclude_double_xy=3 # micron
exclude_double_z=depth_increment+5 # micron
iso_odi_contour_range = [0, 0.2]
iso_odi_contour_linestyle = ["-", "--"]

# Experiment type specific settings
convert_to_micron_x = 1192/1024
convert_to_micron_y = 1019/1024
invert_odi_values = False


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Load data

print("Loading imaging volume of mouse {}".format(args.mousename))
print("  << {} >>".format(datapath))
volume,parameter_names,aspect_ratio,_,_ = singlevolumeodfunctions.load_volume( datapath, args.mousename, start_depth=start_depth, depth_increment=depth_increment, skip_first_plane=skip_first_plane, convert_to_micron_x=convert_to_micron_x, convert_to_micron_y=convert_to_micron_y, include_sign=include_sign, exclude_double_xy=exclude_double_xy, exclude_double_z=exclude_double_z, invert_odi_values=invert_odi_values, include_fovpos=True )
print("Loaded parameters:")
for nr,name in enumerate(parameter_names):
    print("{:>2d}: {}".format(nr,name))

# Get data that should be displayed
XY = volume[:, [parameter_names.index("x"),parameter_names.index("y")]]
AZI = volume[:,parameter_names.index("Pref azim")]
ELE = volume[:,parameter_names.index("Pref elev")]

# Azimuth colors: 0=red, 1=yellow, 2=green, 3=blue, 4=purple
azimuth_values = [-48, -24, 0, 24, 48]
elevation_values = [24, 0, -24]
print("Azimuths: {}".format(np.unique(AZI)))
print("Elevations: {}".format(np.unique(ELE)))

# Shows the basic map with neurons colored by preferred azimuth
fig,ax = plottingtools.init_figure(fig_size=(8*aspect_ratio,8))
singlevolumeodfunctions.show_param_2d( ax, XY[:,0], XY[:,1], AZI, "Azimuth map", name=None, cmap="hsv", vmin=0, vmax=5, d1=370, d2=430, size=1 )
savefile = os.path.join( savepath, "Fig-S2b-tiled-cellmaps-azimuth-{}".format(args.mousename) )
print("Saving azimuth map to file: {}".format(savefile+".pdf"))
plottingtools.finish_figure( filename=savefile )

# Shows the basic map with neurons colored by preferred direction
fig,ax = plottingtools.init_figure(fig_size=(8*aspect_ratio,8))
singlevolumeodfunctions.show_param_2d( ax, XY[:,0], XY[:,1], ELE, "Elevation map", name=None, cmap="hsv", vmin=0, vmax=3, d1=370, d2=430, size=1 )
savefile = os.path.join( savepath, "Fig-S2c-tiled-cellmaps-elevation-{}".format(args.mousename) )
print("Saving elevation map to file: {}".format(savefile+".pdf"))
plottingtools.finish_figure( filename=savefile )


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# That's all folks !!
print("\nDone.\n")
