#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This script loads data of a single imaging volume, and makes scatter plots showing the retinotopic preference of each cell

Created on Monday 16 Dec 2024

python make-retinotopy-scatter-maps.py O03

@author: pgoltstein
"""


# Imports
import sys, os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

parser = argparse.ArgumentParser( description = "This script loads data of a single imaging volume, and makes scatter plots showing the retinotopic preference of each cell.\n (written by Pieter Goltstein - Dec 2024)")
parser.add_argument('mouse', type=str, help= 'name of the mouse to analyze')
args = parser.parse_args()


#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Settings

# Path settings
settingspath = "../settings"
savepath = "../../figureout"
datapath = os.path.join("../../data/part1-planedata-retinotopy-layer4")
print(f"{datapath=}")
if args.mouse == "O03":
    figname = "Fig-1f-"
elif int(args.mouse[1:]) < 20:
    figname = "Fig-S1d-"
else:
    figname = "Fig-S7f-"

# Data
start_depth=370
depth_increment=20
skip_first_plane=False
include_sign = 0.05
exclude_double_xy=3 # micron
exclude_double_z=depth_increment+5 # micron

# Experiment type specific settings
if int(args.mouse[1:]) < 20:
    convert_to_micron_x = 1192/1024
    convert_to_micron_y = 1019/1024
else:
    convert_to_micron_x = 1180/1024
    convert_to_micron_y = 982/1024


#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Main

print("Loading imaging volume of mouse {}".format(args.mouse))
print("  << {} >>".format(datapath))
volume,parameter_names,aspect_ratio,_,_ = singlevolumeodfunctions.load_volume( datapath, args.mouse, start_depth=start_depth, depth_increment=depth_increment, skip_first_plane=skip_first_plane, convert_to_micron_x=convert_to_micron_x, convert_to_micron_y=convert_to_micron_y, include_sign=include_sign, exclude_double_xy=exclude_double_xy, exclude_double_z=exclude_double_z, invert_odi_values=False )
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
fig,ax = plottingtools.init_figure(fig_size=(12*aspect_ratio,12))
singlevolumeodfunctions.show_param_2d( ax, XY[:,0], XY[:,1], AZI, "Azimuth map", name=None, cmap="hsv", vmin=0, vmax=5, d1=370, d2=430 )
savefile = os.path.join( savepath, figname+"{}-azimuth-scatter-map".format(args.mouse) )
print("Saving azimuth map to file: {}".format(savefile+"pdf"))
plottingtools.finish_figure( filename=savefile, wspace=0.2, hspace=0.2 )

# Shows the basic map with neurons colored by preferred direction
fig,ax = plottingtools.init_figure(fig_size=(12*aspect_ratio,12))
singlevolumeodfunctions.show_param_2d( ax, XY[:,0], XY[:,1], ELE, "Elevation map", name=None, cmap="hsv", vmin=0, vmax=3, d1=370, d2=430 )
savefile = os.path.join( savepath, figname+"{}-elevation-scatter-map".format(args.mouse) )
print("Saving elevation map to file: {}".format(savefile+"pdf"))
plottingtools.finish_figure( filename=savefile, wspace=0.2, hspace=0.2 )


#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# That's all folks !!
print("\nDone.\n")
