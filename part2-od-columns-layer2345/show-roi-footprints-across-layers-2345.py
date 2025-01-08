#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This script loads data of the L23-L4-L5 imaging volumes of all mice, finds footprints of ROIs and plots size distribution per layer.

Created on Friday 2 Feb 2024

python show-roi-footprints-across-layers-2345.py

@author: pgoltstein
"""

# Global imports
import sys, os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# Local imports
sys.path.append('../xx_analysissupport')
import plottingtools
import odcfunctions

# Module settings
plottingtools.font_size = { "title": 5, "label": 5, "tick": 5, "text": 5, "legend": 5 }

#_______________________________________________________
# Settings
savepath = "../../figureout"
datapath = os.path.join("../../data/part2-roi-footprints")
print(f"{datapath=}")

# Settings
mice = ["O02", "O03", "O06", "O07", "O09", "O10", "O11", "O12", "O13"]
depths = np.arange(170,530,30)
n_mice = len(mice)
n_depths = len(depths)
n_bins = 20
max_radius = 14
max_npixels = 300
max_aspect = 2.5
depth_cmap = matplotlib.cm.get_cmap('cool', 256)
depth_cmap.set_bad(color="white")

# Load data
df = pd.DataFrame(columns=["Depth","Radius","nPixels","Aspect"])
datafile = np.load(os.path.join(datapath,"roi-footprints-per-depth.npz"), allow_pickle=True)
radii = datafile["radii"]
npixels = datafile["npixels"]
aspect = datafile["aspect"]
df_data = datafile["df"]
df = pd.DataFrame(columns=["Depth","Radius","nPixels","Aspect"])
for i in range(df_data.shape[0]):
    df.loc[len(df)] = {"Depth": df_data[i,0], "Radius": df_data[i,1], 
                        "nPixels": df_data[i,2], "Aspect": df_data[i,3]}
x_radii = datafile["x_radii"]
x_npix = datafile["x_npix"]
x_aspect = datafile["x_aspect"]
depths = datafile["depths"]


# -----------------------------
# Plot histogram and mean

fig = plottingtools.init_fig(fig_size=(10,12))


#################################################################
# RADIUS
#################################################################

ax = plt.subplot(3,2,1)
for d_nr,depth in enumerate(depths):
    y = np.nanmean(radii[d_nr,:,:],axis=1)
    plt.plot(x_radii, y, marker="o", linestyle='-', color=depth_cmap(d_nr/n_depths), markersize=2, markeredgecolor=depth_cmap(d_nr/n_depths), markerfacecolor="None", linewidth=1)
plottingtools.finish_panel( ax, title="Size of ROI's", ylabel="Fraction", xlabel="Radius", legend="off", y_minmax=[0,1], y_step=[0.2,1], y_margin=0, y_axis_margin=0, x_minmax=[0,max_radius], x_margin=0.55, x_axis_margin=0.4, x_ticks=np.arange(0,max_radius,2), x_ticklabels=np.arange(0,max_radius,2), x_tick_rotation=0 )

ax = plt.subplot(3,2,2)
sns.swarmplot( data=df , x="Depth", y="Radius",
               linewidth=1, edgecolor=None, size=1, ax=ax )
odcfunctions.redraw_markers( ax, ["o",]*n_depths, ["#000000",]*n_depths, size=2, reduce_x_width=1 )
for d_nr,depth in enumerate(depths):
    y = np.nanmean(df["Radius"][df["Depth"]==depth])
    plt.bar( d_nr, y, 0.7, color='None', edgecolor=depth_cmap(d_nr/n_depths), linewidth=1 )
plottingtools.finish_panel( ax, title="Size of ROI's", ylabel="Radius", xlabel="Depth", legend="off", y_minmax=[0,10], y_step=[2,0], y_margin=0, y_axis_margin=0, x_minmax=[0,n_depths-1], x_margin=0.55, x_axis_margin=0.4, x_ticks=np.arange(n_depths), x_ticklabels=depths, x_tick_rotation=0 )


#################################################################
# N-PIXELS
#################################################################

ax = plt.subplot(3,2,3)
for d_nr,depth in enumerate(depths):
    y = np.nanmean(npixels[d_nr,:,:],axis=1)
    plt.plot(x_npix, y, marker="o", linestyle='-', color=depth_cmap(d_nr/n_depths), markersize=2, markeredgecolor=depth_cmap(d_nr/n_depths), markerfacecolor="None", linewidth=1)
plottingtools.finish_panel( ax, title="Size of ROI's", ylabel="Fraction", xlabel="nPixels", legend="off", y_minmax=[0,1], y_step=[0.2,1], y_margin=0, y_axis_margin=0, x_minmax=[0,max_npixels], x_margin=0.55, x_axis_margin=0.4, x_ticks=np.arange(0,max_npixels,50), x_ticklabels=np.arange(0,max_npixels,50), x_tick_rotation=0 )

ax = plt.subplot(3,2,4)
sns.swarmplot( data=df , x="Depth", y="nPixels", linewidth=1, edgecolor=None, size=1, ax=ax )
odcfunctions.redraw_markers( ax, ["o",]*n_depths, ["#000000",]*n_depths, size=2, reduce_x_width=1 )
for d_nr,depth in enumerate(depths):
    y = np.nanmean(df["nPixels"][df["Depth"]==depth])
    plt.bar( d_nr, y, 0.7, color='None', edgecolor=depth_cmap(d_nr/n_depths), linewidth=1 )
plottingtools.finish_panel( ax, title="Size of ROI's", ylabel="nPixels", xlabel="Depth", legend="off", y_minmax=[0,200], y_step=[50,0], y_margin=0, y_axis_margin=0, x_minmax=[0,n_depths-1], x_margin=0.55, x_axis_margin=0.4, x_ticks=np.arange(n_depths), x_ticklabels=depths, x_tick_rotation=0 )


#################################################################
# ASPECT
#################################################################

ax = plt.subplot(3,2,5)
for d_nr,depth in enumerate(depths):
    y = np.nanmean(aspect[d_nr,:,:],axis=1)
    plt.plot(x_aspect, y, marker="o", linestyle='-', color=depth_cmap(d_nr/n_depths), markersize=2, markeredgecolor=depth_cmap(d_nr/n_depths), markerfacecolor="None", linewidth=1)
plottingtools.finish_panel( ax, title="Size of ROI's", ylabel="Fraction", xlabel="Aspect", legend="off", y_minmax=[0,1], y_step=[0.2,1], y_margin=0, y_axis_margin=0, x_minmax=[0,max_aspect], x_margin=0.55, x_axis_margin=0.4, x_ticks=np.arange(0,max_aspect,0.5), x_ticklabels=np.arange(0,max_aspect,0.5), x_tick_rotation=0 )

ax = plt.subplot(3,2,6)
sns.swarmplot( data=df , x="Depth", y="Aspect", linewidth=1, edgecolor=None, size=1, ax=ax )
odcfunctions.redraw_markers( ax, ["o",]*n_depths, ["#000000",]*n_depths, size=2, reduce_x_width=1 )

for d_nr,depth in enumerate(depths):
    y = np.nanmean(df["Aspect"][df["Depth"]==depth])
    plt.bar( d_nr, y, 0.7, color='None', edgecolor=depth_cmap(d_nr/n_depths), linewidth=1 )

plottingtools.finish_panel( ax, title="Size of ROI's", ylabel="Aspect", xlabel="Depth", legend="off", y_minmax=[0,1.5], y_step=[0.25,2], y_margin=0, y_axis_margin=0, x_minmax=[0,n_depths-1], x_margin=0.55, x_axis_margin=0.4, x_ticks=np.arange(n_depths), x_ticklabels=depths, x_tick_rotation=0 )

savename = os.path.join( savepath, "Fig-S14-roi-footprints-per-depth" )
plottingtools.finish_figure( filename=savename )

# -----------------------------
# Done

print("\nDone.. that's all folks!")
