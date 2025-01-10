#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This script analyses the clusters data and map interactions processed using the geometry method

Created on Tuesday 21 Nov 2023

python show-od-map-relations.py

@author: pgoltstein
"""

# Global imports
import sys, os
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from collections import namedtuple
from scipy.stats import pearsonr

# Local imports
sys.path.append('../xx_analysissupport')
import odcfunctions
import plottingtools
import statstools

# Module settings
plottingtools.font_size = { "title": 5, "label": 5, "tick": 5, "text": 5, "legend": 5 }

# Probably shouldn't do this, but got tired of "mean of empty slice" warnings
import warnings
warnings.filterwarnings('ignore')

# Display options for pandas
def round_to_sign_number( number, n_values=3 ):
    if number == 0:
        return 0
    if ~np.isnan(number) and ~isinstance(number,str):
        order_of_magn = math.floor(math.log10(np.abs(number)))
        return np.round(number, -1 * (order_of_magn-n_values+1))
    else:
        return number
pd.set_option('display.precision',4)
pd.set_option('display.max_columns',40)
pd.set_option('display.float_format', lambda x: '{:10.8g}'.format(round_to_sign_number(x)))


# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Load the 'real' data

datapath = os.path.join("../../data/part3-processeddata-layer4")
datafile = np.load( os.path.join(datapath,"od-clusters-descriptives.npz"), allow_pickle=True)

# Load settings
settings = datafile['settings'].item()
msettings_loaded = datafile['msettings'].item()
msettings = odcfunctions.mousesettings(msettings_loaded.name)

# Load the data into a list with "ODpatterns" Box's
cluster_specs = []
for row in datafile['cluster_descriptives']:
    cluster_specs.append(odcfunctions.ODpatterns( *row ))


# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Put the 'real' data into a data frame

# All the fields to include - matched between Box names and DF names, in correct order (1-to-1 mapping)
include_entries = ["mouse", "no", "sh_nr", "type", "perim", 
                   "length", "len_alo", "width", "wid_alo", "angle", "ang_alo", 
                   "area", "ecc", "odi", "odi_cells", "absodi_cells", "odi_ipsi_cells", "odi_contra_cells", 
                   "azi", "azi_cells", "ele", "ele_cells", "cmf", "cmf_azi", "cmf_ele", "azi_ang", "ele_ang", 
                   "ret_angle", "ret_ratio", "rfsize", "rfsize_cells", 
                   "odi_steepness", "odi_skeleton", "odi_border", 
                   "fpd_all", "fpd_ipsi", "fpd_contra", "cv_all", "cv_ipsi", "cv_contra", 
                   "dsi_all", "dsi_ipsi", "dsi_contra", "Rs_r0", "Mmax_Mmin", "Rs_w_to_iso_odi", "ori_len", "ori_to_od_grad_angle"]

df_columns = ["Mouse", "Nr", "Shuffle nr", "Cluster type", "Perimeter", 
              "Length", "Length (Alonso)", "Width", "Width (Alonso)", "Angle", "Angle (Alonso)", 
              "Area", "Eccentricity", "ODI", "ODI (cells)", "Abs ODI (cells)", "Ipsi ODI (cells)", "Contra ODI (cells)", 
              "Azimuth", "Azimuth (cells)", "Elevation", "Elevation (cells)", "CMF", "CMF (azimuth)", "CMF (elevation)", "Azimuth angle", "Elevation angle", 
              "Retinotopy angle", "Retinotopy ratio", "RF size", "RF size (cells)", 
              "ODI (steepness)", "ODI (skeleton)", "ODI (border)", 
              "fPD (all)", "fPD (ipsi)", "fPD (contra)", "CV (all)", "CV (ipsi)", "CV (contra)", 
              "DSI (all)", "DSI (ipsi)", "DSI (contra)", "Rs_r0", "Mmax_Mmin", "Rs_w to iso-ODI", "fPO (res len)", "OD-fPO angle"]

# Function to load data in a pandas dataframe
def load_dataframe(cluster_specs, include_entries, df_columns, shuffle_nr=np.NaN):
    # I expect to see RuntimeWarnings -- mean of empty slice -- in this block
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
    
        df = pd.DataFrame(columns=df_columns)
        for cnr,cdat in enumerate(cluster_specs):
            cdict = cdat._asdict()
            cdict["sh_nr"] = float(shuffle_nr)
            tmp_dict = {df_columns[ix] : cdict[requested_key] for ix,requested_key in enumerate(include_entries)}
            if len(tmp_dict["Width (Alonso)"]) > 0:
                tmp_dict["Width (Alonso)"] = np.nanmean(tmp_dict["Width (Alonso)"])
            else:
                tmp_dict["Width (Alonso)"] = np.NaN
            if len(tmp_dict["Angle (Alonso)"]) > 0:
                tmp_dict["Angle (Alonso)"] = odcfunctions.circ_mean(tmp_dict["Angle (Alonso)"])
            else:
                tmp_dict["Angle (Alonso)"] = np.NaN
            tmp_dict["RF size (cells)"] = np.nanmean(tmp_dict["RF size (cells)"])
            tmp_dict["fPD (all)"] = odcfunctions.circ_mean(tmp_dict["fPD (all)"])
            tmp_dict["fPD (ipsi)"] = odcfunctions.circ_mean(tmp_dict["fPD (ipsi)"])
            tmp_dict["fPD (contra)"] = odcfunctions.circ_mean(tmp_dict["fPD (contra)"])
            tmp_dict["CV (all)"] = np.nanmean(tmp_dict["CV (all)"])
            tmp_dict["CV (ipsi)"] = np.nanmean(tmp_dict["CV (ipsi)"])
            tmp_dict["CV (contra)"] = np.nanmean(tmp_dict["CV (contra)"])
            tmp_dict["DSI (all)"] = np.nanmean(tmp_dict["DSI (all)"])
            tmp_dict["DSI (ipsi)"] = np.nanmean(tmp_dict["DSI (ipsi)"])
            tmp_dict["DSI (contra)"] = np.nanmean(tmp_dict["DSI (contra)"])
            df.loc[len(df)] = tmp_dict
    return df

# Function to add computed / derivative data into dataframe
def add_computed_variables(df):
    # Add a column which specifies mouse type (gcamp or jrgeco)
    mousetype = []
    for m in df["Mouse"]:
        if int(m[1:]) <= 13:
            mousetype.append(1)
        else:
            mousetype.append(2)
    df["mousetype"] = mousetype

    # Number of ipsi and contra clusters
    nipsiclusters = []
    ncontraclusters = []
    for m in df["Mouse"]:
        n = np.sum(np.logical_and(df["Mouse"]==m,df["Cluster type"]==1)*1.0)
        nipsiclusters.append(n)
        n = np.sum(np.logical_and(df["Mouse"]==m,df["Cluster type"]==2)*1.0)
        ncontraclusters.append(n)
        # print(m,n)
    df["n-ipsi"] = nipsiclusters
    df["n-contra"] = ncontraclusters
    return df

# Now actually put the data into a dataframe
df_od = load_dataframe(cluster_specs, include_entries, df_columns)
df_od = add_computed_variables(df_od)
df_od.info()


# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Load the shuffled data into a data frame

# Shuffled ODI data
cluster_specs_sh_odi = []
df_sh_odi = pd.DataFrame(columns=df_columns)
for sh in range(10):
    filename = os.path.join(datapath,"od-clusters-descriptives-sh-odi-{}.npz".format(sh))
    print("Loading: {}".format(filename))
    datafile_sh = np.load(filename, allow_pickle=True)
    
    # Load the data inro named tuples and then a dataframe
    cluster_specs_sh_odi.append([])
    for row in datafile_sh['cluster_descriptives']:
        row = np.concatenate([row, [0,0]]) # Placeholders for n-ipsi-cells and n-contra-cells which are not included in shuffled data
        cluster_specs_sh_odi[sh].append(odcfunctions.ODpatterns( *row ))
    df_sh_x = load_dataframe(cluster_specs_sh_odi[sh], include_entries, df_columns, shuffle_nr=sh)
    df_sh_x = add_computed_variables(df_sh_x)

    # Add to df
    df_sh_odi = pd.concat( [df_sh_odi, df_sh_x] )

# Set columns to numeric
df_sh_odi["Nr"] = pd.to_numeric(df_sh_odi["Nr"])
df_sh_odi["Length (Alonso)"] = pd.to_numeric(df_sh_odi["Length (Alonso)"])
df_sh_odi["Area"] = pd.to_numeric(df_sh_odi["Area"])
df_sh_odi.info()

# Shuffled Preferred Orientation data
cluster_specs_sh_ori = []
df_sh_ori = pd.DataFrame(columns=df_columns)
for sh in range(10):
    filename = os.path.join(datapath,"od-clusters-descriptives-sh-ori-{}.npz".format(sh))
    print("Loading: {}".format(filename))
    datafile_sh = np.load(filename, allow_pickle=True)
    
    # Load the data inro named tuples and then a dataframe
    cluster_specs_sh_ori.append([])
    for row in datafile_sh['cluster_descriptives']:
        row = np.concatenate([row, [0,0]]) # Placeholders for n-ipsi-cells and n-contra-cells which are not included in shuffled data
        cluster_specs_sh_ori[sh].append(odcfunctions.ODpatterns( *row ))
    df_sh_x = load_dataframe(cluster_specs_sh_ori[sh], include_entries, df_columns, shuffle_nr=sh)
    df_sh_x = add_computed_variables(df_sh_x)

    # Add to df
    df_sh_ori = pd.concat( [df_sh_ori, df_sh_x] )
    
# Set columns to numeric
df_sh_ori["Nr"] = pd.to_numeric(df_sh_ori["Nr"])
df_sh_ori["Length (Alonso)"] = pd.to_numeric(df_sh_ori["Length (Alonso)"])
df_sh_ori["Area"] = pd.to_numeric(df_sh_ori["Area"])
df_sh_ori.info()


# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Select data and get mean per mouse

# GCaMP6s mice
df_od = df_od.query("mousetype==1")
df_sh_odi = df_sh_odi.query("mousetype==1")
df_sh_ori = df_sh_ori.query("mousetype==1")

# Get mean per mouse
df_od_m = df_od.groupby(['Mouse','Cluster type']).mean()
df_od_m = df_od_m.reset_index()

# Get mean per mouse of the shuffled ODI data
# First mean across clusters of the same type
df_sh_odi_m_shnr = df_sh_odi.groupby(['Mouse','Cluster type','Shuffle nr']).mean()
df_sh_odi_m_shnr = df_sh_odi_m_shnr.reset_index()

# Then mean across shuffles
df_sh_odi_m = df_sh_odi_m_shnr.groupby(['Mouse','Cluster type']).mean()
df_sh_odi_m = df_sh_odi_m.reset_index()

# Get mean per mouse of the shuffled ORI data
# First mean across clusters of the same type
df_sh_ori_m_shnr = df_sh_ori.groupby(['Mouse','Cluster type','Shuffle nr']).mean()
df_sh_ori_m_shnr = df_sh_ori_m_shnr.reset_index()

# Then mean across shuffles
df_sh_ori_m = df_sh_ori_m_shnr.groupby(['Mouse','Cluster type']).mean()
df_sh_ori_m = df_sh_ori_m.reset_index()

df_ipsi = df_od.query("`Cluster type` == 1")
df_contra = df_od.query("`Cluster type` == 2")

# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Functions for plotting

# Colors, markers
marker_list = ["o","x","o","x"]
color_list = ["#aa0000","#888888","#0000aa","#888888"]
marker_list_3 = ["o","x","o","x","o","x"]
color_list_3 = ["#aa0000","#888888","#aa00aa","#888888","#0000aa","#888888"]

def three_bar_plot(df_od_m, df_sh_odi_m, varname, ylabel, marker_list_3, color_list_3, y_minmax, y_step, savetag, msettings, y_margin=0, y_axis_margin=0):
    # CMF
    data_ipsi = np.array(df_od_m.query("`Cluster type` == 1")[varname])
    data_inter = np.array(df_od_m.query("`Cluster type` == 3")[varname])
    data_contra = np.array(df_od_m.query("`Cluster type` == 2")[varname])
    
    # CMF, shuffled
    data_ipsi_sh = np.array(df_sh_odi_m.query("`Cluster type` == 1")[varname])
    data_inter_sh = np.array(df_sh_odi_m.query("`Cluster type` == 3")[varname])
    data_contra_sh = np.array(df_sh_odi_m.query("`Cluster type` == 2")[varname])
    
    data = np.stack([ 
        np.concatenate( [np.zeros_like(data_ipsi), np.zeros_like(data_ipsi_sh), np.zeros_like(data_inter)+1, np.zeros_like(data_inter_sh)+1, 
                         np.zeros_like(data_contra)+2, np.zeros_like(data_contra_sh)+2]),
        np.concatenate( [np.zeros_like(data_ipsi), np.zeros_like(data_ipsi_sh)+1, np.zeros_like(data_inter), np.zeros_like(data_inter_sh)+1, 
                         np.zeros_like(data_contra), np.zeros_like(data_contra_sh)+1]),
        np.concatenate( [data_ipsi, data_ipsi_sh, data_inter, data_inter_sh, data_contra, data_contra_sh]) ])
    df_data = pd.DataFrame(data.T, columns=["Eye","Shuffle",varname])
    
    # Make barplot
    fig,ax = plottingtools.init_figure(fig_size=(4.5,3.5))
    sns.swarmplot( data=df_data, x="Eye", y=varname, hue="Shuffle",
                   linewidth=1, edgecolor=None, size=1, dodge=True, ax=ax )
    odcfunctions.redraw_markers( ax, marker_list_3, color_list_3, size=1, reduce_x_width=1 )
        
    plt.bar( 0-0.2, np.mean(data_ipsi), 0.25, color='None', edgecolor="#000000", linewidth=1 )
    plt.bar( 0+0.2, np.mean(data_ipsi_sh), 0.25, color='None', edgecolor="#888888", linewidth=1 )
    plt.bar( 1-0.2, np.mean(data_inter), 0.25, color='None', edgecolor="#000000", linewidth=1 )
    plt.bar( 1+0.2, np.mean(data_inter_sh), 0.25, color='None', edgecolor="#888888", linewidth=1 )
    plt.bar( 2-0.2, np.mean(data_contra), 0.25, color='None', edgecolor="#000000", linewidth=1 )
    plt.bar( 2+0.2, np.mean(data_contra_sh), 0.25, color='None', edgecolor="#888888", linewidth=1 )
    
    plottingtools.finish_panel( ax, title="", ylabel=ylabel, xlabel="Eye", legend="off", 
                                y_minmax=y_minmax, y_step=y_step, y_margin=y_margin, y_axis_margin=y_axis_margin, 
                                x_minmax=[0,2], x_margin=0.55, x_axis_margin=0.4, 
                                x_ticks=[0,1,2], x_ticklabels=["Ipsi","Inter","Contra"], x_tick_rotation=0 )
    
    savename = os.path.join( msettings.savepath, "{}".format(savetag) )
    ax.set_position((0.4,0.3,0.5,0.6))
    plt.savefig(savename+".pdf", transparent=True)

    # Normalize data per mouse
    mouse_mean = (data_ipsi+data_inter+data_contra) / 3
    data_ipsi_n = data_ipsi - mouse_mean
    data_inter_n = data_inter - mouse_mean
    data_contra_n = data_contra - mouse_mean
    
    print(" ")
    print(varname)
    p = statstools.report_kruskalwallis( [data_ipsi,data_inter,data_contra], n_indents=0, alpha=0.05, bonferroni=1, preceding_text="" )
    p = statstools.report_kruskalwallis( [data_ipsi_n,data_inter_n,data_contra_n], n_indents=0, alpha=0.05, bonferroni=1, preceding_text="Normalized per mouse (paired), " )
    
    p = statstools.report_wmpsr_test( data_ipsi, data_contra, n_indents=0, 
                                  alpha=0.05, alternative="two-sided", bonferroni=1, preceding_text="Ipsi vs Contra: " )
    p = statstools.report_wmpsr_test( data_ipsi, data_inter, n_indents=0, 
                                  alpha=0.05, alternative="two-sided", bonferroni=1, preceding_text="Ipsi vs Inter: " )
    p = statstools.report_wmpsr_test( data_inter, data_contra, n_indents=0, 
                                  alpha=0.05, alternative="two-sided", bonferroni=1, preceding_text="Inter vs Contra: " )
    p = statstools.report_wmpsr_test( data_ipsi, data_ipsi_sh, n_indents=0, 
                                  alpha=0.05, alternative="two-sided", bonferroni=1, preceding_text="Ipsi vs Ipsi-shuffle: " )
    p = statstools.report_wmpsr_test( data_inter, data_inter_sh, n_indents=0, 
                                  alpha=0.05, alternative="two-sided", bonferroni=1, preceding_text="Inter vs Inter-shuffle: " )
    p = statstools.report_wmpsr_test( data_contra, data_contra_sh, n_indents=0, 
                                  alpha=0.05, alternative="two-sided", bonferroni=1, preceding_text="Contra vs Contra-shuffle: " )


# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Make plot for number of clusters (Fig 3c)

# Number of clusters per mouse
df_od_m_1 = df_od_m.query("`Cluster type` == 1")
n_ipsi_clusters,n_contra_clusters = np.array(df_od_m_1["n-ipsi"]),np.array(df_od_m_1["n-contra"])

# Number of clusters per mouse, shuffled
df_sh_odi_m_1 = df_sh_odi_m.query("`Cluster type` == 1")
n_ipsi_clusters_sh,n_contra_clusters_sh = np.array(df_sh_odi_m_1["n-ipsi"]),np.array(df_sh_odi_m_1["n-contra"])

data = np.stack([ 
    np.concatenate( [np.zeros_like(n_ipsi_clusters), np.zeros_like(n_ipsi_clusters_sh), np.zeros_like(n_contra_clusters)+1, np.zeros_like(n_contra_clusters_sh)+1]),
    np.concatenate( [np.zeros_like(n_ipsi_clusters), np.zeros_like(n_ipsi_clusters_sh)+1, np.zeros_like(n_contra_clusters), np.zeros_like(n_contra_clusters_sh)+1]),
    np.concatenate( [n_ipsi_clusters, n_ipsi_clusters_sh, n_contra_clusters, n_contra_clusters_sh]) ])
df_n_clust = pd.DataFrame(data.T, columns=["Eye","Shuffle","n-clusters"])

# Make barplot
fig,ax = plottingtools.init_figure(fig_size=(3.5,3.5))
sns.swarmplot( data=df_n_clust, x="Eye", y="n-clusters", hue="Shuffle",
               linewidth=1, edgecolor=None, size=1, dodge=True, ax=ax )
odcfunctions.redraw_markers( ax, marker_list, color_list, size=1, reduce_x_width=1 )

for x in range(2):
    y = np.mean( df_n_clust.query("Eye == {} and Shuffle ==  0".format(x))["n-clusters"] )
    y_sh = np.mean( df_n_clust.query("Eye == {} and Shuffle ==  1".format(x))["n-clusters"] )

    plt.bar( x-0.2, y, 0.25, color='None', edgecolor="#000000", linewidth=1 )
    plt.bar( x+0.2, y_sh, 0.25, color='None', edgecolor="#888888", linewidth=1 )

plottingtools.finish_panel( ax, title="", ylabel="# Clusters", xlabel="Eye", legend="off", 
                            y_minmax=[0,10], y_step=[5,0], y_margin=0, y_axis_margin=0, 
                            x_minmax=[0,1], x_margin=0.55, x_axis_margin=0.4, 
                            x_ticks=[0,1], x_ticklabels=["Ipsi","Contra"], x_tick_rotation=0 )

savename = os.path.join( msettings.savepath, "Fig-3c-number-of-clusters" )
ax.set_position((0.4,0.3,0.5,0.6))
plt.savefig(savename+".pdf", transparent=True)

# Statistics
data_ipsi = np.array(df_n_clust.query("Eye == 0 and Shuffle ==  0")["n-clusters"])
data_ipsi_sh = np.array(df_n_clust.query("Eye == 0 and Shuffle ==  1")["n-clusters"])
data_contra = np.array(df_n_clust.query("Eye == 1 and Shuffle ==  0")["n-clusters"])
data_conyra_sh = np.array(df_n_clust.query("Eye == 1 and Shuffle ==  1")["n-clusters"])

print("\nNumber of clusters")
p = statstools.report_wmpsr_test( data_ipsi, data_contra, n_indents=0, 
                              alpha=0.05, alternative="two-sided", bonferroni=1, preceding_text="Ipsi vs Contra: " )
p = statstools.report_wmpsr_test( data_ipsi, data_ipsi_sh, n_indents=0, 
                              alpha=0.05, alternative="two-sided", bonferroni=1, preceding_text="Ipsi vs Ipsi-shuffle: " )
p = statstools.report_wmpsr_test( data_contra, data_conyra_sh, n_indents=0, 
                              alpha=0.05, alternative="two-sided", bonferroni=1, preceding_text="Contra vs Contra-shuffle: " )


# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Make plot for Area (Fig 3d)

# Cluster area
area_ipsi = np.array(df_od_m.query("`Cluster type` == 1")["Area"])
area_contra = np.array(df_od_m.query("`Cluster type` == 2")["Area"])

# Cluster area, shuffled
area_ipsi_sh = np.array(df_sh_odi_m.query("`Cluster type` == 1")["Area"])
area_contra_sh = np.array(df_sh_odi_m.query("`Cluster type` == 2")["Area"])

data = np.stack([ 
    np.concatenate( [np.zeros_like(area_ipsi), np.zeros_like(area_ipsi_sh), np.zeros_like(area_contra)+1, np.zeros_like(area_contra_sh)+1]),
    np.concatenate( [np.zeros_like(area_ipsi), np.zeros_like(area_ipsi_sh)+1, np.zeros_like(area_contra), np.zeros_like(area_contra_sh)+1]),
    np.concatenate( [area_ipsi, area_ipsi_sh, area_contra, area_contra_sh]) ])
df_area = pd.DataFrame(data.T, columns=["Eye","Shuffle","Area"])

# Make barplot
fig,ax = plottingtools.init_figure(fig_size=(3.5,3.5))
sns.swarmplot( data=df_area, x="Eye", y="Area", hue="Shuffle",
               linewidth=1, edgecolor=None, size=1, dodge=True, ax=ax )
odcfunctions.redraw_markers( ax, marker_list, color_list, size=1, reduce_x_width=1 )

plt.bar( 0-0.2, np.mean(area_ipsi), 0.25, color='None', edgecolor="#000000", linewidth=1 )
plt.bar( 0+0.2, np.mean(area_ipsi_sh), 0.25, color='None', edgecolor="#888888", linewidth=1 )
plt.bar( 1-0.2, np.mean(area_contra), 0.25, color='None', edgecolor="#000000", linewidth=1 )
plt.bar( 1+0.2, np.mean(area_contra_sh), 0.25, color='None', edgecolor="#888888", linewidth=1 )

plottingtools.finish_panel( ax, title="", ylabel="Area (um^2)", xlabel="Eye", legend="off", 
                            y_minmax=[0,50000], y_step=[25000,0], y_margin=0, y_axis_margin=0, 
                            x_minmax=[0,1], x_margin=0.55, x_axis_margin=0.4, 
                            x_ticks=[0,1], x_ticklabels=["Ipsi","Contra"], x_tick_rotation=0 )

savename = os.path.join( msettings.savepath, "Fig-3d-area-of-clusters" )
ax.set_position((0.4,0.3,0.5,0.6))
plt.savefig(savename+".pdf", transparent=True)

print("\nArea")
p = statstools.report_wmpsr_test( area_ipsi, area_contra, n_indents=0, 
                              alpha=0.05, alternative="two-sided", bonferroni=1, preceding_text="Ipsi vs Contra: " )
p = statstools.report_wmpsr_test( area_ipsi, area_ipsi_sh, n_indents=0, 
                              alpha=0.05, alternative="two-sided", bonferroni=1, preceding_text="Ipsi vs Ipsi-shuffle: " )
p = statstools.report_wmpsr_test( area_contra, area_contra_sh, n_indents=0, 
                              alpha=0.05, alternative="two-sided", bonferroni=1, preceding_text="Contra vs Contra-shuffle: " )


# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Make plot for Width (Fig 3e)

max_hist = 100
n_bins = 25

# Get histograms width widths for all measurements along the skeleton axes
widths_ipsi = np.full((n_bins+1, 9, 10), fill_value=np.NaN) # bins x mice x clusters
widths_contra = np.full((n_bins+1, 9, 10), fill_value=np.NaN) # bins x mice x clusters
widths_ipsi_sh = np.full((n_bins+1, 9, 10, 10), fill_value=np.NaN) # bins x mice x clusters x shuffles
widths_contra_sh = np.full((n_bins+1, 9, 10, 10), fill_value=np.NaN) # bins x mice x clusters x shuffles

widths_ipsi_mean = [[],]
widths_contra_mean = [[],]
widths_ipsi_sh_mean = [[] for i in range(10)]
widths_contra_sh_mean = [[] for i in range(10)]

# IPSI
m = 0
c = 0
last_mouse = "O02"
for cdata in cluster_specs:
    if cdata.type == 1 and int(cdata.mouse[1:])<15:
        if cdata.mouse != last_mouse:
            last_mouse = cdata.mouse
            widths_ipsi_mean.append([])
            m += 1
            c = 0
        widths_hist, edges = np.histogram( cdata.wid_alo, bins=n_bins, range=[0,max_hist] )
        n_extremes = np.array([np.sum(cdata.wid_alo>max_hist),])
        widths_ipsi[:,m,c] = np.concatenate( [widths_hist, np.array(n_extremes)] )
        c += 1
        widths_ipsi_mean[m].append(cdata.wid_alo)

# CONTRA
m = 0
c = 0
last_mouse = "O02"
for cdata in cluster_specs:
    if cdata.type == 2 and int(cdata.mouse[1:])<15:
        if cdata.mouse != last_mouse:
            last_mouse = cdata.mouse
            widths_contra_mean.append([])
            m += 1
            c = 0
        widths_hist, edges = np.histogram( cdata.wid_alo, bins=n_bins, range=[0,max_hist] )
        n_extremes = np.array([np.sum(cdata.wid_alo>max_hist),])
        widths_contra[:,m,c] = np.concatenate( [widths_hist, np.array(n_extremes)] )
        c += 1
        widths_contra_mean[m].append(cdata.wid_alo)

# IPSI SHUFFLE
for sh in range(10):
    widths_ipsi_sh_mean[sh] = [[],]
    m = 0
    c = 0
    last_mouse = "O02"
    for cdata in cluster_specs_sh_odi[sh]:
        if cdata.type == 1 and int(cdata.mouse[1:])<15:
            if cdata.mouse != last_mouse:
                last_mouse = cdata.mouse
                widths_ipsi_sh_mean[sh].append([])
                m += 1
                c = 0
            widths_hist, edges = np.histogram( cdata.wid_alo, bins=n_bins, range=[0,max_hist] )
            n_extremes = np.array([np.sum(cdata.wid_alo>max_hist),])
            widths_ipsi_sh[:,m,c,sh] = np.concatenate( [widths_hist, np.array(n_extremes)] )
            c += 1
            widths_ipsi_sh_mean[sh][m].append(cdata.wid_alo)

# CONTRA SHUFFLE
for sh in range(10):
    widths_contra_sh_mean[sh] = [[],]
    m = 0
    c = 0
    last_mouse = "O02"
    for cdata in cluster_specs_sh_odi[sh]:
        if cdata.type == 2 and int(cdata.mouse[1:])<15:
            if cdata.mouse != last_mouse:
                last_mouse = cdata.mouse
                widths_contra_sh_mean[sh].append([])
                m += 1
                c = 0
            widths_hist, edges = np.histogram( cdata.wid_alo, bins=n_bins, range=[0,max_hist] )
            n_extremes = np.array([np.sum(cdata.wid_alo>max_hist),])
            widths_contra_sh[:,m,c,sh] = np.concatenate( [widths_hist, np.array(n_extremes)] )
            c += 1
            widths_contra_sh_mean[sh][m].append(cdata.wid_alo)

# Get the mean width across mice (thus longer clusters count for more)
for m in range(len(widths_ipsi_mean)):
    widths_ipsi_mean[m] = np.nanmean(np.concatenate(widths_ipsi_mean[m]))
    
for m in range(len(widths_contra_mean)):
    widths_contra_mean[m] = np.nanmean(np.concatenate(widths_contra_mean[m]))

for sh in range(10):
    for m in range(len(widths_ipsi_sh_mean[sh])):
        widths_ipsi_sh_mean[sh][m] = np.nanmean(np.concatenate(widths_ipsi_sh_mean[sh][m]))

for sh in range(10):
    for m in range(len(widths_contra_sh_mean[sh])):
        widths_contra_sh_mean[sh][m] = np.nanmean(np.concatenate(widths_contra_sh_mean[sh][m]))

# Convert to numpy and get the mean across shuffles
widths_ipsi_mean = np.array(widths_ipsi_mean)
widths_contra_mean = np.array(widths_contra_mean)
widths_ipsi_sh_mean = np.nanmean(np.array(widths_ipsi_sh_mean),axis=0)
widths_contra_sh_mean = np.nanmean(np.array(widths_contra_sh_mean),axis=0)

print("\nWidth")
p = statstools.report_wmpsr_test( widths_ipsi_mean, widths_ipsi_sh_mean, n_indents=0, 
                              alpha=0.05, alternative="two-sided", bonferroni=1, preceding_text="Ipsi vs Ipsi-shuffle: " )
p = statstools.report_wmpsr_test( widths_contra_mean, widths_contra_sh_mean, n_indents=0, 
                              alpha=0.05, alternative="two-sided", bonferroni=1, preceding_text="Contra vs Contra-shuffle: " )

# Convert histograms to summed width counts into [bins x mice (x shuffles)]
widths_ipsi_m = np.nansum(widths_ipsi,axis=2)
widths_contra_m = np.nansum(widths_contra,axis=2)
widths_ipsi_sh_m = np.nansum(widths_ipsi_sh,axis=2)
widths_contra_sh_m = np.nansum(widths_contra_sh,axis=2)

# Now normalize to sum to 1
for m in range(widths_ipsi_m.shape[1]):
    widths_ipsi_m[:,m] = widths_ipsi_m[:,m] / np.nansum(widths_ipsi_m[:,m])
    widths_contra_m[:,m] = widths_contra_m[:,m] / np.nansum(widths_contra_m[:,m])
    for sh in range(10):
        widths_ipsi_sh_m[:,m,sh] = widths_ipsi_sh_m[:,m,sh] / np.nansum(widths_ipsi_sh_m[:,m,sh])
        widths_contra_sh_m[:,m,sh] = widths_contra_sh_m[:,m,sh] / np.nansum(widths_contra_sh_m[:,m,sh])

# Finally, get the mean across shuffles
widths_ipsi_sh_m = np.nanmean( widths_ipsi_sh_m, axis=2)
widths_contra_sh_m = np.nanmean( widths_contra_sh_m, axis=2)


# Make histogram plots
xvalues = np.arange(0,max_hist,max_hist/n_bins) + (0.5*(max_hist/n_bins))
x_extr = xvalues[-1] + (2*(max_hist/n_bins))

# IPSI
fig,ax = plottingtools.init_figure(fig_size=(3.5,2.5))
m,e,n = statstools.mean_sem(widths_ipsi_sh_m,axis=1)
plottingtools.line( xvalues, m[:n_bins], e[:n_bins], line_color='#888888', line_width=1, sem_color="#aaaaaa", shaded=True, top_bar_width=0.2 )
plt.plot(x_extr, m[n_bins], marker="x", linestyle='None', color="#888888", markersize=2, markeredgecolor="#888888", markerfacecolor="None")

m,e,n = statstools.mean_sem(widths_ipsi_m,axis=1)
plottingtools.line( xvalues, m[:n_bins], e[:n_bins], line_color='#aa0000', line_width=1, sem_color="#ffaaaa", shaded=True, top_bar_width=0.2 )
plt.plot(x_extr, m[n_bins], marker="o", linestyle='None', color="#aa0000", markersize=2, markeredgecolor="#aa0000", markerfacecolor="None")

plottingtools.finish_panel( ax, title="", ylabel="p", xlabel="Width (um)", legend="off", 
                            y_minmax=[0,0.2], y_step=[0.1,1], y_margin=0.01, y_axis_margin=0, 
                            x_minmax=[0,x_extr], x_step=[50,0], x_margin=10, x_axis_margin=0 )

savename = os.path.join( msettings.savepath, "Fig-3e-width-of-ipsi-clusters" )
ax.set_position((0.4,0.3,0.5,0.6))
plt.savefig(savename+".pdf", transparent=True)


# CONTRA
fig,ax = plottingtools.init_figure(fig_size=(3.5,2.5))
m,e,n = statstools.mean_sem(widths_contra_sh_m,axis=1)
plottingtools.line( xvalues, m[:n_bins], e[:n_bins], line_color='#888888', line_width=1, sem_color="#aaaaaa", shaded=True, top_bar_width=0.2 )
plt.plot(x_extr, m[n_bins], marker="x", linestyle='None', color="#888888", markersize=2, markeredgecolor="#888888", markerfacecolor="None")

m,e,n = statstools.mean_sem(widths_contra_m,axis=1)
plottingtools.line( xvalues, m[:n_bins], e[:n_bins], line_color='#0000aa', line_width=1, sem_color="#aaaaff", shaded=True, top_bar_width=0.2 )
plt.plot(x_extr, m[n_bins], marker="o", linestyle='None', color="#0000aa", markersize=2, markeredgecolor="#0000aa", markerfacecolor="None")

plottingtools.finish_panel( ax, title="", ylabel="p", xlabel="Width (um)", legend="off", 
                            y_minmax=[0,0.2], y_step=[0.1,1], y_margin=0.01, y_axis_margin=0, 
                            x_minmax=[0,x_extr], x_step=[50,0], x_margin=10, x_axis_margin=0 )

savename = os.path.join( msettings.savepath, "Fig-3e-width-of-contra-regions" )
ax.set_position((0.4,0.3,0.5,0.6))
plt.savefig(savename+".pdf", transparent=True)


# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Make plot for CMF (Fig 3f)

# Cortical magnification factor
three_bar_plot( df_od_m, df_sh_odi_m, varname="CMF", ylabel="CMF (mm^2/deg^2)", 
                marker_list_3=marker_list_3, color_list_3=color_list_3, y_minmax=[0,0.001], y_step=[0.0005,4], savetag="Fig-3f-cmf-of-clusters", msettings=msettings)


# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Make plot for Rs_w to iso-ODI (Fig 3g, h)

# Angle between Rs vector and iso-ODI lines
three_bar_plot( df_od_m, df_sh_odi_m, varname="Rs_w to iso-ODI", ylabel="Rs_w - iso-ODI", 
                marker_list_3=marker_list_3, color_list_3=color_list_3, y_minmax=[0,60], y_step=[30,0], savetag="Fig-3g-rsw2isoodi-of-clusters", msettings=msettings)

max_hist = 90
n_bins = 30

# Show the histogram for entire V1b
Rs_w_to_iso_odi_hist = []
for cdata in cluster_specs:
    if cdata.type == 4 and int(cdata.mouse[1:])<15:
        Rs_w_to_iso_odi_hist.append( cdata.Rs_w_to_iso_odi_hist / np.nansum(cdata.Rs_w_to_iso_odi_hist) )
Rs_w_to_iso_odi_hist = np.stack(Rs_w_to_iso_odi_hist,axis=1)

Rs_w_to_iso_odi_hist_sh = []
for sh in range(10):
    Rs_w_to_iso_odi_hist_sh.append([])
    for cdata in cluster_specs_sh_odi[sh]:
        if cdata.type == 4 and int(cdata.mouse[1:])<15:
            Rs_w_to_iso_odi_hist_sh[sh].append( cdata.Rs_w_to_iso_odi_hist / np.nansum(cdata.Rs_w_to_iso_odi_hist) )
    Rs_w_to_iso_odi_hist_sh[sh] = np.stack(Rs_w_to_iso_odi_hist_sh[sh],axis=1)
Rs_w_to_iso_odi_hist_sh = np.stack(Rs_w_to_iso_odi_hist_sh,axis=2)

# Finally, get the mean across shuffles
Rs_w_to_iso_odi_hist_sh = np.nanmean( Rs_w_to_iso_odi_hist_sh, axis=2)

# Make histogram plots
xvalues = np.arange(0,max_hist,max_hist/n_bins) + (0.5*(max_hist/n_bins))

# V1b
fig,ax = plottingtools.init_figure(fig_size=(3.5,2.5))
m,e,n = statstools.mean_sem(Rs_w_to_iso_odi_hist_sh,axis=1)
plottingtools.line( xvalues, m[:n_bins], e[:n_bins], line_color='#888888', line_width=1, sem_color="#aaaaaa", shaded=True, top_bar_width=0.2 )

m,e,n = statstools.mean_sem(Rs_w_to_iso_odi_hist,axis=1)
plottingtools.line( xvalues, m[:n_bins], e[:n_bins], line_color='#000000', line_width=1, sem_color="#666666", shaded=True, top_bar_width=0.2 )

plottingtools.finish_panel( ax, title="", ylabel="p", xlabel="Degrees", legend="off", 
                            y_minmax=[0,0.06], y_step=[0.03,2], y_margin=0.01, y_axis_margin=0, 
                            x_minmax=[0,90], x_step=[45,0], x_margin=8, x_axis_margin=0 )

savename = os.path.join( msettings.savepath, "Fig-3h-rsw2isoodi-v1b-histogram" )
ax.set_position((0.4,0.3,0.5,0.6))
plt.savefig(savename+".pdf", transparent=True)


# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Make plot for Iso-PrefOri to iso-ODI (Fig 3i)

max_hist = 90
n_bins = 30

# Show the histogram for entire V1b
od_to_fpo_hist = []
for cdata in cluster_specs:
    if cdata.type == 4 and int(cdata.mouse[1:])<15:
        od_to_fpo_hist.append( cdata.ori_to_od_grad_angle_hist / np.nansum(cdata.ori_to_od_grad_angle_hist) )
od_to_fpo_hist = np.stack(od_to_fpo_hist,axis=1)

od_to_fpo_hist_sh = []
for sh in range(10):
    od_to_fpo_hist_sh.append([])
    for cdata in cluster_specs_sh_odi[sh]:
        if cdata.type == 4 and int(cdata.mouse[1:])<15:
            od_to_fpo_hist_sh[sh].append( cdata.ori_to_od_grad_angle_hist / np.nansum(cdata.ori_to_od_grad_angle_hist) )
    od_to_fpo_hist_sh[sh] = np.stack(od_to_fpo_hist_sh[sh],axis=1)
od_to_fpo_hist_sh = np.stack(od_to_fpo_hist_sh,axis=2)

# Finally, get the mean across shuffles
od_to_fpo_hist_sh = np.nanmean( od_to_fpo_hist_sh, axis=2)

# Make histogram plots
xvalues = np.arange(0,max_hist,max_hist/n_bins) + (0.5*(max_hist/n_bins))

# V1b
fig,ax = plottingtools.init_figure(fig_size=(3.5,2.5))
m,e,n = statstools.mean_sem(od_to_fpo_hist_sh,axis=1)
plottingtools.line( xvalues, m[:n_bins], e[:n_bins], line_color='#888888', line_width=1, sem_color="#aaaaaa", shaded=True, top_bar_width=0.2 )

m,e,n = statstools.mean_sem(od_to_fpo_hist,axis=1)
plottingtools.line( xvalues, m[:n_bins], e[:n_bins], line_color='#000000', line_width=1, sem_color="#666666", shaded=True, top_bar_width=0.2 )

plottingtools.finish_panel( ax, title="", ylabel="p", xlabel="Degrees", legend="off", 
                            y_minmax=[0,0.06], y_step=[0.03,2], y_margin=0.01, y_axis_margin=0, 
                            x_minmax=[0,90], x_step=[45,0], x_margin=8, x_axis_margin=0 )

savename = os.path.join( msettings.savepath, "Fig-3i-isoodi2isofpoang-v1b-histogram" )
ax.set_position((0.4,0.3,0.5,0.6))
plt.savefig(savename+".pdf", transparent=True)


# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Make plot for Azimuth (Fig S12d)

# Azimuth
three_bar_plot( df_od_m, df_sh_odi_m, varname="Azimuth (cells)", ylabel="Azimuth (deg)", 
                marker_list_3=marker_list_3, color_list_3=color_list_3, y_minmax=[0,45], y_step=[15,0], savetag="Fig-S12d-azimuth-in-clusters", msettings=msettings)


# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Make plot for Elevation (Fig S12e)

# Elevation
three_bar_plot( df_od_m, df_sh_odi_m, varname="Elevation (cells)", ylabel="Elevation (deg)", 
                marker_list_3=marker_list_3, color_list_3=color_list_3, y_minmax=[0,30], y_step=[15,0], savetag="Fig-S12e-elevation-in-clusters", msettings=msettings, y_margin=4, y_axis_margin=3)


# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Make plot for Circ Var (Fig S12f)

# Circular variance
three_bar_plot( df_od_m, df_sh_odi_m, varname="CV (all)", ylabel="Circ. var.", 
                marker_list_3=marker_list_3, color_list_3=color_list_3, y_minmax=[0,0.8], y_step=[0.4,1], savetag="Fig-S12f-circvar-in-clusters", msettings=msettings)


# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Make plot for DSI (Fig S12g)

# Direction selectivity
three_bar_plot( df_od_m, df_sh_odi_m, varname="DSI (all)", ylabel="DSI", 
                marker_list_3=marker_list_3, color_list_3=color_list_3, y_minmax=[0,0.5], y_step=[0.25,2], savetag="Fig-S12g-dsi-in-clusters", msettings=msettings)


# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Make plot for RF size (Fig S12h)

# Receptive field size
three_bar_plot( df_od_m, df_sh_odi_m, varname="RF size (cells)", ylabel="RF size (deg)", 
                marker_list_3=marker_list_3, color_list_3=color_list_3, y_minmax=[0,20], y_step=[10,0], savetag="Fig-S12h-rfsize-in-clusters", msettings=msettings)


# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Function for plotting correlation

def correlation_plot_mice(xdata, ydata, xdata_sh, ydata_sh, xlabel, ylabel, y_minmax, y_step, x_minmax, x_step, savetag, msettings, 
                          y_margin=0, y_axis_margin=0, x_margin=0, x_axis_margin=0):
    
    # Make scatter plot
    fig,ax = plottingtools.init_figure(fig_size=(3.5,3.5))
    if xdata_sh is not None:
        sns.regplot(x=np.array(xdata_sh), y=np.array(ydata_sh), color="#888888", marker=".", scatter_kws={"s": 3}, 
                    line_kws={"color": "#888888", "linewidth": 0.5} )
    sns.regplot(x=np.array(xdata), y=np.array(ydata), color="#000000", marker=".", scatter_kws={"s": 3}, 
                line_kws={"color": "#000000", "linewidth": 0.5} )
        
    plottingtools.finish_panel( ax, title="", ylabel=ylabel, xlabel=xlabel, legend="off", 
                                y_minmax=y_minmax, y_step=y_step, y_margin=y_margin, y_axis_margin=y_axis_margin, 
                                x_minmax=x_minmax, x_step=x_step, x_margin=x_margin, x_axis_margin=x_axis_margin, x_tick_rotation=0 )
    
    savename = os.path.join( msettings.savepath, "{}".format(savetag) )
    ax.set_position((0.4,0.3,0.5,0.5))
    plt.savefig(savename+".pdf", transparent=True)
    
    print("\nCorrelation {} vs {}".format(xlabel,ylabel))
    r,p = pearsonr(xdata,ydata)
    print("- Real data: r={}, p={}".format(r,p))
    if xdata_sh is not None:
        r,p = pearsonr(xdata_sh,ydata_sh)
        print("- Shuffle: r={}, p={}".format(r,p))
        
        
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Make plot for correlation between CMF and # ipsi clusters (Fig S13a)

# Number of ipsi-clusters vs CMF (V1b)
xdata = np.array(df_od_m.query("`Cluster type` == 1")["n-ipsi"])
ydata = np.array(df_od_m.query("`Cluster type` == 4")["CMF"])
correlation_plot_mice(xdata=xdata, ydata=ydata, xdata_sh=None, ydata_sh=None, 
                      ylabel="CMF (mm^2/deg^2)", y_minmax=[0,0.001], y_step=[0.0005,4], 
                      xlabel="# ipsi clusters", x_minmax=[2,5], x_step=[1,0], x_margin=0.5, x_axis_margin=0.2,
                      savetag="Fig-S13a-CMF_V1b-vs-number_of_ipsiclusters-mouse", msettings=msettings)


# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Make plot for correlation between CMF and ODI (Fig S13b)

# ODI in ipsi-clusters vs CMF (V1b)
xdata = np.array(df_od_m.query("`Cluster type` == 1")["ODI"])
ydata = np.array(df_od_m.query("`Cluster type` == 4")["CMF"])
correlation_plot_mice(xdata=xdata, ydata=ydata, xdata_sh=None, ydata_sh=None, 
                      ylabel="CMF (mm^2/deg^2)", y_minmax=[0,0.001], y_step=[0.0005,4], 
                      xlabel="ODI", x_minmax=[-0.1,0.2], x_step=[0.1,1], x_margin=0.05, x_axis_margin=0.03,
                      savetag="Fig-S13b-CMF_V1b-vs-ODI_ipsiclusters-mouse", msettings=msettings)


# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Make plot for correlation between RF size and # ipsi clusters (Fig S13c)

# Number of ipsi-clusters vs RF size (V1b)
xdata = np.array(df_od_m.query("`Cluster type` == 1")["n-ipsi"])
ydata = np.array(df_od_m.query("`Cluster type` == 4")["RF size (cells)"])
correlation_plot_mice(xdata=xdata, ydata=ydata, xdata_sh=None, ydata_sh=None, 
                      ylabel="RF size (deg)", y_minmax=[14,20], y_step=[2,0], y_margin=2, y_axis_margin=1,
                      xlabel="# ipsi clusters", x_minmax=[2,5], x_step=[1,0], x_margin=0.5, x_axis_margin=0.2,
                      savetag="Fig-S13c-RFsizecells_V1b-vs-number_of_ipsiclusters-mouse", msettings=msettings)


# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Make plot for correlation between RF size and ODI (Fig S13d)

# ODI in ipsi-clusters vs RF size (V1b)
xdata = np.array(df_od_m.query("`Cluster type` == 1")["ODI"])
ydata = np.array(df_od_m.query("`Cluster type` == 4")["RF size (cells)"])
correlation_plot_mice(xdata=xdata, ydata=ydata, xdata_sh=None, ydata_sh=None, 
                      ylabel="RF size (deg)", y_minmax=[14,20], y_step=[2,0], y_margin=2, y_axis_margin=1,
                      xlabel="ODI", x_minmax=[-0.1,0.2], x_step=[0.1,1], x_margin=0.05, x_axis_margin=0.03,
                      savetag="Fig-S13d-RFsizecells_V1b-vs-ODI_ipsiclusters-mouse", msettings=msettings)       
        

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# That's all folks !!
print("\nDone.\n")

