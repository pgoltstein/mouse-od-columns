#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Functions supporting analysis and plotting of single volume analysis

Created on Wednesday 9 Dec 2020

@author: pgoltstein
"""


# Imports
import sys, os, glob
import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sns

# Local imports
sys.path.append('../xx_analysissupport')
import plottingtools
import statstools



#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Functions


def load_volume( datapath, mousename, start_depth=180, depth_increment=10, exp_type="", skip_first_plane=False, include_very_first_plane=True, convert_to_micron_x=1.0, convert_to_micron_y=1.0, include_sign=False, exclude_double_xy=3.0, exclude_double_z=30.0, invert_odi_values=False, include_fovpos=False ):

    # Find all plane files
    plane_files = sorted(glob.glob(os.path.join(datapath, mousename+"*"+exp_type+"*")))
    count_planes = 0
    plane_files_new = []
    if skip_first_plane:
        for name in plane_files:
            if include_very_first_plane and count_planes == 0:
                plane_files_new.append(name)
            elif "-plane0-" not in name:
                plane_files_new.append(name)
            count_planes += 1
        plane_files = plane_files_new

    # Prepare data containers
    all_planes = []
    all_tms = []
    all_ims = []

    # Loop & load all planes
    depth = start_depth
    for nr,name in enumerate(plane_files):

        # Update depth
        if not include_fovpos:
            if nr > 0:
                depth += depth_increment
        else:
            # Use alternative depth method
            plane_no = int(name[name.find("plane")+5])
            depth = start_depth + (plane_no*depth_increment)
        print("{:>3d}: {}".format(depth,name))

        # Load data matrix
        plane_dict = np.load(name,allow_pickle=True).item()
        plane_data = plane_dict["tuning_parameters"]
        parameter_names = plane_dict["parameter_names"]
        aspect_ratio = plane_dict["aspect_ratio"]
        tm = plane_dict["tuningmatrix"]
        im = plane_dict["I"]

        # Add the field-of-view position to the neuron coordinates
        if include_fovpos:
            fovpos_x = plane_dict["fovpos_x"]*-1
            fovpos_y = plane_dict["fovpos_y"]
            print("Shifting spatial coordinates with FOV position: {},{} (Y,X)".format(fovpos_y,fovpos_x))
            plane_data[:, parameter_names.index("x")] = plane_data[:, parameter_names.index("x")] + fovpos_x
            plane_data[:, parameter_names.index("y")] = plane_data[:, parameter_names.index("y")] + fovpos_y

        # Add column with depth
        parameter_names.append("z")
        plane_data = np.concatenate([ plane_data, np.zeros((plane_data.shape[0],1))+depth ], axis=1)

        # Gather plane data and increase depth increment
        all_planes.append(plane_data)
        all_tms.append(tm)
        all_ims.append(im)

    # Create volumes and tuning matrix concatenating all depths
    volume = np.concatenate(all_planes,axis=0)
    tuningmatrix = np.concatenate(all_tms,axis=0)
    print("Volume size, all neurons: {}".format(volume.shape))

    # Exclude non-significant neurons if requested
    if include_sign is not False:
        sign = volume[:, parameter_names.index("Significance")]
        volume = volume[sign<include_sign, :]
        tuningmatrix = tuningmatrix[sign<include_sign, :, :, :]
        print("Volume size, only significant neurons: {}".format(volume.shape))

    # Convert xy coordinates from pixels to micron
    if convert_to_micron_x is not False and convert_to_micron_y is not False:
        print("Converting pixels to micron (x: {}, y: {})".format(convert_to_micron_x,convert_to_micron_y))
        volume[:, parameter_names.index("x")] = volume[:, parameter_names.index("x")] * convert_to_micron_x
        volume[:, parameter_names.index("y")] = volume[:, parameter_names.index("y")] * convert_to_micron_y

    # Exlude neurons that appear in multiple planes
    if exclude_double_xy is not False and exclude_double_z is not False:
        print("Looking for overlapping neurons")

        # Loop all neurons, from last to first
        ncnt = 0
        for n in range(volume.shape[0]-1,-1,-1):

            # Get parameters of this neuron
            x = volume[:, parameter_names.index("x")]
            y = volume[:, parameter_names.index("y")]
            z = volume[:, parameter_names.index("z")]
            sign = volume[:, parameter_names.index("Significance")]

            # Find overlapping neurons
            nearby_xy = np.argwhere( np.logical_and(np.logical_and( np.abs(x-x[n])<exclude_double_xy, np.abs(y-y[n])<exclude_double_xy ), np.abs(z-z[n])<exclude_double_z) ).ravel()
            if len(nearby_xy) > 1:

                # Find most significant neuron of the list
                most_sign = np.argmin( sign[nearby_xy] )

                # Remove others
                nearby_xy_less_sign = np.delete( nearby_xy, most_sign )
                volume = np.delete( volume, nearby_xy_less_sign, axis=0 )
                tuningmatrix = np.delete( tuningmatrix, nearby_xy_less_sign, axis=0 )
                ncnt += len(nearby_xy_less_sign)
        print("Removed {} overlapping neurons".format(ncnt))
        print("Volume size, excluding overlapping neurons: {}".format(volume.shape))

    # Invert ODI, in case recording was from right hemisphere
    if invert_odi_values:
        print("Inverting ODI values, recording was from right hemisphere")
        volume[:,parameter_names.index("ODI")] = volume[:,parameter_names.index("ODI")]*-1.0

    # Set the lowest neuron coordinate to 0,0
    if include_fovpos:
        min_x = np.min(volume[:,parameter_names.index("x")])
        min_y = np.min(volume[:,parameter_names.index("y")])
        print("Adjusting spatial coordinates to base 0,0 by subtracting the minimum along each axis (Y={}, X={})".format(min_y,min_x))
        volume[:,parameter_names.index("x")] = volume[:,parameter_names.index("x")] - min_x
        volume[:,parameter_names.index("y")] = volume[:,parameter_names.index("y")] - min_y

    # Return data
    return volume,parameter_names,aspect_ratio,all_ims,tuningmatrix


def generic_psth_overview_OD( psth, tm, xvalues, include_neurons=None, x_labels=np.arange(0,316,45), y_labels=["I","C"], scalebar=None, n_rows=5, n_cols=6, savepath="" ):

    if include_neurons is None:
        include_neurons = np.arange(psth.shape[0])

    # Display some plots
    last_safe_nr = 0
    for nr,neuron in enumerate(include_neurons):
        if np.mod(nr,n_rows*n_cols) == 0:
            if nr > 0:
                filename = os.path.join(savepath,"Neurons-{:03d}-{:03d}".format( include_neurons[last_safe_nr], include_neurons[nr] ))
                plottingtools.finish_figure( filename=filename, wspace=0.2, hspace=0.2 )
                last_safe_nr = include_neurons[nr]
            fig,_ = plottingtools.init_figure(fig_size=(29.7,21.0))
        ax = plt.subplot2grid( (n_rows,n_cols), (int(np.mod(nr/n_cols,n_rows)), int(np.mod(nr,n_cols))) )
        yscale = np.max(np.mean(psth[neuron,:,:,:,:],axis=2))
        x0,x1,y0,x1 = plottingtools.plot_psth_grid( ax, xvalues, psth[neuron,:,:,:,:], bs=None, y_scale=yscale, x_labels=x_labels, y_labels=y_labels, scalebar=scalebar )

        # Statistics
        ttc = tm[neuron,:,:,:]
        ttc = np.reshape(ttc,[-1,ttc.shape[-1]])
        p,_,_,_,_ = statstools.kruskalwallis( ttc )
        statstools.report_kruskalwallis(ttc)
        ax.set_title("N: {}, p={:7.5f}".format(neuron,p), fontsize=6)
        plt.axis('off')

        if nr == (n_rows*n_cols)-1:
            break

    # Save last figure as well
    filename = os.path.join(savepath,"Neurons-{:03d}-{:03d}".format( include_neurons[last_safe_nr], include_neurons[nr] ))
    plottingtools.finish_figure( filename=filename, wspace=0.2, hspace=0.2 )



def show_param_2d( ax, x, y, param, title, name=None, cmap="seismic_r", vmin=-1, vmax=1, d1=180, d2=540, size=12 ):
    plt.scatter( x, y, c=param, s=size, alpha=1.0, vmin=vmin, vmax=vmax, cmap=cmap, edgecolors="None" )
    if title == "mean":
        m,e,_ = statstools.mean_sem(param)
        ax.set_title("Depths {}-{}um\n{}: {:4.2f} (±{:4.2f})".format(d1,d2,name,m,e), fontsize=6)
    elif title is None:
        pass
    else :
        ax.set_title("Depths {}-{}um\n".format(d1,d2)+title, fontsize=6)
    ax.axis("equal")
    ax.invert_yaxis()
    plt.axis("off")



def show_local_od_density( volume, parameter_names, im_dims_yx=(1024,1024), aspect_ratio=1.2, n_bins=20, alpha=0.05, savepath="", mousename="", d1=180, d2=540 ):

    min_n_neurons = 4

    # Select only tuned neurons
    sign = volume[:, parameter_names.index("Significance")]
    volume = volume[sign<alpha, :]
    z = volume[:, parameter_names.index("z")]
    z_ix = np.logical_and( z>=d1, z<=d2 )
    volume = volume[z_ix, :]

    # Get parameters
    x = volume[:, parameter_names.index("x")]
    y = volume[:, parameter_names.index("y")]
    max_x = im_dims_yx[1]*aspect_ratio
    max_y = im_dims_yx[0]

    z = volume[:, parameter_names.index("z")]
    odi = volume[:, parameter_names.index("ODI")]
    mean_odi = np.nanmean(odi)
    print("ODI = {:4.2f}".format(mean_odi))
    # np.random.shuffle(odi_sh)
    all_zs = np.unique(z)
    n_neurons = len(odi)
    print("#neurons = {}".format(n_neurons))

    fig,_ = plottingtools.init_figure(fig_size=(36,24))

    ax = plt.subplot2grid((2,3),(0,0))
    polycollection0=plt.hexbin(x, y, gridsize=n_bins, cmap='coolwarm', extent=(0,max_x,0,max_y))
    ax.axis("equal")
    ax.invert_yaxis()
    plt.axis("off")
    offsets0 = polycollection0.get_offsets()
    bins0 = polycollection0.get_array()
    print("offsets0={}".format(offsets0.shape))
    print("bins0={}".format(bins0.shape))

    odi = volume[:, parameter_names.index("ODI")]
    volume_ipsi = volume[odi<0, :]
    x = volume_ipsi[:, parameter_names.index("x")]
    y = volume_ipsi[:, parameter_names.index("y")]
    print("#ipsi neurons = {}".format(len(x)))
    ax = plt.subplot2grid((2,3),(0,1))
    polycollection1=plt.hexbin(x, y, gridsize=n_bins, cmap='coolwarm', extent=(0,max_x,0,max_y))
    offsets1 = polycollection1.get_offsets()
    bins1 = polycollection1.get_array()
    print("offsets1={}".format(offsets1.shape))
    print("bins1={}".format(bins1.shape))
    ax.axis("equal")
    ax.invert_yaxis()
    plt.axis("off")

    odi = volume[:, parameter_names.index("ODI")]
    volume_contra = volume[odi>=0, :]
    x = volume_contra[:, parameter_names.index("x")]
    y = volume_contra[:, parameter_names.index("y")]
    print("#contra neurons = {}".format(len(x)))
    ax = plt.subplot2grid((2,3),(0,2))
    polycollection2=plt.hexbin(x, y, gridsize=n_bins, cmap='coolwarm', extent=(0,max_x,0,max_y))
    offsets2 = polycollection2.get_offsets()
    bins2 = polycollection2.get_array()
    print("offsets2={}".format(offsets2.shape))
    print("bins2={}".format(bins2.shape))
    ax.axis("equal")
    ax.invert_yaxis()
    plt.axis("off")

    x = offsets0[:,0]
    y = offsets0[:,1]
    binned_ipsi_frac = np.divide(bins1, bins0)
    binned_ipsi_frac[bins0<min_n_neurons] = np.NaN
    ax = plt.subplot2grid((2,3),(1,1))
    plt.hexbin( x=x, y=y, C=binned_ipsi_frac, gridsize=n_bins, cmap='coolwarm', extent=(0,max_x,0,max_y), vmin=0, vmax=1.0 )
    ax.axis("equal")
    ax.invert_yaxis()
    plt.axis("off")

    x = offsets0[:,0]
    y = offsets0[:,1]
    binned_contra_frac = np.divide(bins2, bins0)
    binned_contra_frac[bins0<min_n_neurons] = np.NaN
    ax = plt.subplot2grid((2,3),(1,2))
    plt.hexbin( x=x, y=y, C=binned_contra_frac, gridsize=n_bins, cmap='coolwarm', extent=(0,max_x,0,max_y), vmin=0, vmax=1.0 )
    ax.axis("equal")
    ax.invert_yaxis()
    plt.axis("off")

    x = offsets0[:,0]
    y = offsets0[:,1]
    binned_od_frac = np.divide(np.subtract(bins2,bins1), bins0)
    binned_od_frac[bins0<min_n_neurons] = np.NaN
    ax = plt.subplot2grid((2,3),(1,0))
    plt.hexbin( x=x, y=y, C=binned_od_frac, gridsize=n_bins, cmap='seismic_r', extent=(0,max_x,0,max_y), vmin=mean_odi-0.5, vmax=mean_odi+0.5 )
    ax.axis("equal")
    ax.invert_yaxis()
    plt.axis("off")

    # Save figure
    if savepath != "":
        filename = os.path.join(savepath,"VolumeODdensity-{}".format(mousename))
        plottingtools.finish_figure( filename=filename, wspace=0.2, hspace=0.2 )


def hexgrid_get_range(max_x, max_y, n_bins):
    fig,ax = plt.subplots()
    polycollection = plt.hexbin(np.arange(0,max_x), np.arange(0,max_y,max_y/max_x), gridsize=n_bins, cmap='coolwarm', extent=(0,max_x,0,max_y))
    coords = polycollection.get_offsets()
    plt.close(fig)
    x_edges = np.unique(coords[:,0])
    y_edges = np.unique(coords[:,1])
    x_step, y_step = x_edges[1], y_edges[1]
    n_x, n_y = len(x_edges), len(y_edges)
    return x_step, y_step, n_x, n_y, fig

def hexgrid_to_vector(polycollection, x_step, y_step, n_x, n_y):
    coords = polycollection.get_offsets()
    values = polycollection.get_array()
    x_ixs = np.round(coords[:,0]/x_step).astype(int)
    y_ixs = np.round(coords[:,1]/y_step).astype(int)
    vec_ixs = y_ixs + (x_ixs*n_y)
    datavec = np.full((n_x*n_y,), np.NaN)
    datavec[vec_ixs] = values
    return datavec

def get_hexgrid_coords(x_step, y_step, n_x, n_y):
    x = ((np.arange(n_x*n_y) / n_y) * x_step).astype(int)
    y = (np.mod(np.arange(n_x*n_y), n_y) * y_step).astype(int)
    return x,y

def plot_hexgrid( ax, x, y, max_x, max_y, n_bins, data=None, cmap="seismic_r", vmin=0, vmax=1, cleanup_fig=False):
    if cleanup_fig:
        fig,ax = plt.subplots()
    polycollection = plt.hexbin(x, y, C=data, gridsize=n_bins, cmap=cmap, extent=(0,max_x,0,max_y), vmin=vmin, vmax=vmax, linewidths=(0.1,))
    if not cleanup_fig:
        ax.axis("equal")
        ax.invert_yaxis()
        plt.axis("off")
    else:
        plt.close(fig)
    return polycollection

def density_across_cortex(volume, parameter_names, im_dims_yx=(1024,1024), aspect_ratio=1.2, n_bins=20, alpha=0.05, savepath="", mousename="", d1=180, d2=540, step=10):

    min_n_neurons = 4

    # General params
    max_x = np.ceil(im_dims_yx[1]*aspect_ratio)
    max_y = im_dims_yx[0]
    x_step, y_step, n_x, n_y, fig_dump = hexgrid_get_range(max_x, max_y, n_bins)

    # Select only tuned neurons
    sign = volume[:, parameter_names.index("Significance")]
    volume = volume[sign<alpha, :]
    z = volume[:, parameter_names.index("z")]
    odi = volume[:, parameter_names.index("ODI")]
    mean_odi = np.nanmean(odi)
    print("ODI = {:4.2f}".format(mean_odi))

    depth_bins = np.arange(d1,d2,step)
    n_depthbins = len(depth_bins)-1
    n_cols = int(np.ceil(np.sqrt(n_depthbins)))
    n_rows = int(np.ceil(n_depthbins/n_cols))
    if n_depthbins == 4:
        n_rows, n_cols = 1,4
    fig,ax = plottingtools.init_figure(fig_size=(10*n_cols,10*n_rows))

    datamat = np.zeros((n_x*n_y, n_depthbins))

    for b in range(n_depthbins):
        d1 = depth_bins[b]
        d2 = depth_bins[b+1]
        z_ix = np.logical_and( z>=d1, z<d2 )
        d_volume = volume[z_ix, :]

        # Get parameters
        x = d_volume[:, parameter_names.index("x")]
        y = d_volume[:, parameter_names.index("y")]
        odi = d_volume[:, parameter_names.index("ODI")]
        n_neurons = len(odi)
        print("* Depth {}-{}, #neurons = {}".format(d1,d2,n_neurons))
        polycollection = plot_hexgrid(ax, x, y, max_x, max_y, n_bins, cleanup_fig=True)
        all_vec = hexgrid_to_vector(polycollection, x_step, y_step, n_x, n_y)

        # # Get hexgrids with ODI and count
        # ax = plt.subplot2grid( (n_rows, n_cols), (int(b/n_cols), int(np.mod(b,n_cols))) )
        # polycollection = plot_hexgrid(ax, x, y, max_x, max_y, n_bins, data=odi, cmap="seismic_r", vmin=-0.8, vmax=0.8, cleanup_fig=False)
        # odi_vec = hexgrid_to_vector(polycollection, x_step, y_step, n_x, n_y)
        # datamat[:,b] = odi_vec

        # Get parameters
        d_volume_ipsi = d_volume[odi<0,:]
        x = d_volume_ipsi[:, parameter_names.index("x")]
        y = d_volume_ipsi[:, parameter_names.index("y")]
        n_neurons = len(x)
        print("           # ipsi neurons = {}".format(d1,d2,n_neurons))
        polycollection = plot_hexgrid(ax, x, y, max_x, max_y, n_bins, cleanup_fig=True)
        ipsi_vec = hexgrid_to_vector(polycollection, x_step, y_step, n_x, n_y)
        print(ipsi_vec)

        d_volume_contra = d_volume[odi>=0,:]
        x = d_volume_contra[:, parameter_names.index("x")]
        y = d_volume_contra[:, parameter_names.index("y")]
        n_neurons = len(x)
        print("         # contra neurons = {}".format(d1,d2,n_neurons))
        polycollection = plot_hexgrid(ax, x, y, max_x, max_y, n_bins, cleanup_fig=True)
        contra_vec = hexgrid_to_vector(polycollection, x_step, y_step, n_x, n_y)
        print(contra_vec)

        x, y = get_hexgrid_coords(x_step, y_step, n_x, n_y)
        binned_od_frac = np.divide((contra_vec-ipsi_vec), (contra_vec+ipsi_vec))
        binned_od_frac[all_vec<min_n_neurons] = np.NaN
        ax = plt.subplot2grid( (n_rows, n_cols), (int(b/n_cols), int(np.mod(b,n_cols))) )
        polycollection = plot_hexgrid(ax, x, y, max_x, max_y, n_bins, data=binned_od_frac, cmap="seismic_r", vmin=-1.0, vmax=1.0, cleanup_fig=False)

    # fig,ax = plottingtools.init_figure(fig_size=(15,15))
    # x, y = get_hexgrid_coords(x_step, y_step, n_x, n_y)
    # polycollection = plot_hexgrid(ax, x, y, max_x, max_y, n_bins, data=odi_std, cmap="seismic_r", vmin=0, vmax=1, cleanup_fig=False)


def odi_z(volume, parameter_names, aspect_ratio=1.2, alpha=0.05, savepath="", mousename="", d1=180, d2=540):

    bin_hw = 20

    # Select only tuned neurons
    sign = volume[:, parameter_names.index("Significance")]
    volume = volume[sign<alpha, :]
    z = volume[:, parameter_names.index("z")]
    z_ix = np.logical_and( z>=d1, z<d2 )
    volume = volume[z_ix, :]

    # Get parameters
    x = volume[:, parameter_names.index("x")]
    y = volume[:, parameter_names.index("y")]
    z = volume[:, parameter_names.index("z")]
    odi = volume[:, parameter_names.index("ODI")]
    odi_sh = np.array(volume[:, parameter_names.index("ODI")])
    np.random.shuffle(odi_sh)
    all_zs = np.unique(z)
    n_neurons = len(odi)
    print("#neurons = {}".format(n_neurons))

    z_dists = []
    odi_dists = []
    odi_dists_sh = []
    for n in range(n_neurons):
        xy_ix = np.logical_and(np.logical_and(x>(x[n]-bin_hw),x<(x[n]+bin_hw)), np.logical_and(y>(y[n]-bin_hw),y<(y[n]+bin_hw)))
        xy_ix[n] = False
        z_dist_col = np.abs(z[xy_ix]-z[n])
        odi_diff_col = np.abs(odi[xy_ix]-odi[n])
        odi_diff_col_sh = np.abs(odi_sh[xy_ix]-odi_sh[n])

        z_dist_ix = ((z_dist_col-d1)/10).astype(int)
        z_dists.append(z_dist_ix)
        odi_dists.append(odi_diff_col)
        odi_dists_sh.append(odi_diff_col_sh)
        if np.mod(n,1000)==0:
            print(n)

    z_dists = np.concatenate(z_dists,axis=0)
    odi_dists = np.concatenate(odi_dists,axis=0)
    odi_dists_sh = np.concatenate(odi_dists_sh,axis=0)

    odi_mean = np.full((36,),np.NaN)
    odi_stderr = np.full((36,),np.NaN)
    odi_sh_mean = np.full((36,),np.NaN)
    odi_sh_stderr = np.full((36,),np.NaN)
    for z in range(36):
        ix = z_dists==z
        odi_mean[z] = np.nanmean(odi_dists[ix])
        odi_stderr[z] = np.nanstd(odi_dists[ix]) / np.sqrt(np.sum(ix))
        odi_sh_mean[z] = np.nanmean(odi_dists_sh[ix])
        odi_sh_stderr[z] = np.nanstd(odi_dists_sh[ix]) / np.sqrt(np.sum(ix))

    xvalues = np.arange(36)
    fig,ax = plottingtools.init_figure(fig_size=(10,10))
    plottingtools.plot_curve( xvalues, odi_mean, stderr=odi_stderr, color="#000000" )
    plottingtools.plot_curve( xvalues, odi_sh_mean, stderr=odi_sh_stderr, color="#888888" )







def show_odi_nearestneighbor( volume, parameter_names, aspect_ratio=1.2, alpha=0.05, savepath="", mousename="", d1=180, d2=540 ):

    # Select only tuned neurons
    sign = volume[:, parameter_names.index("Significance")]
    volume = volume[sign<alpha, :]
    z = volume[:, parameter_names.index("z")]
    z_ix = np.logical_and( z>=d1, z<d2 )
    volume = volume[z_ix, :]

    # Get parameters
    x = volume[:, parameter_names.index("x")]
    y = volume[:, parameter_names.index("y")]
    z = volume[:, parameter_names.index("z")]
    odi = volume[:, parameter_names.index("ODI")]
    odi_sh = np.array(volume[:, parameter_names.index("ODI")])
    np.random.shuffle(odi_sh)
    all_zs = np.unique(z)
    n_neurons = len(odi)
    print("#neurons = {}".format(n_neurons))

    # Calculate 2d distance in ODI and um
    distances_2d = []
    diff_odi_2d = []
    diff_odi_2d_sh = []
    distances_2d_ipsi = []
    diff_odi_2d_ipsi = []
    distances_2d_contra = []
    diff_odi_2d_contra = []
    for n in range(n_neurons-1):
        distances_2d.append( np.sqrt( np.power(x[n+1:]-x[n],2) + np.power(y[n+1:]-y[n],2) ) )
        diff_odi_2d.append( np.abs( odi[n+1:] - odi[n] ) )
        diff_odi_2d_sh.append( np.abs( odi_sh[n+1:] - odi_sh[n] ) )
        if odi[n] < 0:
            distances_2d_ipsi.append( np.sqrt( np.power(x[n+1:]-x[n],2) + np.power(y[n+1:]-y[n],2) ) )
            diff_odi_2d_ipsi.append( np.abs( odi[n+1:] - odi[n] ) )
        else:
            distances_2d_contra.append( np.sqrt( np.power(x[n+1:]-x[n],2) + np.power(y[n+1:]-y[n],2) ) )
            diff_odi_2d_contra.append( np.abs( odi[n+1:] - odi[n] ) )
    distances_2d = np.concatenate(distances_2d, axis=0)
    diff_odi_2d = np.concatenate(diff_odi_2d, axis=0)
    diff_odi_2d_sh = np.concatenate(diff_odi_2d_sh, axis=0)
    distances_2d_ipsi = np.concatenate(distances_2d_ipsi, axis=0)
    diff_odi_2d_ipsi = np.concatenate(diff_odi_2d_ipsi, axis=0)
    distances_2d_contra = np.concatenate(distances_2d_contra, axis=0)
    diff_odi_2d_contra = np.concatenate(diff_odi_2d_contra, axis=0)
    print(distances_2d_ipsi.shape)
    print(distances_2d_contra.shape)

    # Bin the difference
    dist_bins = np.arange(0,2000,50)
    xvalues = dist_bins[:-1] - ((dist_bins[1]-dist_bins[0])/2)
    n_bins = len(dist_bins)-1
    dist_hist_mean = np.full((n_bins,), np.NaN)
    dist_hist_sh_mean = np.full((n_bins,), np.NaN)
    dist_hist_stderr = np.full((n_bins,), np.NaN)
    dist_hist_mean_ipsi = np.full((n_bins,), np.NaN)
    dist_hist_stderr_ipsi = np.full((n_bins,), np.NaN)
    dist_hist_mean_contra = np.full((n_bins,), np.NaN)
    dist_hist_stderr_contra = np.full((n_bins,), np.NaN)
    for b in range(n_bins):
        ix = np.logical_and( distances_2d>=dist_bins[b], distances_2d<dist_bins[b+1] )
        dist_hist_mean[b] = np.nanmean(diff_odi_2d[ix])
        dist_hist_sh_mean[b] = np.nanmean(diff_odi_2d_sh[ix])
        # dist_hist_stderr[b] = np.nanstd(diff_odi_2d[ix]) / np.sqrt(np.sum(ix))

        ix = np.logical_and( distances_2d_ipsi>=dist_bins[b], distances_2d_ipsi<dist_bins[b+1] )
        dist_hist_mean_ipsi[b] = np.nanmean(diff_odi_2d_ipsi[ix])
        dist_hist_stderr_ipsi[b] = np.nanstd(diff_odi_2d_ipsi[ix]) / np.sqrt(np.sum(ix))

        ix = np.logical_and( distances_2d_contra>=dist_bins[b], distances_2d_contra<dist_bins[b+1] )
        dist_hist_mean_contra[b] = np.nanmean(diff_odi_2d_contra[ix])
        dist_hist_stderr_contra[b] = np.nanstd(diff_odi_2d_contra[ix]) / np.sqrt(np.sum(ix))

    # Initialize figure
    fig,_ = plottingtools.init_figure(fig_size=(20,20))
    ax = plt.subplot2grid((2,2),(0,0))
    plottingtools.plot_curve( xvalues, dist_hist_mean-dist_hist_sh_mean, stderr=None, color="#000000" )

    ax = plt.subplot2grid((2,2),(0,1))
    plottingtools.plot_curve( xvalues, dist_hist_mean_contra, stderr=dist_hist_stderr_contra, color="#0000AA" )
    plottingtools.plot_curve( xvalues, dist_hist_mean_ipsi, stderr=dist_hist_stderr_ipsi, color="#AA0000" )

    # Distances between ipsi neurons
    odi = volume[:, parameter_names.index("ODI")]
    volume_ipsi = volume[odi<0, :]
    x = volume_ipsi[:, parameter_names.index("x")]
    y = volume_ipsi[:, parameter_names.index("y")]
    z = volume_ipsi[:, parameter_names.index("z")]
    odi = volume_ipsi[:, parameter_names.index("ODI")]
    n_neurons = len(odi)
    print("#ipsi neurons = {}".format(n_neurons))

    # Calculate 2d distance in ODI and um
    distances_2d_ipsi = []
    for n in range(n_neurons-1):
        distances_2d_ipsi.append( np.sqrt( np.power(x[n+1:]-x[n],2) + np.power(y[n+1:]-y[n],2) ) )
    distances_2d_ipsi = np.concatenate(distances_2d_ipsi, axis=0)

    odi = volume[:, parameter_names.index("ODI")]
    volume_contra = volume[odi>=0, :]
    x = volume_contra[:, parameter_names.index("x")]
    y = volume_contra[:, parameter_names.index("y")]
    z = volume_contra[:, parameter_names.index("z")]
    odi = volume_contra[:, parameter_names.index("ODI")]
    n_neurons = len(odi)
    print("#contra neurons = {}".format(n_neurons))

    # Calculate 2d distance in ODI and um
    distances_2d_contra = []
    for n in range(n_neurons-1):
        distances_2d_contra.append( np.sqrt( np.power(x[n+1:]-x[n],2) + np.power(y[n+1:]-y[n],2) ) )
    distances_2d_contra = np.concatenate(distances_2d_contra, axis=0)

    ax = plt.subplot2grid((2,2),(1,0))
    sns.histplot(distances_2d_ipsi,binwidth=20)
    ax = plt.subplot2grid((2,2),(1,1))
    sns.histplot(distances_2d_contra,binwidth=20)



def odi_map( volume, parameter_names, aspect_ratio=1.2, alpha=0.05, savepath="", mousename="", d1=180, d2=540 ):

    # Shuffle order in matrix so that no single plane gets priority
    volume = np.array(volume)
    np.random.shuffle(volume)

    # Parameters for full volume
    sign = volume[:, parameter_names.index("Significance")]

    # Select only tuned neurons
    volume = volume[sign<alpha, :]
    odi = volume[:, parameter_names.index("ODI")]
    # ori = volume[:, parameter_names.index("Pref ori")]
    # cv = volume[:, parameter_names.index("Circ var")]
    x = volume[:, parameter_names.index("x")]
    y = volume[:, parameter_names.index("y")]

    # ODI map
    odi_cnt = np.zeros((103,np.ceil(103*aspect_ratio).astype(int)))
    odi_map = np.zeros((103,np.ceil(103*aspect_ratio).astype(int)))
    for i in range(len(odi)):
        odi_cnt[int(y[i]/10),int(x[i]/10)] += 1.0
        odi_map[int(y[i]/10),int(x[i]/10)] += odi[i]

    odi_map = np.divide(odi_map, odi_cnt)
    odi_map[odi_cnt==0] = 0

    fft_im = np.fft.fft2(odi_map)
    fft_im = np.fft.fftshift(fft_im)
    print(fft_im.shape)

    def distance(point1,point2):
        return math.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

    def idealFilterLP(D0,imgShape):
        base = np.zeros(imgShape[:2])
        rows, cols = imgShape[:2]
        center = (rows/2,cols/2)
        for x in range(cols):
            for y in range(rows):
                if distance((y,x),center) < D0:
                    base[y,x] = 1
        return base

    def idealFilterHP(D0,imgShape):
        base = np.ones(imgShape[:2])
        rows, cols = imgShape[:2]
        center = (rows/2,cols/2)
        for x in range(cols):
            for y in range(rows):
                if distance((y,x),center) < D0:
                    base[y,x] = 0
        return base

    LP = idealFilterLP(4,fft_im.shape)==0
    HP = idealFilterHP(1000,fft_im.shape)==0

    fft_im[~np.logical_and(LP,HP)] = 0
    inv_center = np.fft.ifftshift(fft_im)
    filt_map = np.fft.ifft2(inv_center)

    fig,_ = plottingtools.init_figure(fig_size=(40*aspect_ratio,20))
    ax = plt.subplot2grid((1,3),(0,0))
    plt.imshow(odi_map)
    plt.colorbar()

    ax = plt.subplot2grid((1,3),(0,1))
    plt.imshow(np.log(1+np.abs(inv_center)))
    plt.colorbar()

    ax = plt.subplot2grid((1,3),(0,2))
    plt.imshow(np.abs(filt_map))
    plt.colorbar()

    return odi_map


def show_parameters_in_map( volume, parameter_names, aspect_ratio=1.2, alpha=0.05, savepath="", mousename="", d1=180, d2=540 ):

    # Shuffle order in matrix so that no single plane gets priority
    volume = np.array(volume)
    np.random.shuffle(volume)

    # Parameters for full volume
    sign = volume[:, parameter_names.index("Significance")]
    x = volume[:, parameter_names.index("x")]
    y = volume[:, parameter_names.index("y")]

    # Initialize figure
    fig,_ = plottingtools.init_figure(fig_size=(20*aspect_ratio,20))

    # Tuned neurons
    ax = plt.subplot2grid((2,2),(0,0))
    title = "#tuned: {}/{} ({:4.1f}%)".format( np.nansum(sign<alpha), volume.shape[0], 100*np.nansum(sign<alpha)/volume.shape[0] )
    show_param_2d( ax, x, y, sign>alpha, title=title, cmap="PiYG", vmin=0, vmax=1, d1=d1, d2=d2 )

    # Select only tuned neurons
    volume = volume[sign<alpha, :]
    odi = volume[:, parameter_names.index("ODI")]
    ori = volume[:, parameter_names.index("Pref ori")]
    cv = volume[:, parameter_names.index("Circ var")]
    x = volume[:, parameter_names.index("x")]
    y = volume[:, parameter_names.index("y")]

    # ODI, Preferred orientation, Circular variance
    ax = plt.subplot2grid((2,2),(0,1))
    show_param_2d( ax, x, y, odi, title="mean", name="ODI", cmap="seismic_r", vmin=-1, vmax=1, d1=d1, d2=d2 )

    ax = plt.subplot2grid((2,2),(1,0))
    show_param_2d( ax, x, y, ori, title="Pref ori", cmap="hsv", vmin=0, vmax=180, d1=d1, d2=d2 )

    ax = plt.subplot2grid((2,2),(1,1))
    show_param_2d( ax, x, y, cv, title="mean", name="Circ var", cmap="cool", vmin=0, vmax=1, d1=d1, d2=d2 )

    # Save figure
    if savepath != "":
        filename = os.path.join(savepath,"VolumeParams-{}".format(mousename))
        plottingtools.finish_figure( filename=filename, wspace=0.2, hspace=0.2 )

def show_parameter_map_depths( param_name, depth_edges, volume, parameter_names, aspect_ratio=1.2, alpha=0.05, savepath="", mousename="", fig_scale=7 ):

    # Shuffle order in matrix so that no single plane gets priority
    volume = np.array(volume)
    np.random.shuffle(volume)

    # Initialize figure
    n_rows = int(np.ceil(np.sqrt(len(depth_edges)-1)))
    n_cols = int(np.ceil((len(depth_edges)-1)/n_rows))
    fig,_ = plottingtools.init_figure(fig_size=(n_cols*fig_scale*aspect_ratio,n_rows*fig_scale))

    # Get parameters needed for plotting
    sign = volume[:, parameter_names.index("Significance")]
    x = volume[:, parameter_names.index("x")]
    y = volume[:, parameter_names.index("y")]
    z = volume[:, parameter_names.index("z")]
    if param_name != "#tuned":
        param = volume[:, parameter_names.index(param_name)]

    # Loop depths
    for nr,d1 in enumerate(depth_edges[:-1]):
        ax = plt.subplot2grid( (n_rows, n_cols), (int(nr/n_cols), int(np.mod(nr,n_cols))) )
        d2 = depth_edges[nr+1]

        # Plot map
        if param_name == "#tuned":
            d_select = np.logical_and(z>=d1,z<d2)
            d_x = x[d_select]
            d_y = y[d_select]
            d_sign = sign[d_select]
            title = "#tuned: {}/{} ({:4.1f}%)".format( np.nansum(d_sign<alpha), d_sign.shape[0], 100*np.nansum(d_sign<alpha)/d_sign.shape[0] )
            show_param_2d( ax, d_x, d_y, d_sign>alpha, title=title, cmap="PiYG", vmin=0, vmax=1, d1=d1, d2=d2 )
        else:
            d_select = np.logical_and(np.logical_and(z>=d1,z<d2),sign<alpha)
            d_x = x[d_select]
            d_y = y[d_select]
            d_param = param[d_select]

            if param_name == "ODI":
                show_param_2d( ax, d_x, d_y, d_param, title="mean", name="ODI", cmap="seismic_r", vmin=-1, vmax=1, d1=d1, d2=d2 )
            if param_name == "Pref ori":
                show_param_2d( ax, d_x, d_y, d_param, title="Pref ori", cmap="hsv", vmin=0, vmax=180, d1=d1, d2=d2 )
            if param_name == "Circ var":
                show_param_2d( ax, d_x, d_y, d_param, title="mean", name="Circ var", cmap="cool", vmin=0, vmax=1, d1=d1, d2=d2 )
            if param_name == "Pref elev":
                print("Pref elev: min={}, max={}".format( np.nanmin(d_param), np.nanmax(d_param) ))
                show_param_2d( ax, d_x, d_y, d_param, title="Pref elev", cmap="Spectral", vmin=0, vmax=2, d1=d1, d2=d2 )
            if param_name == "Pref azim":
                print("Pref azim: min={}, max={}".format( np.nanmin(d_param), np.nanmax(d_param) ))
                show_param_2d( ax, d_x, d_y, d_param, title="Pref azim", cmap="Spectral", vmin=0, vmax=4, d1=d1, d2=d2 )

    # Save figure
    if savepath != "":
        filename = os.path.join(savepath,"Volume-{}-{}".format(param_name,mousename))
        plottingtools.finish_figure( filename=filename, wspace=0.2, hspace=0.2, filetype="png" )
