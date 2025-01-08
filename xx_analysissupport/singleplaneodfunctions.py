#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Functions supporting analysis and plotting of single plane analysis

Created on Saturday 5 Dec 2020

@author: pgoltstein
"""

import sys, os
import numpy as np
import matplotlib.pyplot as plt

# Local imports
sys.path.append('../xx_analysissupport')
import plottingtools
import analysistools
import statstools



#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Functions

def generic_psth_overview_OD( psth, tm, xvalues, include_neurons, x_labels=np.arange(0,316,45), y_labels=["I","C"], scalebar=None, n_rows=5, n_cols=6, savepath="", savename="" ):

    # Display some plots
    fig,_ = plottingtools.init_figure(fig_size=(n_cols*4,n_rows*3))
    for nr,neuron in enumerate(include_neurons):
        ax = plt.subplot2grid( (n_rows,n_cols), (int(np.mod(nr/n_cols,n_rows)), int(np.mod(nr,n_cols))) )
        yscale = np.max( np.mean(psth[neuron,:,:,:,:],axis=2) + (np.std(psth[neuron,:,:,:,:],axis=2)/np.sqrt(psth.shape[3])) )

        x0,x1,y0,x1 = plottingtools.plot_psth_grid( ax, xvalues, psth[neuron,:,:,:,:], bs=None, y_scale=yscale, x_labels=x_labels, y_labels=y_labels, scalebar=scalebar )

        # Statistics
        ttc = tm[neuron,:,:,:]
        ttc = np.reshape(ttc,[-1,ttc.shape[-1]])
        try:
            p,_,_,_,_ = statstools.kruskalwallis( ttc )
        except:
            p = 1.0
        
        # statstools.report_kruskalwallis(ttc)
        ax.set_title("N: {}, p={:7.5f}".format(neuron,p), fontsize=6)
        plt.axis('off')

    # Save figure
    filename = os.path.join(savepath,savename)
    plottingtools.finish_figure( filename=filename, wspace=0.2, hspace=0.2 )


def calculate_OD_orientation_tuning( tm ):
    param_names = ["ROInr", "ODI", "ODI indep PD", "ODI all dirs", "Pref dir", "Pref ori", "Circ var", "Bandwidth", "Significance"]
    tun_params = np.zeros((tm.shape[0],len(param_names)))
    for nr in range(tm.shape[0]):

        # ROInr
        tun_params[nr,0] = nr

        # ODI
        ipsi_tc = np.mean( tm[nr,0,:,:], axis=1 )
        contra_tc = np.mean( tm[nr,1,:,:], axis=1 )
        tun_params[nr,1] = analysistools.odi(ipsi_tc, contra_tc, method=0)
        tun_params[nr,2] = analysistools.odi(ipsi_tc, contra_tc, method=1)
        tun_params[nr,3] = analysistools.odi(ipsi_tc, contra_tc, method=2)

        # Calculate the remaining on the tuning curve of the dominant eye
        if tun_params[nr,1] < 0:
            dom_tc = ipsi_tc
        else:
            dom_tc = contra_tc

        # Pref dir, Pref ori
        tun_params[nr,4],_ = analysistools.preferreddirection(dom_tc)
        tun_params[nr,5],_ = analysistools.preferredorientation(dom_tc)

        # Circ var, Bandwidth
        reslen,_ = analysistools.resultant(dom_tc, resultant_type="Orientation")
        tun_params[nr,6] = 1-reslen
        tun_params[nr,7] = analysistools.halfwidthhalfmax(dom_tc)

        # Significance
        ttc = tm[nr,:,:,:]
        ttc = np.reshape(ttc,[-1,ttc.shape[-1]])
        try:
            tun_params[nr,8],_,_,_,_ = statstools.kruskalwallis( ttc )
        except:
            tun_params[nr,8] = 1.0

    return tun_params, param_names


def calculate_retinotopic_tuning( tm ):
    param_names = ["ROInr", "Pref elev", "Pref azim", "Significance"]
    tun_params = np.zeros((tm.shape[0],len(param_names)))
    for nr in range(tm.shape[0]):

        # ROInr
        tun_params[nr,0] = nr

        # ODI
        elev_tc = np.mean( np.mean( tm[nr,:,:,:], axis=2 ), axis=1 )
        azim_tc = np.mean( np.mean( tm[nr,:,:,:], axis=2 ), axis=0 )

        # Pref dir, Pref ori
        tun_params[nr,1] = np.argmax(elev_tc)
        tun_params[nr,2] = np.argmax(azim_tc)

        # Significance
        ttc = tm[nr,:,:,:]
        ttc = np.reshape(ttc,[-1,ttc.shape[-1]])
        tun_params[nr,3],_,_,_,_ = statstools.kruskalwallis( ttc )

    return tun_params, param_names
