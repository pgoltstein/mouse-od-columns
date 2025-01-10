#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Functions to display analyzed imaging datasets

Created on Fri Dec 4, 2020

@author: pgoltstein
"""


#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Imports

import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# Local imports
import sys
sys.path.append('../xx_analysissupport')
import statstools


#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Defaults

# settings for retaining pdf font
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.sans-serif'] = "Arial"
matplotlib.rcParams['font.family'] = "sans-serif"
# matplotlib.rcParams['axes.linewidth'] = 1.0

# Default settings
font_size = { "title": 6, "label": 6, "tick": 6, "text": 6, "legend": 6 }

# seaborn color context
color_context = {   'axes.edgecolor': '#000000',
                    'axes.labelcolor': '#000000',
                    'boxplot.capprops.color': '#000000',
                    'boxplot.flierprops.markeredgecolor': '#000000',
                    'grid.color': '#000000',
                    'patch.edgecolor': '#000000',
                    'text.color': '#000000',
                    'xtick.color': '#000000',
                    'ytick.color': '#000000' }
sns.set_context("notebook")


#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Functions


def init_fig( fig_size=(10,7), dpi=80, facecolor="w", edgecolor="w" ):
    # Convert fig size to inches (default is inches, fig_size argument is supposed to be in cm)
    inch2cm = 2.54
    fig_size = fig_size[0]/inch2cm,fig_size[1]/inch2cm
    with sns.axes_style(style="ticks",rc=color_context):
        fig = plt.figure(num=None, figsize=fig_size, dpi=dpi,
            facecolor=facecolor, edgecolor=edgecolor)
        return fig

def init_figure( fig_size=(10,7), dpi=80, facecolor="w", edgecolor="w" ):
    # Convert fig size to inches (default is inches, fig_size argument is supposed to be in cm)
    inch2cm = 2.54
    fig_size = fig_size[0]/inch2cm,fig_size[1]/inch2cm
    with sns.axes_style(style="ticks",rc=color_context):
        fig,ax = plt.subplots(num=None, figsize=fig_size, dpi=dpi,
            facecolor=facecolor, edgecolor=edgecolor)
        return fig,ax

def finish_figure( filename=None, wspace=None, hspace=None, filetype="pdf" ):
    """ Finish up layout and save """
    plt.tight_layout()
    if wspace is not None or hspace is not None:
        if wspace is None: wspace = 0.6
        if hspace is None: hspace = 0.8
        plt.subplots_adjust( wspace=wspace, hspace=hspace )
    if filename is not None:
        if filetype == "pdf":
            plt.savefig(filename+"."+filetype, transparent=True)
        else:
            plt.savefig(filename+"."+filetype, transparent=False)

def finish_panel( ax, title="", ylabel="", xlabel="", legend="off", y_minmax=None, y_step=None, y_margin=0.02, y_axis_margin=0.01, x_minmax=None, x_step=None, x_margin=0.55, x_axis_margin=0.55, x_ticks=None, x_ticklabels=None, y_ticks=None, y_ticklabels=None, x_tick_rotation=0, tick_size=None, label_size=None, title_size=None, legend_size=None, despine=True, legendpos=0):
    """ Finished axis formatting of an individual plot panel """
    if tick_size is None: tick_size=font_size['tick']
    if label_size is None: label_size=font_size['label']
    if title_size is None: title_size=font_size['title']
    if legend_size is None: legend_size=font_size['legend']

    # Set limits and trim spines
    if y_minmax is not None:
        ax.set_ylim(y_minmax[0]-y_margin,y_minmax[1]+y_margin)
    if x_minmax is not None:
        ax.set_xlim(x_minmax[0]-x_margin,x_minmax[-1]+x_margin)
    if despine:
        sns.despine(ax=ax, offset=0, trim=True)

    # Set tickmarks and labels
    if x_ticklabels is not None:
        plt.xticks( x_ticks, x_ticklabels, rotation=x_tick_rotation, fontsize=tick_size )
    elif x_minmax is not None and x_step is not None:
        plt.xticks( np.arange(x_minmax[0],x_minmax[1]+0.0000001,x_step[0]), suck_on_that_0point0(x_minmax[0], x_minmax[1]+0.0000001, step=x_step[0], format_depth=x_step[1]), rotation=x_tick_rotation, fontsize=tick_size )
    else:
        ticks, _ = plt.xticks()
        plt.xticks(ticks, ticks, fontsize=tick_size)

    if y_ticklabels is not None:
        plt.yticks( y_ticks, y_ticklabels, fontsize=tick_size )
    elif y_minmax is not None and y_step is not None:
        plt.yticks( np.arange(y_minmax[0],y_minmax[1]+0.0000001,y_step[0]), suck_on_that_0point0(y_minmax[0], y_minmax[1]+0.0000001, step=y_step[0], format_depth=y_step[1]), rotation=0, fontsize=tick_size )
    else:
        ticks, _ = plt.yticks()
        plt.yticks(ticks, ticks, fontsize=tick_size)

    ax.tick_params(length=3)

    # Set spine limits
    if y_minmax is not None:
        ax.spines['left'].set_bounds( y_minmax[0]-y_axis_margin, y_minmax[1]+y_axis_margin )
    if x_minmax is not None:
        ax.spines['bottom'].set_bounds( x_minmax[0]-x_axis_margin, x_minmax[1]+x_axis_margin )

    # Add title and legend
    if title != "":
        plt.title(title, fontsize=title_size)
    if ylabel != "":
        plt.ylabel(ylabel, fontsize=label_size)
    if xlabel != "":
        plt.xlabel(xlabel, fontsize=label_size)
    if legend == "on":
        lgnd = plt.legend(loc=legendpos, fontsize=legend_size, ncol=1, frameon=True)
        lgnd.get_frame().set_facecolor('#ffffff')

def suck_on_that_0point0( start, stop, step=1, format_depth=1 ):
    values = []
    for i in np.arange( start, stop, step ):
        if i == 0:
            values.append('0')
        else:
            values.append('{:0.{dpt}f}'.format(i,dpt=format_depth))
    return values

def plot_psth_in_grid( gx, gy, x, y, e=None, x_scale=1, y_scale=1,
                    color="#000000" ):
    plt.plot( (gx*x_scale)+x, (gy*y_scale)+y, color=color, linewidth=1 )
    if e is not None:
        plt.fill_between( (gx*x_scale)+x, (gy*y_scale)+(y-e), (gy*y_scale)+(y+e), facecolor=color, alpha=0.4, linewidth=0 )
    return (gx*x_scale)+x[0], (gx*x_scale)+x[-1]

def plot_tuning_curve( xvalues, x_by_trial_data, errorbars="SEM", color="#000000" ):
    if len(x_by_trial_data.shape) == 1:
        mean_curve = x_by_trial_data.ravel()
    else:
        mean_curve,sem_curve,_ = statstools.mean_sem(x_by_trial_data,axis=1)
        if errorbars.lower() == "sem":
            plt.fill_between( xvalues, mean_curve-sem_curve, mean_curve+sem_curve, facecolor=color, alpha=0.4, linewidth=0 )
        if errorbars.lower() == "indiv":
            for t in range(len(x_by_trial_data.shape[0])):
                plt.plot( xvalues, x_by_trial_data[t,:], color=color, alpha=0.5, linewidth=0.5 )
    plt.plot( xvalues, mean_curve, color=color, linewidth=1 )

def plot_curve( xvalues, mean, stderr=None, color="#000000", label=None ):
    if stderr is not None:
        plt.fill_between( xvalues, mean-stderr, mean+stderr, facecolor=color, alpha=0.4, linewidth=0 )
    if label is not None:
        plt.plot( xvalues, mean, color=color, linewidth=1, label=label )
    else:
        plt.plot( xvalues, mean, color=color, linewidth=1 )

def line( x, y, e=None, line_color='#000000', line_width=1, sem_color=None, shaded=False, top_bar_width=0.2, label=None, linestyle="-" ):
    if e is not None:
        if shaded:
            if sem_color is None:
                sem_color = line_color
            plt.fill_between( x, y-e, y+e, facecolor=sem_color, alpha=0.4, linewidth=0 )
        else:
            if sem_color is None:
                sem_color = '#000000'
            for xx,yy,ee in zip(x,y,e):
                plt.plot( [xx,xx], [yy-ee,yy+ee], color=sem_color, linewidth=1, linestyle=linestyle )
                plt.plot( [xx-top_bar_width,xx+top_bar_width], [yy-ee,yy-ee], color=sem_color, linewidth=1, linestyle=linestyle )
                plt.plot( [xx-top_bar_width,xx+top_bar_width], [yy+ee,yy+ee], color=sem_color, linewidth=1, linestyle=linestyle )
    if label is None:
        plt.plot( x, y, color=line_color, linewidth=line_width, linestyle=linestyle )
    else:
        plt.plot( x, y, color=line_color, linewidth=line_width, linestyle=linestyle, label=label )

def line_y( x, y, e=None, line_color='#000000', line_width=1, sem_color=None, shaded=False, top_bar_width=0.2, linestyle="-" ):
    if e is not None:
        if shaded:
            if sem_color is None:
                sem_color = line_color
            plt.fill_betweenx( y, x-e, x+e, facecolor=sem_color, alpha=0.4, linewidth=0 )
        else:
            if sem_color is None:
                sem_color = '#000000'
            for xx,yy,ee in zip(x,y,e):
                plt.plot( [xx-ee,xx+ee], [yy,yy], color=sem_color, linewidth=1 )
                plt.plot( [xx-ee,xx-ee],[yy-top_bar_width,yy+top_bar_width], color=sem_color, linewidth=1 )
                plt.plot( [xx+ee,xx+ee], [yy-top_bar_width,yy+top_bar_width], color=sem_color, linewidth=1 )
    plt.plot( x, y, color=line_color, linewidth=line_width, linestyle=linestyle )
    
def bar( x, y, e, width=0.8, edge="off", bar_color='#000000', sem_color='#000000', label=None, bottom=0, error_width=0.5 ):
    error_halfwidth = 0.5 * error_width * width
    if type(e) is list or type(e) is tuple:
        # Two sided confidence interval
        plt.plot( [x,x], [e[0]+bottom,e[1]+bottom], color=sem_color, linewidth=1 )
        plt.plot( [x-error_halfwidth,x+error_halfwidth], [e[1]+bottom,e[1]+bottom], color=sem_color, linewidth=1 )
        plt.plot( [x-error_halfwidth,x+error_halfwidth], [e[0]+bottom,e[0]+bottom], color=sem_color, linewidth=1 )
    elif e > 0:
        # One sided errorbars
        ye = y+bottom
        if y < 0:
            plt.plot( [x,x], [ye,ye-e], color=sem_color, linewidth=1 )
            plt.plot( [x-error_halfwidth,x+error_halfwidth], [ye-e,ye-e], color=sem_color, linewidth=1 )
        else:
            plt.plot( [x,x], [ye,ye+e], color=sem_color, linewidth=1 )
            plt.plot( [x-error_halfwidth,x+error_halfwidth], [ye+e,ye+e], color=sem_color, linewidth=1 )
    edgecolor,lw = (sem_color,1) if "on" in edge.lower() else ('None',0)
    if label is None:
        plt.bar( x-(0.5*width), y, width, color=bar_color, edgecolor=edgecolor, linewidth=lw, align='edge', bottom=bottom )
    else:
        plt.bar( x-(0.5*width), y, width, color=bar_color, edgecolor=edgecolor, linewidth=lw, align='edge', label=label, bottom=bottom )

def plot_psth_grid( ax, xvalues, psth, bs, y_scale, x_labels, y_labels, scalebar=None ):
    """ Plots an entire grid with psths at the according places """
    x_scale = 1.2*(xvalues[-1]-xvalues[0])
    # print("psth-shape: {}".format(psth.shape))
    n_rows,n_cols,n_trials,_n_frames = psth.shape
    x_plotted = []
    x_lims = []
    for row in range(n_rows):
        for col in range(n_cols):
            mean_curve,sem_curve,_ = statstools.mean_sem(psth[row,col,:,:],axis=0)
            if bs is not None:
                mean_curve = mean_curve - np.mean(bs[row,col,:])
            plotxlims = plot_psth_in_grid( gx=col, gy=row, x=xvalues, y=mean_curve, e=sem_curve, x_scale=x_scale, y_scale=y_scale, color="#000000" )
            x_plotted.append(col)
            x_lims += plotxlims

    # x labels
    x_left_ticklabels = (min(x_plotted)*x_scale)+xvalues[0]-(0.2*x_scale)
    for y in range(n_rows):
        plt.text(x_left_ticklabels, y*y_scale, "{}".format(y_labels[y]),
            rotation=0, ha='right', va='center', size=6, color='#000000' )

    # y labels
    y_bottom_ticklabels = -0.2*y_scale
    for x in range(n_cols):
        if x >= min(x_plotted) and x <= max(x_plotted):
            plt.text((x*x_scale)+np.median(xvalues), y_bottom_ticklabels,
                "{}".format(x_labels[x]), rotation=0,
                ha='center', va='top', size=6, color='#000000' )

    # Scalebar
    if scalebar is not None:
        plt.plot( (max(x_lims)+0.5*x_scale, max(x_lims)+0.5*x_scale), (0, scalebar), linewidth=1, color="#000000" )

    # Limits
    ymin = -0.35*y_scale
    ymax = n_rows*y_scale
    xmin = min(x_lims)-0.5*x_scale
    xmax = max(x_lims)+0.6*x_scale
    ax.set_ylim(ymin,ymax)
    ax.set_xlim(xmin,xmax)
    return xmin,xmax,ymin,ymax
