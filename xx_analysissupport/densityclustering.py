#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Script custom implements the fast search and clustering by density peaks
algorithm

Created on Sunday 1 Aug 2021

@author: pgoltstein
"""

import numpy as np
import sklearn.metrics
import matplotlib.pyplot as plt

def find_clusters(X, fraction=0.05, rho_min=0.2, delta_min=0.2, weights=None, rho_x_delta_min=None, show_rho_vs_delta=False, quiet=False):
    """ runs the cluster detection from start to end

        Inputs
        - X: data matrix X (samples X features)
        - fraction: fraction of data to include for distance calculation
        - rho_min: minimum rho for cluster to be included
        - weights: value to scale the density contribution of each datapoint, e.g. ocular dominance index
        - rho_x_delta_min: minimum product of rho and delta to be included as cluster, if value given, it supersedes rho_min and delta_min
        - delta_min: minimum delta for cluster to be included

        Outputs
        - clusters: list that holds dictionaries with cluster stats
            [cluster = {'X':#, 'Y':#, 'rho':#, 'delta':#}, ...]
    """

    # Output dictionary
    clusters = []

    # Cluster detection cascade
    D = distance_matrix(X, quiet=quiet)
    d_c = estimate_d_c(D,fraction)
    if weights is not None:
        rho = weighed_local_density(D, d_c,weights=weights,normalize=True)
    else:
        rho = local_density(D, d_c,normalize=True)
    delta, nearest = distance_to_larger_density(D, rho,normalize=True)
    centers = cluster_centers(rho, delta, rho_min=rho_min, delta_min=delta_min, rho_x_delta_min=rho_x_delta_min)

    # Output nicely :-)
    for c in centers:
        clusters.append({'X': X[c,0], 'Y': X[c,1], 'rho': rho[c], 'delta': delta[c]})

    # Show scatter plot of rho vs delta
    if show_rho_vs_delta:
        plt.subplots()
        if rho_x_delta_min is not None:
            x = np.arange(0,1,0.01)
            y = np.arange(0,1,0.01)
            xx,yy = np.meshgrid(x, y)
            xy = 1.0 * ( (xx * yy) > rho_x_delta_min )
            plt.contour(x,y,xy,0, linestyles=":")
        else:
            plt.plot([rho_min,rho_min],[0,1],"r:")
            plt.plot([0,1],[delta_min,delta_min], "b:")
        plt.scatter(rho,delta)

    return clusters

def value_per_shell(X, Y, data_var, clusters, bin_size=50, start=0, end=500):
    """ Calculates the mean value of variable data_var per distance bin from cluster centers

    Parameters
    ----------
    X : np.array
        x-coordinate per data point
    Y : np.array
        y-coordinate per data point (array)
    data_var : np.array
        variable value per data point (array)
    clusters : list
        output of the find_clusters function
    bin_size : int
        size of the shells
    start : int
        distance to start, default is 0 micron
    end : int
        distance to end, default is 500 micron

    returns
    ----------
    bin_values : np.arrays
        matrix with data values [clusters,bins]

    """

    # Prepare output variables
    bins = np.arange(start,end,bin_size,dtype=float)
    n_bins = len(bins)-1
    bin_values = np.zeros( (len(clusters),n_bins) )
    xvalues = bins[:-1] + (0.5*bin_size)

    # Loop clusters
    for c_nr, c in enumerate(clusters):

        # Find distance of neurons to cluster center
        D = np.sqrt( (X-c["X"])**2 + (Y-c["Y"])**2 )

        # Loop shell-bins and get mean value per shell-bin
        for b_nr in range(n_bins):
            include_ix = np.argwhere( np.logical_and( D>=bins[b_nr], D<bins[b_nr+1] ) ).ravel()
            bin_values[c_nr,b_nr] = np.nanmean(data_var[include_ix])

    # return
    return bin_values, xvalues

def count_per_shell(X, Y, clusters, bin_size=50, start=0, end=500):
    """ Calculates the number of datapoints per distance bin from cluster centers

    Parameters
    ----------
    X : np.array
        x-coordinate per data point
    Y : np.array
        y-coordinate per data point (array)
    clusters : list
        output of the find_clusters function
    bin_size : int
        size of the shells
    start : int
        distance to start, default is 0 micron
    end : int
        distance to end, default is 500 micron

    returns
    ----------
    bin_values : np.arrays
        matrix with data values [clusters,bins]

    """

    # Prepare output variables
    bins = np.arange(start,end,bin_size,dtype=float)
    n_bins = len(bins)-1
    bin_counts = np.zeros( (len(clusters),n_bins) )
    xvalues = bins[:-1] + (0.5*bin_size)

    # Loop clusters
    for c_nr, c in enumerate(clusters):

        # Find distance of neurons to cluster center
        D = np.sqrt( (X-c["X"])**2 + (Y-c["Y"])**2 )

        # Loop shell-bins and get mean value per shell-bin
        for b_nr in range(n_bins):
            bin_counts[c_nr,b_nr] = np.nansum( np.logical_and( D>=bins[b_nr], D<bins[b_nr+1] ) * 1.0)

    # return
    return bin_counts, xvalues

def distance_matrix(X, quiet=False):
    """ calculates distance between all entries in data matrix X (samples X features)"""
    if not quiet:
        print("Calculating pairwise distance matrix for {} samples".format(X.shape[0]))
    D = sklearn.metrics.pairwise_distances(X, metric="euclidean")
    np.fill_diagonal(D, np.NaN)
    return D

def estimate_d_c(D, fraction):
    """ estimates the d_c value, cut-off for distance """
    d_array = []
    for s in range(D.shape[0]):
        d_array.append(D[s,s+1:])
    d_array = np.concatenate(d_array,axis=0)
    # print("d_array.shape={}".format(d_array.shape))
    d_c = np.percentile(d_array, fraction*100)
    # print("d_c={}".format(d_c))
    return d_c

def weighed_local_density(D, d_c, weights, normalize=False):
    """ Calculates rho, the local density

        Inputs
        - D: pairwise distance matrix D (samples X samples)
        - d_c: cut-off value
        - weights: value to scale the density contribution of each datapoint, e.g. ocular dominance index
        - normalize: whether or not to normalize to max rho=1

        Output
        - rho: list of local density (rho) per data point
    """

    # Apply cuttoff
    D_cuttoff = D<d_c
    # print("D: \n{}".format(D_cuttoff[:3,:3]))

    # Some fancy weighing magic ... (see matlab script by author, they refer to this as a "Gaussian Kernel")
    rho = np.zeros((D.shape[0],))
    for s in range(len(rho)):

        # All distances to cells within the neighborhood d_c
        neighborhood_vector = D[s,D_cuttoff[s,:]]
        neighborhood_weights = weights[D_cuttoff[s,:]]

        # in the next line, smaller values in the neighborhood_vector result in larger values contributing to rho
        neighborhood_vector = np.exp(-(neighborhood_vector / d_c)**2)

        # Now we will scale the neighborhood by our weight
        neighborhood_vector = neighborhood_vector * neighborhood_weights

        # Finally, we sum all the scaled weights to a local weighed density
        rho[s] = np.sum(neighborhood_vector)

    # Normalize
    if normalize:
        rho = rho / np.max(rho)
        # rho = (rho-np.min(rho)) / (np.max(rho)-np.min(rho))

    # Calculate and return the local density vector
    return rho


def local_density(D, d_c, normalize=False):
    """ Calculates rho, the local density

        Inputs
        - D: pairwise distance matrix D (samples X samples)
        - d_c: cut-off value
        - normalize: whether or not to normalize to max rho=1

        Output
        - rho: list of local density (rho) per data point
    """

    # Apply cuttoff
    D_cuttoff = D<d_c
    # print("D: \n{}".format(D_cuttoff[:3,:3]))

    # Some fancy weighing magic ... (see matlab script by author, they refer to this as a "Gaussian Kernel")
    rho = np.zeros((D.shape[0],))
    for s in range(len(rho)):
        rho[s] = np.sum(np.exp(-(D[s,D_cuttoff[s,:]] / d_c)**2))

    # Normalize
    if normalize:
        rho = rho / np.max(rho)

    # Calculate and return the local density vector
    return rho


def distance_to_larger_density(D, rho, normalize=False):
    """ Calculates delta, the distance to the closest point with a higher local density

        Inputs
        - D: pairwise distance matrix D (samples X samples)
        - rho: local density per data point
        - normalize: whether or not to normalize to max delta=1

        Output
        - delta: list of distance to point with higher local density
    """
    # Output vector
    delta = np.zeros_like(rho)
    nearest = np.zeros_like(rho, dtype=np.int64)

    # Loop samples
    for s in range(len(rho)):

        # Find all samples with higher density
        nearby_ix = rho>rho[s]

        # If no samples having a larger density, set to maximum distance
        if np.sum(nearby_ix*1.0) == 0:
            delta[s] = np.nanmax(D[s,:])

        # Else set to minimum distance a point with higher density
        else:
            d_vec = D[s,:]
            d_vec[nearby_ix==False] = np.nanmax(d_vec)
            nearestby_ix = np.argmin(d_vec)
            delta[s] = D[s,nearestby_ix]
            nearest[s] = nearestby_ix

    # Normalize
    if normalize:
        delta = delta / np.max(delta)

    return delta, nearest

def cluster_centers(rho, delta, rho_min=0.2, delta_min=0.2, rho_x_delta_min=None):
    """ returns the indices of the cluster centers """
    if rho_x_delta_min is None:
        centers = np.argwhere( np.logical_and( rho>rho_min, delta>delta_min ) ).ravel()
    else:
        rho_x_delta = rho*delta
        centers = np.argwhere( rho_x_delta>rho_x_delta_min ).ravel()

    center_rho_sort_ix = np.argsort(rho[centers])
    return centers[center_rho_sort_ix[::-1]]

def assign_cluster_id(rho, nearest, centers):
    """ assigns cluster ID's to each point """
    order = np.argsort(rho)[::-1]
    ids = np.zeros_like(rho, dtype=np.int64)
    for s in range(len(centers)):
        ids[centers[s]] = s
    for s in range(len(rho)):
        if order[s] not in centers:
            ids[order[s]] = ids[nearest[order[s]]]
    return ids

def core(D, d_c, rho, ids):
    """ returns a list with all samples are part of the cluster core """
    avg_border_rho = np.zeros( (len(np.unique(ids)),) )
    core = np.zeros_like(rho, dtype=bool)

    # Get matrix indicating only nearby samples by True
    D_cuttoff = D<d_c

    # Loop samples
    for s1 in range(len(rho)-1):
        for s2 in range(s1+1,len(rho)):

            # Find rho of all nearby sample without the same id (border sample)
            if ids[s1] != ids[s2] and D_cuttoff[s1,s2]:
                avg_density = 0.5 * (rho[s1] + rho[s2])
                if avg_density > avg_border_rho[ids[s1]]:
                    avg_border_rho[ids[s1]] = avg_density
                if avg_density > avg_border_rho[ids[s2]]:
                    avg_border_rho[ids[s2]] = avg_density

    for s in range(len(rho)):
        if rho[s] > avg_border_rho[ids[s]]:
            core[s] = True

    return core
