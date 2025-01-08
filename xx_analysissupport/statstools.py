#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 4, 2019

@author: pgoltstein
"""

#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Imports

import numpy as np
import scipy.stats as scistats
import statsmodels.api as statsmodels


#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Descriptives

def mean_sem( datamat, axis=0 ):
    mean = np.nanmean(datamat,axis=axis)
    n = np.sum( ~np.isnan( datamat ), axis=axis )
    sem = np.nanstd( datamat, axis=axis ) / np.sqrt( n )
    return mean,sem,n

def report_mean(sample1, sample2):
    print("  Group 1, Mean (SEM) = {} ({}) n={}".format(*mean_sem(sample1.ravel())))
    print("  Group 2, Mean (SEM) = {} ({}) n={}".format(*mean_sem(sample2.ravel())))


#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Functions for reporting statistical tests

def report_chisquare_test( sample1, sample2, n_indents=2, alpha=0.05, bonferroni=1  ):
    p,t,df,n = chisquared_test( sample1, sample2 )
    p_b = p*bonferroni
    print('{}one-sided Chisquare test, X^2({:0.0f},N={:0.0f})={:0.3f}, p={}{}'.format( " "*n_indents, df, n, t, p, "  >> sig." if p<(alpha/bonferroni) else "." ))
    return p_b

def report_paired_ttest( sample1, sample2, n_indents=2, alpha=0.05, bonferroni=1 ):
    p,t,df,d,n = paired_ttest( sample1, sample2 )
    p_b = p*bonferroni
    print('{}Paired t-test, t({:0.0f})={:0.3f}, p={:4E}, p(b)={:4E}, d={:0.3f}, n={:0.0f}{}'.format( " "*n_indents, df, t, p, p_b, d, n, "  >> SIGN !!" if p<(alpha/bonferroni) else "." ))

def report_ttest( sample1, sample2, equal_var=True, n_indents=2, alpha=0.05, bonferroni=1 ):
    p,t,df,d,n1,n2 = ttest( sample1, sample2, equal_var=equal_var )
    p_b = p*bonferroni
    print('{}t-test, t({:0.0f})={:0.3f}, p={:4E}, p(b)={:4E}, d={:0.3f}, n1={:0.0f}, n2={:0.0f}{}'.format( " "*n_indents, df, t, p, p_b, d, n1, n2, "  >> SIGN !!" if p<(alpha/bonferroni) else "." ))

def report_wmpsr_test( sample1, sample2, n_indents=2, alpha=0.05, alternative="two-sided", bonferroni=1, preceding_text=""):
    p,Z,n = wilcoxon_matched_pairs_signed_rank_test( sample1, sample2, alternative=alternative )
    if alternative=="two-sided":
        preceding_text += "two-sided "
    else:
        preceding_text="one-sided "
    if bonferroni>1:
        p_b = p*bonferroni
        if p_b < 0.001:
            print('{}{}WMPSR test, W={:0.0f}, p_bonf={:4E}, n={:0.0f}{}'.format( " "*n_indents, preceding_text, Z, p_b, n, "  >> sig." if p<(alpha/bonferroni) else "." ))
        else:
            print('{}{}WMPSR test, W={:0.0f}, p_bonf={:0.4f}, n={:0.0f}{}'.format( " "*n_indents, preceding_text, Z, p_b, n, "  >> sig." if p<(alpha/bonferroni) else "." ))
        return p_b
    else:
        if p < 0.001:
            print('{}{}WMPSR test, W={:0.0f}, p={:4E}, n={:0.0f}{}'.format( " "*n_indents, preceding_text, Z, p, n, "  >> sig." if p<(alpha/bonferroni) else "." ))
        else:
            print('{}{}WMPSR test, W={:0.0f}, p={:0.4f}, n={:0.0f}{}'.format( " "*n_indents, preceding_text, Z, p, n, "  >> sig." if p<(alpha/bonferroni) else "." ))
        return p

def report_mannwhitneyu_test( sample1, sample2, n_indents=2, alpha=0.05, bonferroni=1, preceding_text="" ):
    p,U,r,n1,n2 = mann_whitney_u_test( sample1, sample2 )
    if bonferroni>1:
        p_b = p*bonferroni
        if p_b < 0.001:
            print('{}{}two-sided Mann-Whitney U test, U={:0.0f}, p_bonf={:4E}, r={:0.3f}, n1={:0.0f}, n2={:0.0f}{}'.format( " "*n_indents, preceding_text, U, p_b, r, n1, n2, "  >> sig." if p<(alpha/bonferroni) else "." ))
        else:
            print('{}{}two-sided Mann-Whitney U test, U={:0.0f}, p_bonf={:0.4f}, r={:0.3f}, n1={:0.0f}, n2={:0.0f}{}'.format( " "*n_indents, preceding_text, U, p_b, r, n1, n2, "  >> sig." if p<(alpha/bonferroni) else "." ))
        return p_b
    else:
        if p < 0.001:
            print('{}{}two-sided Mann-Whitney U test, U={:0.0f}, p={:4E}, r={:0.3f}, n1={:0.0f}, n2={:0.0f}{}'.format( " "*n_indents, preceding_text, U, p, r, n1, n2, "  >> sig." if p<(alpha/bonferroni) else "." ))
        else:
            print('{}{}two-sided Mann-Whitney U test, U={:0.0f}, p={:0.4f}, r={:0.3f}, n1={:0.0f}, n2={:0.0f}{}'.format( " "*n_indents, preceding_text, U, p, r, n1, n2, "  >> sig." if p<(alpha/bonferroni) else "." ))
        return p

def report_kruskalwallis( samplelist, n_indents=2, alpha=0.05, bonferroni=1, preceding_text="" ):
    p,H,DFbetween,DFwithin,n = kruskalwallis( samplelist )
    if bonferroni>1:
        p_b = p*bonferroni
        if p_b < 0.001:
            print("{}{}two-sided Kruskal-Wallis test, H({:0.0f}) = {:0.3f}, p = {:4E}, n={:0.0f}{}".format( " "*n_indents, preceding_text, DFbetween, H, p_b, n, "  >> sig." if p<(alpha/bonferroni) else "." ))
        else:
            print("{}{}two-sided Kruskal-Wallis test, H({:0.0f}) = {:0.3f}, p = {:0.4f}, n={:0.0f}{}".format( " "*n_indents, preceding_text, DFbetween, H, p_b, n, "  >> sig." if p<(alpha/bonferroni) else "." ))
        return p_b
    else:
        if p < 0.001:
            print("{}{}two-sided Kruskal-Wallis test, H({:0.0f}) = {:0.3f}, p = {:4E}, n={:0.0f}{}".format( " "*n_indents, preceding_text, DFbetween, H, p, n, "  >> sig." if p<(alpha/bonferroni) else "." ))
        else:
            print("{}{}two-sided Kruskal-Wallis test, H({:0.0f}) = {:0.3f}, p = {:0.4f}, n={:0.0f}{}".format( " "*n_indents, preceding_text, DFbetween, H, p, n, "  >> sig." if p<(alpha/bonferroni) else "." ))
        return p

def report_anova1way( samplemat, n_indents=2, alpha=0.05 ):
    DFbetween,DFwithin, F, p, eta_sqrd,n = anova1way( samplemat )
    print("{}ANOVA, F({:0.0f},{:0.0f}) = {:0.3f}, p = {}, eta^2={:0.3f}, n={:0.0f}".format( " "*n_indents, DFbetween, DFwithin, F, p, eta_sqrd, n, "  >> SIGN !!" if p<alpha else "." ))


#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Functions for performing statistical tests

def chisquared_test( sample1, sample2 ):
    if len(np.unique(sample1)) > 2 or len(np.unique(sample2)) > 2:
        print("Only two samples, and boolean data allowed for chi square test")
        return 1.0,np.NaN,np.NaN,0
    sample1 = sample1[~np.isnan(sample1)].ravel()
    sample2 = sample2[~np.isnan(sample2)].ravel()
    n_categories = 2
    n_groups = 2
    df = (n_categories-1)*(n_groups-1)
    n1 = len(sample1)
    n2 = len(sample2)
    n = n1 + n2
    frequency_samples = np.array([np.sum(sample1==1),np.sum(sample2==1)])
    n_samples = np.array([n1,n2])
    chisq,p,(f_real,f_expected) = statsmodels.stats.proportions_chisquare( count=frequency_samples, nobs=n_samples )
    return p,chisq,df,n

def paired_ttest( sample1, sample2 ):
    matched_not_nan = np.logical_and(~np.isnan(sample1), ~np.isnan(sample2))
    sample1 = sample1[matched_not_nan].ravel()
    sample2 = sample2[matched_not_nan].ravel()
    t,p = scistats.ttest_rel( sample1, sample2 )
    df = sample2.shape[0]-1
    z = sample1 - sample2
    d = np.mean(z) / np.std(z)
    n = len(sample1)
    return p,t,df,d,n

def ttest( sample1, sample2, equal_var=True ):
    sample1 = sample1[~np.isnan(sample1)].ravel()
    sample2 = sample2[~np.isnan(sample2)].ravel()
    t,p = scistats.ttest_ind( sample1, sample2, equal_var=equal_var )
    n1 = sample1.shape[0]
    n2 = sample2.shape[0]
    df = n1 + n2 - 2
    d = (np.mean(sample1) - np.mean(sample2)) / np.sqrt( ( ((n1-1) * (np.std(sample1)**2)) + ((n2-1) * (np.std(sample2)**2)) ) / df)
    return p,t,df,d,n1,n2

def wilcoxon_matched_pairs_signed_rank_test( sample1, sample2, alternative="two-sided" ):
    matched_not_nan = np.logical_and(~np.isnan(sample1), ~np.isnan(sample2))
    sample1 = sample1[matched_not_nan].ravel()
    sample2 = sample2[matched_not_nan].ravel()
    if np.count_nonzero(sample1)==0 and np.count_nonzero(sample2)==0:
        return 1.0,np.NaN,np.NaN
    else:
        Z,p = scistats.wilcoxon(sample1, sample2, alternative=alternative)
        n = len(sample1)
        return p,Z,n

def mann_whitney_u_test( sample1, sample2 ):
    sample1 = sample1[~np.isnan(sample1)].ravel()
    sample2 = sample2[~np.isnan(sample2)].ravel()
    U,p = scistats.mannwhitneyu(sample1, sample2)
    n1 = len(sample1)
    n2 = len(sample2)
    r = U / np.sqrt(n1+n2)
    return p,U,r,n1,n2

def kruskalwallis( samplelist ):
    # Clean up sample list and calculate N
    N = 0
    no_nan_samplelist = []
    for b in range(len(samplelist)):
        no_nan_samples = samplelist[b][~np.isnan(samplelist[b])]
        if len(no_nan_samples) > 0:
            no_nan_samplelist.append(no_nan_samples)
            N += len(no_nan_samples)

    # Calculate degrees of freedom
    k = len(samplelist)
    DFbetween = k - 1
    DFwithin = N - k
    DFtotal = N - 1
    H,p = scistats.kruskal( *no_nan_samplelist )
    return p,H,DFbetween,DFwithin,N

def anova1way( samplemat ):
    """ runs 1way anova on groups separated in different columns """

    # Get number of samples
    n,k = samplemat.shape
    N = n*k

    # Calculate degrees of freedom
    DFbetween = k - 1
    DFwithin = N - k
    DFtotal = N - 1

    # Get mean per condition and overall
    y = np.nanmean(samplemat)
    y_i = np.nanmean(samplemat,axis=0)

    # Calculate sum of squares
    SSbetween = n * np.nansum( (y_i-y)**2 )
    SSwithin = np.nansum([ np.nansum( (samplemat[:,i]-y_i[i])**2 ) for i in range(k) ])
    SStotal = np.nansum((samplemat-y)**2)

    # Calculate mean squares
    MSbetween = SSbetween/DFbetween
    MSwithin = SSwithin/DFwithin

    # Calculate F and get p
    F = MSbetween/MSwithin
    p = scistats.f.sf(F, DFbetween, DFwithin)

    # Calculate eta-squared and omega-squared
    eta_sqrd = SSbetween/SStotal
    om_sqrd = (SSbetween - (DFbetween * MSwithin))/(SStotal + MSwithin)

    return DFbetween,DFwithin, F, p, eta_sqrd, n


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Class for handling bootstrapped statistics

class Bootstrapped(object):
    """
    This class provides bootstrapped statistics for the data sample that is supplied to the initialization routine.
    """

    def __init__(self, data, axis=0, n_samples=1000, sample_size=None, multicompare=1):
        """
        Initialize the Bootstrapped object with the provided data and parameters.

        Parameters:
        data (np.ndarray): The data sample to bootstrap.
        axis (int): The axis along which to bootstrap. Default is 0.
        n_samples (int): The number of bootstrap samples to generate. Default is 1000.
        sample_size (int or None): The size of each bootstrap sample. Default is None, which uses the size of the data.
        multicompare (int): The number of comparisons for multiple comparison correction (affects only confidence intervals). Default is 1.
        """

        # Get shape of output matrices
        shape = data.shape
        n = shape[axis]
        self._shape = tuple(np.delete(shape, axis))

        # Check if data is empty
        if len(data.shape) == 1 and data.shape[0] == 0:
            print("Warning: No data was supplied, bootstrap results in NaN's")
            self._mean = np.NaN
            self._stderr = np.NaN
            self._upper95 = np.NaN
            self._lower95 = np.NaN
            self._upper99 = np.NaN
            self._lower99 = np.NaN

        else:

            # If no sample size supplied, set it to the size of data
            if sample_size is None:
                sample_size = n

            # Get the bootstrap samples
            bootstraps = []
            for r in range(n_samples):
                random_sample = np.random.choice(n, size=sample_size, replace=True)
                bootstraps.append( np.nanmean( np.take(data, random_sample, axis=axis), axis=axis ) )
            bootstraps = np.stack(bootstraps,axis=0)

            # Correct thresholds for multiple comparison (Bonferroni)
            low95 = 2.5/multicompare
            up95 = 100.0 - low95
            low99 = 0.5/multicompare
            up99 = 100.0 - low95

            # Calculate the statistics
            self._mean = np.nanmean(bootstraps,axis=0)
            self._stderr = np.nanstd(bootstraps,axis=0)
            self._upper95 = np.nanpercentile(bootstraps,up95,axis=0)
            self._lower95 = np.nanpercentile(bootstraps,low95,axis=0)
            self._upper99 = np.nanpercentile(bootstraps,up99,axis=0)
            self._lower99 = np.nanpercentile(bootstraps,low99,axis=0)

    @property
    def shape(self):
        """ Return the shape of the output matrices. """
        return self._shape

    @property
    def mean(self):
        """ Return the mean of the bootstrapped samples. """
        return self._mean

    @property
    def stderr(self):
        """ Return the standard error of the bootstrapped samples. """
        return self._stderr

    @property
    def ci95(self):
        """ Return the 95% confidence interval of the bootstrapped samples. """
        return self._lower95,self._upper95

    @property
    def ci99(self):
        """ Return the 99% confidence interval of the bootstrapped samples. """
        return self._lower99,self._upper99
