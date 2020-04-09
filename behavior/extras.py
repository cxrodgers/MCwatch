"""Various performance metrics and plotting

"""
from __future__ import print_function
from __future__ import division
from builtins import range
from past.utils import old_div
import pandas
import my
import numpy as np
import matplotlib.pyplot as plt
import os


def calculate_perf_from_trial_matrix(tm, groupby=None, 
    random_hits_and_errors_only=True,):
    """Calculate perf from trial matrix split various ways
    
    This replaces the other versions in this module and in TrialMatrix.
    
    tm : trial_matrix with columns 'outcome' and whatever is in 'groupby'
    
    groupby : column names to group by
        Default: ['rewside']
    
    random_hits_and_errors_only : if True, discard all trials that aren't
        random hits or errors
    
    Returns: DataFrame
        index: MultiIndex with the entries in groupby
        columns: 'error', 'hit', 'n', 'perf', 'ci_h', 'ci_l'
    """
    tm = tm.copy()
    
    # Include only random hits and errors
    if random_hits_and_errors_only:
        tm = tm[tm['outcome'].isin(['hit', 'error']) & 
            tm.isrnd]
    if len(tm) <= 1:
        raise ValueError("not enough rows in tm")

    # Default groupby
    if groupby is None:
        groupby = ['rewside']
    else:
        groupby = list(groupby)
    
    # Group by whatever is requested, but then always outcome at the end
    if 'outcome' not in groupby:
        groupby.append('outcome')

    # Group, now 'error' and 'hit' are columns, and a MultiIndex with the
    # specified variables
    outcomedf = tm.groupby(groupby).apply(len).unstack('outcome')
    
    # Ensure all ints and no nans
    outcomedf[outcomedf.isnull()] = 0
    outcomedf = outcomedf.astype(np.int)
    
    # If no hits or no errors, insert zeros
    for column in ['hit', 'error']:
        if column not in outcomedf:
            outcomedf[column] = 0
    
    # Calculate perf
    outcomedf['n'] = outcomedf['hit'] + outcomedf['error']
    outcomedf['perf'] = old_div(outcomedf['hit'], outcomedf['n'])

    # Insert CIs
    outcomedf['ci_l'] = np.nan
    outcomedf['ci_h'] = np.nan
    for idx in outcomedf.index:
        ci_l, ci_h = my.stats.binom_confint(
            outcomedf.loc[idx, 'hit'],
            outcomedf.loc[idx, 'hit'] + outcomedf.loc[idx, 'error']
        )
        outcomedf.loc[idx, 'ci_l'] = ci_l
        outcomedf.loc[idx, 'ci_h'] = ci_h
    
    return outcomedf

def calculate_perf_by_number_of_contacts(tm, bins=None):
    """Bin the performance by number of contacts.
    
    The trial matrix must already have n_contacts inserted.
    
    All trials in tm are included, so pre-drop forced if desired.
    
    Returns: Series with index equal to number of contacts and
        values equal to performance for that "bin". Each "bin" includes
        all the trials with n_contacts that are less than the index and are
        not included in any lower bin.
    """
    if bins is None:
        max_contacts = tm.n_contacts.max()
        bins = np.array([v ** 2 for v in range(
            1 + int(np.ceil(np.sqrt(max_contacts))))])
    else:
        bins = np.asarray(bins)
    
    if np.any(tm.n_contacts > bins.max()):
        print("warning: dropping trials with more contacts than maximum bin")
        old_div(1,0)
    
    # This counts hits and errors for every value of n_contacts
    cdf = tm.groupby(
        'n_contacts').apply(
        lambda df: df.outcome.value_counts())
    
    # Edge case where no errors occurred means no unstacking necessary
    if 'error' not in tm.outcome.values:
        cdf['error'] = 0
    else:
        cdf = cdf.unstack()

    # Accumulate
    cumcdf = cdf.cumsum()
    
    # Reindex by all integer bins, forward filling when no data available,
    # and starting at zero if no data for 0 contats
    cumcdf = cumcdf.reindex(
        pandas.Index(list(range(bins.max() + 1)))).fillna(method='ffill').fillna(0)
    
    # Then index by desired bins and diff (to undo the cumulative)
    binned_cdf = cumcdf.ix[bins].diff().fillna(cumcdf.ix[0])
    binned_cdf = binned_cdf.astype(np.int)
    binned_cdf['perf'] = old_div(binned_cdf['hit'], binned_cdf.sum(1))    
    
    return binned_cdf

def histogram_number_of_contacts(n_contacts, bins=None):
    """Binned histogram of number of contacts"""
    if bins is None:
        max_contacts = n_contacts.max()
        bins = np.array([v ** 2 for v in range(
            1 + int(np.ceil(np.sqrt(max_contacts))))])
    else:
        bins = np.asarray(bins)
    
    if np.any(n_contacts > bins.max()):
        print("warning: dropping trials with more contacts than maximum bin")
        old_div(1,0)
    
    cdf = n_contacts.value_counts().sort_index()
    cumcdf = cdf.cumsum()
    # Reindex by all integer bins, forward filling when no data available,
    
    # and starting at zero if no data for 0 contats
    cumcdf = cumcdf.reindex(
        pandas.Index(list(range(bins.max() + 1)))).fillna(method='ffill').fillna(0)
    
    # Then index by desired bins and diff (to undo the cumulative)
    binned_cdf = cumcdf.ix[bins].diff().fillna(cumcdf.ix[0])
    binned_cdf = binned_cdf.astype(np.int)
    return binned_cdf
    

def calculate_perf_by_radius_distance_and_side(tm, outcome_column='outcome'):
    """Averages performance by radius and servo_pos from trial_matrix.
    
    tm : trial matrix with stepper_pos, outcome, isrnd, servo_pos, rewside,
        trial columns
    
    outcome_column : column name to grade the performance on
    
    Inserts a 'radius' column if one doesn't already exist.
    Includes only random hits and errors.
    
    Returns: perfdf
        index: radius, side
        columns: servo_pos
        values: performance
    """
    # Plot model performance vs distance
    tm = tm.copy()
    if 'radius' not in tm.columns:
        tm['radius'] = 'hard'
        tm.loc[tm.stepper_pos.isin([50, 150]), 'radius'] = 'easy'
    
    # Include only random hits and errors
    tm = tm[
        tm[outcome_column].isin(['hit', 'error']) & 
        tm.isrnd]

    # Count outcomes
    outcomedf = tm.groupby(
        ['radius', 'rewside', 'servo_pos'])[outcome_column].value_counts().unstack(
        ['outcome', 'servo_pos']).sort_index(axis=1)
    
    # Divide hits by totals
    if 'error' not in outcomedf:
        perfdf = outcomedf['hit'] / outcomedf['hit']
    else:
        perfdf = outcomedf['hit'] / (outcomedf['hit'] + outcomedf['error'])
        
    return perfdf

def calculate_perf_by_distance_with_cis(tm):
    """Calculate performance by distance"""
    # Include only random hits and errors
    tm = tm[
        tm.outcome.isin(['hit', 'error']) & 
        tm.isrnd].copy()
    
    outcomedf = tm.groupby('outcome').apply(
        lambda df: df.pivot_table(
            index='servo_pos', columns='rewside',
            values='trial', aggfunc='count')).T.stack()
    outcomedf[outcomedf.isnull()] = 0
    
    #~ perfdf = outcomedf['hit'] / (outcomedf['hit'] + outcomedf['error'])    
    for idx in outcomedf.index:
        ci_l, ci_h = my.stats.binom_confint(
            outcomedf.loc[idx, 'hit'],
            outcomedf.loc[idx, 'hit'] + outcomedf.loc[idx, 'error'])
        outcomedf.loc[idx, 'ci_l'] = ci_l
        outcomedf.loc[idx, 'ci_h'] = ci_h
        outcomedf.loc[idx, 'perf'] = old_div(outcomedf.loc[idx, 'hit'], \
            (outcomedf.loc[idx, 'hit'] + outcomedf.loc[idx, 'error']))
    
    return outcomedf
    

def plot_perf_by_radius_distance_and_side(perfdf, ax=None):
    if ax is None:
        f, ax = plt.subplots()
    
    rewside2color = {'left': 'blue', 'right': 'red'}
    radius2ls = {'hard': '--', 'easy': '-'}
    for rewside in ['left', 'right']:
        for radius in ['hard', 'easy']:
            
            sub = perfdf[rewside][radius]
            ax.plot(sub.index, sub.values, 
                color=rewside2color[rewside], ls=radius2ls[radius])
    
    ax.plot([1670, 1850], [.5, .5], 'k--')
    
    ax.set_xticks([1670, 1760, 1850])
    ax.set_xticklabels(['+5.4', '+2.7', 'closest'])
    ax.set_xlabel('stimulus position (mm)')
    
    # Reverse the order of the x-axis
    ax.set_xlim(ax.get_xlim()[::-1])
    
    ax.set_yticks((0, .25, .5, .75, 1))
    ax.set_ylim((-.1, 1.1))
    
    return ax


def plot_perf_by_distance_and_side(perfdf, ax=None):
    if ax is None:
        f, ax = plt.subplots()
    
    rewside2color = {'left': 'blue', 'right': 'red'}
    for rewside in ['left', 'right']:
        sub = perfdf[rewside]
        ax.plot(sub.index, sub.values, 
            color=rewside2color[rewside],)
    
    ax.plot([1690, 1850], [.5, .5], 'k--')
    
    ax.set_xticks([1670, 1760, 1850])
    ax.set_xticklabels(['+5.4', '+2.7', 'closest'])
    ax.set_xlabel('stimulus position (mm)')
    
    # Reverse the order of the x-axis
    ax.set_xlim(ax.get_xlim()[::-1])
    
    ax.set_yticks((0, .25, .5, .75, 1))
    ax.set_ylim((-.1, 1.1))
    
    return ax