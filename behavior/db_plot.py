"""Module for plotting diagnostics from db"""

import numpy as np
import pandas
import ArduFSM
import MCwatch
import my
import datetime
from ArduFSM import TrialMatrix, TrialSpeak, mainloop
import matplotlib.pyplot as plt
from my.plot import generate_colorbar
import networkx as nx
import glob
import os

def plot_by_training_stage():
    """Plot performance by stage of training for every mouse.
    
    Breaks the performance plot with vertical lines after every:
    * Scheduler change
    * Stimulus change
    * Whisker trim
    
    Note that lick training days will show up as partitions, but
    there is no performance data.
    
    Requires:
    * Internet access to google doc of whisker trims
    * runner django database to be in Dropbox
    
    Returns: list of Figure
    """
    # Get perf data partitioned by training stage
    session_table, change_table = \
        MCwatch.behavior.db.calculate_perf_by_training_stage()

    # Partition plot by mouse
    n_mouse_per_figure = 4
    f, axa = None, None
    fignum, axnum = 0, 0
    fig_l = []
    
    # Get mouse sorted by number
    import runner.models
    mouse_l = runner.models.Mouse.objects.filter(
        in_training=True).order_by('number').values_list('name', flat=True)

    # Plot each mouse in its own axis
    gobj = session_table.groupby('mouse')
    for mouse in mouse_l:
        try:
            msessions = gobj.get_group(mouse)
        except KeyError:
            # in training, but no data
            continue
        mchanges = change_table.ix[msessions.index]
        
        if f is None or axnum == n_mouse_per_figure:
            f, axa = plt.subplots(n_mouse_per_figure, 1,
                figsize=(8, 2.6*n_mouse_per_figure))
            f.subplots_adjust(bottom=.05, wspace=.2, hspace=.4, 
                left=.05, right=.95, top=.95)
            axnum = 0
            fignum += 1
            fig_l.append(f)

        ax = axa[axnum]
        axnum += 1
        plot_by_training_stage_one_mouse(msessions, mchanges, ax=ax)
        ax.set_title(mouse)

    return fig_l

def shorten_param_value(param, value):
    """Shorten the session parameter to display more nicely"""
    if param == 'stimulus_set':
        try:
            value = value[12:]
        except IndexError:
            pass            
        value = value.replace('CCL_', '')
        value = value.replace('2shapes', '2sh')
        value = value.replace('srvpos', 'pos')
        value = value.replace('_', '\n')
        
    
    elif param == 'scheduler':
        value = value.replace('ForcedAlternation', 'FA')
    
    elif param == 'trim':
        value = value.replace('; ', '')
    
    return value

def plot_by_training_stage_one_mouse(msessions, mchanges, ax=None,
    mouse=None, start_date=None, delta_days=60):
    """Plot single mouse's performance split by training stage.
    
    msessions : subset of session_table for a single mouse from 
        MCwatch.behavior.db.calculate_perf_by_training_stage
    mchanges : same but for change_table
    """
    if ax is None:
        f, ax = plt.subplots()
    
    if mouse is not None:
        msessions = msessions[msessions.mouse == mouse]
        mchanges = mchanges.ix[msessions.index]
    
    # Choose start date
    if start_date is None:
        start_date = (datetime.datetime.now() - 
            datetime.timedelta(days=delta_days))
    
    # Group by partition
    msessions2 = msessions.reset_index()
    mchanges2 = mchanges.reset_index().drop('index', 1)
    gobj = msessions2.groupby('partition')

    # xticks
    xt = np.array(list(range(len(msessions2))))
    xtl = msessions2['date_time_start'].apply(
        lambda dt: dt.strftime('%m-%d'))
    beyond_start_date = msessions2['date_time_start'] >= start_date
    try:
        start_idx = beyond_start_date.ix[beyond_start_date.values].index[0]
    except IndexError:
        # nothing to plot
        print "no data to plot"
        return ax
    
    # plot each partition
    for partnum, psessions in gobj:
        # Get corresponding df from mchanges
        pchanges = mchanges2.ix[psessions.index]
        
        # All changes should have occurred in first row
        if pchanges[1:].any().any():
            raise ValueError("unexpected changes in middle of partition")
        changed_params = pchanges.columns[pchanges.iloc[0].values]
        
        # Create a part_label based on what has changed in this partition
        partlabel = ''
        for cp in changed_params:
            value = psessions[cp].iloc[0]
            if pandas.isnull(value):
                value = 'NA'
            assert (psessions[cp].dropna() == value).all()
            shortened_value = shorten_param_value(cp, value)
            partlabel += shortened_value + '\n'

        idxs = psessions.index.values
        ax.plot(xt[idxs], psessions.perf.values, marker='o', ls='-',
            color='b')
        
        ax.text(np.mean(xt[idxs]), .1, partlabel, ha='center', 
            size='xx-small', clip_on=True)
        
        cutval = xt[idxs[-1]] + .5
        ax.plot([cutval, cutval], [0, 1], 'r-')

    ax.set_ylim((0, 1))
    ax.set_xticks(xt)
    ax.set_xticklabels(xtl, rotation=45)
    ax.set_xlim((xt[start_idx] - .5, xt[-1] + .5))
    ax.plot(ax.get_xlim(), [.5, .5], 'k--')
    
    return ax

def plot_weights(delta_days=20):
    """Plot the weight over time.
    
    Returns: f, piv
        f - Figure, one axis per cohort
        piv - pivoted data with weight and date
    """
    cohorts = MCwatch.behavior.db.getstarted()['cohorts']
    f, axa = plt.subplots(len(cohorts), 1, figsize=(7, 3.75 * len(cohorts)))

    # Get training data
    data = pandas.read_excel('/home/mouse/Documents/training_plan2.xls',
        sheetname=1)
    data = data[['Date', 'Mouse', 'Weight']]
    data = data.ix[len(data)-500:]
    
    # Fill the dates
    data['Date'] = data['Date'].fillna(method='ffill')

    # Float the weights and drop junk
    data['Weight'] = data['Weight'].convert_objects(convert_numeric=True)
    data = data.dropna()
    
    # Pivot
    piv = data.pivot_table(index='Date', columns='Mouse', values='Weight')
    
    # Delta_days
    piv = piv.ix[piv.index[-delta_days:]]

    for cohort, ax in zip(cohorts, axa):
        cohort = [mouse for mouse in cohort if mouse in piv.columns]
        ax.plot(piv[cohort], marker='s', ls='-')
        ax.set_xlim((0, len(piv)))
        ax.set_xticks(range(len(piv)))
        labels = piv.index.format(formatter = lambda x: x.strftime('%m-%d'))
        ax.set_xticklabels(labels, rotation=45, size='small')
        ax.legend(cohort, loc='lower left', fontsize='small')
        #~ ax.set_ylim((0, ax.get_ylim()[1]))
    plt.show()
    
    return f, piv

def status_check(delta_days=30):
    """Run the daily status check"""
    # For right now this same function checks for missing sessions, etc,
    # but this should be broken out
    cohorts = MCwatch.behavior.db.getstarted()['cohorts']
    for cohort in cohorts:
        plot_pivoted_performances(keep_mice=cohort, delta_days=delta_days)
    
    MCwatch.behavior.db_plot.display_perf_by_servo_from_day()
    MCwatch.behavior.db_plot.display_perf_by_rig()
    plt.show()


def plot_logfile_check(logfile, state_names='original'):
    # Run the check
    check_res = MCwatch.behavior.db.check_logfile(logfile)

    # State numbering
    if state_names == 'original':
        state_num2names = MCwatch.behavior.db.get_state_num2names()  
    elif state_names == 'debug':
        state_num2names = MCwatch.behavior.db.get_state_num2names_dbg()  
    else:
        raise ValueError("unknown state names: %r" % state_names)
   
    ## Graph
    # Form the graph object
    G = nx.DiGraph()

    # Nodes
    for arg0 in check_res['norm_stm'].index:
        G.add_node(arg0)

    # Edges, weighted by transition prob
    for arg1, arg1_col in check_res['norm_stm'].iteritems():
        for arg0, val in arg1_col.iteritems():
            if not np.isnan(val):
                G.add_edge(arg0, arg1, weight=val)

    # Edge labels are the weights
    edge_labels=dict([((u,v,), "%0.2f" % d['weight']) 
        for u, v, d in G.edges(data=True)])

    # Draw
    pos = nx.circular_layout(G)
    pos[17] = np.asarray([.25, .05])
    pos[9] = np.asarray([.45, .05])
    pos[8] = np.asarray([0, .05])
    pos[13] = np.asarray([.5, .5])
    f, ax = plt.subplots(figsize=(14, 8))
    nx.draw(G, pos, ax=ax, with_labels=False, node_size=3000, node_shape='s')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)
    nx.draw_networkx_labels(G, pos, check_res['node_labels'], font_size=8)

    f.subplots_adjust(left=.01, right=.99, bottom=.01, top=.99, wspace=0, hspace=0)

    ## Plot hists of all durations
    f, axa = my.plot.auto_subplot(len(check_res['node_all_durations']),
        figsize=(14, 10))
    for node_num, ax in zip(
        sorted(check_res['node_all_durations'].keys()), axa.flatten()):
        data = check_res['node_all_durations'][node_num]
        rng = data.max() - data.min()
        
        tit_str = "%d: %s, %0.3f" % (node_num, 
            state_num2names[node_num].lower(), rng)
        ax.set_title(tit_str, size='small')
        ax.hist(data)
        
        if np.diff(ax.get_xlim()) < .01:
            mean_ax_xlim = np.mean(ax.get_xlim())
            ax.set_xlim((mean_ax_xlim - .005, mean_ax_xlim + .005))

        my.plot.rescue_tick(ax=ax, x=4, y=3)

    f.tight_layout()
    plt.show()      

def plot_pivoted_performances(start_date=None, delta_days=15, piv=None,
    drop_perfect=True, keep_mice=None, drop_mice=None,
    perf_unforced_only=False, by_day_of_training=False, f=None, ax=None,
    marker='s', color='auto', show_legend=True, stop_date=None):
    """Plots figures of performances over times and returns list of figs
    
    start_date : when to start plotting data
    delta_days : if start_date is None, plot this many recent days
    piv : if you don't want me to calculate it myself using
        calculate_pivoted_performances
    drop_perfect: assume days with perfect performance are artefactual
        and drop them
    keep_mice : keep only these mice
    drop_mice : drop these mice
    perf_unforced_only : only show one metric
    by_day_of_training : Plot everything locked to first day of data,
        rather than actual date. This is implemented very simply where null
        dates are dropped. So for instance, munged days will be dropped and
        then it will appear that they learned faster.
    f, ax : If perf_unforced_only is True, you can provide f and ax to plot
        into.
    marker : marker to use
    color : if auto, will use equally spaced in jet. otherwise can be 'k'
    stop_date : inclusive
    """
    # Choose start date
    if start_date is None:
        start_date = datetime.datetime.now() - \
            datetime.timedelta(days=delta_days)
    
    # Get pivoted unless provided
    if piv is None:
        piv = MCwatch.behavior.db.calculate_pivoted_performances(start_date=start_date,
            drop_perfect=drop_perfect, stop_date=stop_date)
    
    # plot each
    if perf_unforced_only:
        to_plot_f_l = [['perf_unforced']]
    else:
        to_plot_f_l = [
            ['perf_unforced', 'perf_all', 'n_trials', 'spoil_frac',],
            #~ ['perf_all', 'fev_corr_all', 'fev_side_all', 'fev_stay_all'],
            #~ ['perf_unforced', 'fev_corr_unforced', 'fev_side_unforced', 'fev_stay_unforced',]
            ]
    mouse_order = piv['perf_unforced'].mean(1)
    mouse_order.sort_values(inplace=True)
    mouse_order = mouse_order.index.values
    
    # Drop some mice
    if drop_mice is not None:
        mouse_order = mouse_order[~np.in1d(mouse_order, drop_mice)]
    if keep_mice is not None:
        mouse_order = mouse_order[np.in1d(mouse_order, keep_mice)]

    res_l = []
    for to_plot in to_plot_f_l:
        # Make figure
        figsize = (7, 3.75 * len(to_plot))
        if f is None and ax is None:
            f, axa = plt.subplots(len(to_plot), 1, figsize=figsize, squeeze=False)
            f.subplots_adjust(top=.95, bottom=.075)
        else:
            # Hack for the case where ax is provided and perf_unforced_only
            # is true
            axa = np.array([[ax]])
        
        # Get mice and color of mice
        mice = mouse_order #piv.index.values
        if color == 'auto':
            colors = generate_colorbar(len(mice), 'jet')
        else:
            colors = [color] * len(mice)
        
        # Iterate over metrics
        for ax, metric in zip(axa.flatten(), to_plot):
            pm = piv[metric]
        
            # Set x-axis
            if by_day_of_training:
                n_days = (~pm.isnull()).sum(1).max()
                xlabels = np.arange(n_days)
                xlabels_num = xlabels
            else:
                xlabels = piv.columns.levels[1].values
                xlabels_num = np.arange(len(xlabels))            
            
            # Plot the metric
            ax.set_ylabel(metric)
            for nmouse, mouse in enumerate(mice):
                if by_day_of_training:
                    # We just drop all the nulls wherever they occur
                    ax.plot(pm.ix[mouse].dropna().values,
                        color=colors[nmouse], ls='-', marker=marker, mec='none',
                        mfc=colors[nmouse])
                else:
                    null_mask = pm.ix[mouse].isnull().values
                    ax.plot(
                        xlabels_num[~null_mask], 
                        pm.ix[mouse].values[~null_mask], 
                        color=colors[nmouse],
                        ls='-', marker=marker, mec='none', mfc=colors[nmouse])

            # ylims and chance line
            if metric != 'n_trials':            
                ax.set_ylim((0, 1))
                ax.set_yticks((0, .25, .5, .75, 1))
            if metric.startswith('perf'):
                ax.plot(xlabels_num, np.ones_like(xlabels_num) * .5, 'k-')

            # Plot error X on missing sessions
            if not by_day_of_training:
                if ax is axa[-1, 0]:
                    for nmouse, mouse in enumerate(mice):
                        null_dates = piv['n_trials'].isnull().ix[mouse].values
                        pm_copy = np.ones_like(null_dates) * \
                            (nmouse + 0.5) / float(len(mice))
                        pm_copy[~null_dates] = np.nan
                        ax.plot(xlabels_num, pm_copy, color=colors[nmouse], marker='x',
                            ls='none', mew=1)

            # xticks
            ax.set_xlim((xlabels_num[0], xlabels_num[-1]))
            if not by_day_of_training:
                if ax is axa[-1, 0]:
                    ax.set_xticks(xlabels_num)
                    ax.set_xticklabels(xlabels, rotation=45, 
                        ha='right', size='small')
                else:
                    ax.set_xticks(xlabels_num)
                    ax.set_xticklabels([''] * len(xlabels_num))
        
        # mouse names in the top
        if show_legend:
            ax = axa[0, 0]
            xlims = ax.get_xlim()
            for nmouse, (mouse, color) in enumerate(zip(mice, colors)):
                xpos = xlims[0] + (nmouse + 0.5) / float(len(mice)) * \
                    (xlims[1] - xlims[0])
                ax.text(xpos, 0.2, mouse, color=color, ha='center', va='center', 
                    rotation=90)
        
        # Store to return
        res_l.append(f)
    
    return res_l

def display_session_plots_from_day(date=None):
    """Display all session plots from date, or most recent date"""
    bdf = MCwatch.behavior.db.get_behavior_df()
    bdf_dates = bdf['dt_end'].apply(lambda dt: dt.date())
    
    # Set to most recent date in database if None
    if date is None:
        date = bdf_dates.max()
    
    # Choose the ones to display
    display_dates = bdf.ix[bdf_dates == date]
    if len(display_dates) > 20:
        raise ValueError("too many dates")
    
    # Display each
    f_l = []
    for idx, row in display_dates.iterrows():
        f = display_session_plot(row['session'])
        f.text(.99, .99, row['session'], size='small', ha='right', va='top')
        f_l.append(f)
    return f_l

def display_overlays_by_rig_from_day(date=None, rigs=('B1', 'B2', 'B3', 'B4'),
    overlay_meth='all'):
    """Plot all overlays from each rig to check positioning
    
    The 'frames' dir needs to be filled out first. The easiest way to do 
    this is to run make_overlays_from_all_fits.
    """
    # Load data
    sbvdf = MCwatch.behavior.db.get_synced_behavior_and_video_df()
    msdf = MCwatch.behavior.db.get_manual_sync_df()
    sbvdf_dates = sbvdf['dt_end'].apply(lambda dt: dt.date())
    
    # Set to most recent date in database if None
    if date is None:
        date = sbvdf_dates.max()
    
    # Choose the ones to display
    display_dates = sbvdf.ix[sbvdf_dates == date]

    # Select by rigs and sort by rig and date
    display_dates = my.pick_rows(display_dates, rig=rigs).sort(
        ['rig', 'dt_end'])

    # Make a figure with rigs on columns
    n_rows = display_dates.groupby('rig').apply(len).max()
    n_cols = len(rigs)
    f, axa = plt.subplots(n_rows, n_cols)

    # Go through each entry and place into appropriate axis
    rownum = np.zeros(n_cols)
    for idx, sub_sbvdf_row in display_dates.iterrows():
        # Figure out which row and column we're in
        col = rigs.index(sub_sbvdf_row['rig'])
        row = rownum[col]
        rownum[col] = rownum[col] + 1
        ax = axa[row, col]
        
        # Title the subplot with the session
        session = sub_sbvdf_row['session']
        ax.set_title(session, size='small')
        
        # Try to construct the meaned frames
        if session in msdf.index:
            # Syncing information available
            MCwatch.behavior.overlays.make_overlays_from_fits(session, 
                verbose=True, ax=ax, ax_meth=overlay_meth)
        else:
            # No syncing information
            ax.set_xticks([])
            ax.set_yticks([])
    
    return f

def display_perf_by_servo_from_day(date=None):
    """Plot perf vs servo position from all sessions from date"""
    # Get bdf and its dates
    bdf = MCwatch.behavior.db.get_behavior_df()
    bdf_dates = bdf['dt_end'].apply(lambda dt: dt.date())
    
    # Set to most recent date in database if None
    if date is None:
        date = bdf_dates.max()
    
    # Choose the ones to display
    display_dates = bdf.ix[bdf_dates == date]
    if len(display_dates) > 20:
        raise ValueError("too many dates")
    
    # Display each
    f, axa = plt.subplots(2, my.rint(np.ceil(len(display_dates) / 2.0)),
        figsize=(15, 5))
    for nax, (idx, row) in enumerate(display_dates.iterrows()):
        ax = axa.flatten()[nax]
        display_perf_by_servo(session=row['session'], ax=ax)
        ax.set_title(row['session'], size='small')
    f.tight_layout()


def display_perf_by_servo(session=None, tm=None, ax=None, mean_meth='lr_pool'):
    """Plot perf by servo from single session into ax.
    
    if session is not None, loads trial matrix from it.
    if session is None, uses tm.
    
    mean_meth: string
        if 'lr_pool': sum together the trials on left and right before
        averaging
        if 'lr_mean': mean the performances on left and right
    """
    # Get trial matrix
    if session is not None:
        tm = MCwatch.behavior.db.get_trial_matrix(session)
    
    # Ax
    if ax is None:
        f, ax = plt.subplots()

    # Pivot perf by servo pos and rewside
    tm = my.pick_rows(tm, isrnd=True)
    if len(tm) < 5:
        return ax
    
    # Group by side and servo and calculate perf
    gobj = tm.groupby(['rewside', 'servo_pos'])
    rec_l = []
    for (rwsd, sp), subdf in gobj:
        ntots = len(subdf)
        nhits = np.sum(subdf.outcome == 'hit')
        rec_l.append({'rewside': rwsd, 'servo_pos': sp, 
            'nhits': nhits, 'ntots': ntots})
    resdf = pandas.DataFrame.from_records(rec_l)
    resdf['perf'] = resdf['nhits'] / resdf['ntots']

    if mean_meth == 'lr_average':
        # mean left and right performances
        meanperf = resdf.groupby('servo_pos')['perf'].mean()
    elif mean_meth == 'lr_pool':
        # combine L and R trials
        lr_combo = resdf.groupby('servo_pos').sum()
        meanperf = lr_combo['nhits'] / lr_combo['ntots']
    else:
        raise ValueError(str(mean_meth))
    

    # Plot
    colors = {'left': 'blue', 'right': 'red', 'mean': 'purple'}
    xticks = resdf['servo_pos'].unique()
    for rwsd, subperf in resdf.groupby('rewside'):
        xax = subperf['servo_pos']
        yax = subperf['perf']
        ax.plot(xax, yax, color=colors[rwsd], marker='s', ls='-')
    ax.plot(meanperf.index, meanperf.values, color=colors['mean'], marker='s', ls='-')
    ax.set_xlim((resdf['servo_pos'].min() - 50, resdf['servo_pos'].max() + 50))
    ax.set_xticks(xticks)
    ax.plot(ax.get_xlim(), [.5, .5], 'k-')
    ax.set_ylim((0, 1))    
    
    return ax

def display_perf_by_rig(piv=None, drop_mice=('KF28', 'KM14', 'KF19')):
    """Display performance by rig over days"""
    # Get pivoted unless provided
    if piv is None:
        piv = MCwatch.behavior.db.calculate_pivoted_perf_by_rig(drop_mice=drop_mice)
    
    # plot each
    to_plot_f_l = [
        ['perf_unforced', 'n_trials', 'fev_side_unforced',]
        ]
    
    # The order of the traces, actually rig_order here not mouse_order
    mouse_order = piv['perf_unforced'].mean(1)
    mouse_order.sort_values(inplace=True)
    mouse_order = mouse_order.index.values
    
    # Plot each
    res_l = []
    for to_plot in to_plot_f_l:
        f, axa = plt.subplots(len(to_plot), 1, figsize=(7, 15))
        f.subplots_adjust(top=.95, bottom=.075)
        xlabels = piv.columns.levels[1].values
        xlabels_num = np.arange(len(xlabels))
        mice = mouse_order #piv.index.values
        colors = generate_colorbar(len(mice), 'jet')
        
        # Iterate over metrics
        for ax, metric in zip(axa, to_plot):
            pm = piv[metric]
            
            # Plot the metric
            for nmouse, mouse in enumerate(mice):
                ax.plot(xlabels_num, pm.ix[mouse].values, color=colors[nmouse],
                    ls='-', marker='s', mec='none', mfc=colors[nmouse])
                ax.set_ylabel(metric)

            # ylims and chance line
            if metric != 'n_trials':            
                ax.set_ylim((0, 1))
                ax.set_yticks((0, .25, .5, .75, 1))
            if metric.startswith('perf'):
                ax.plot(xlabels_num, np.ones_like(xlabels_num) * .5, 'k-')

            # Plot error X on missing sessions
            if ax is axa[-1]:
                for nmouse, mouse in enumerate(mice):
                    null_dates = piv['n_trials'].isnull().ix[mouse].values
                    pm_copy = np.ones_like(null_dates) * \
                        (nmouse + 0.5) / float(len(mice))
                    pm_copy[~null_dates] = np.nan
                    ax.plot(xlabels_num, pm_copy, color=colors[nmouse], marker='x',
                        ls='none', mew=1)

            # xticks
            ax.set_xlim((xlabels_num[0], xlabels_num[-1]))
            if ax is axa[-1]:
                ax.set_xticks(xlabels_num)
                ax.set_xticklabels(xlabels, rotation=45, ha='right', size='small')
            else:
                ax.set_xticks(xlabels_num)
                ax.set_xticklabels([''] * len(xlabels_num))
        
        # mouse names in the top
        ax = axa[0]
        xlims = ax.get_xlim()
        for nmouse, (mouse, color) in enumerate(zip(mice, colors)):
            xpos = xlims[0] + (nmouse + 0.5) / float(len(mice)) * \
                (xlims[1] - xlims[0])
            ax.text(xpos, 0.2, mouse, color=color, ha='center', va='center', 
                rotation=90)
        
        # Store to return
        res_l.append(f)
    
    return res_l

def display_session_plot(session, stimulus_set=None):
    """Display the real-time plot that was shown during the session.
    
    Currently the trial_types is not saved anywhere, so we'll have to
    assume. Should think of a way to save metadata from trial_setter that
    is not saved in the logfile.
    """
    # Find the filename
    bdf = MCwatch.behavior.db.get_behavior_df()
    rows = bdf[bdf.session == session]
    if len(rows) != 1:
        raise ValueError("cannot find unique session for %s" % session)
    filename = rows['filename'].iloc[0]
    if stimulus_set is None:
        stimulus_set = rows['stimulus_set'].iloc[0]

    # If we can't get the stimulus set, try everything
    if pandas.isnull(stimulus_set) or stimulus_set == '':
        # Get the trial types by listing stim sets
        trial_types_glob = os.path.join(
            os.path.split(ArduFSM.__file__)[0],
            'stim_sets', 'trial_types*')
        trial_types_to_try = glob.glob(trial_types_glob)
    
        trial_types_to_try = [os.path.split(tt)[1] for tt in trial_types_to_try]
        trial_types_to_try = sorted(trial_types_to_try)[::-1]
    else:
        trial_types_to_try = [stimulus_set]
    
    # Now try each one
    for tttt in trial_types_to_try:
        trial_types = mainloop.get_trial_types(tttt)
        plotter = ArduFSM.plot.PlotterWithServoThrow(trial_types)
        plotter.init_handles()
        try:
            plotter.update(filename)     
        except (ArduFSM.plot.TrialTypesError, KeyError):
            print "warning: trying different trial types"
            plt.close(plotter.graphics_handles['f'])
            continue
        break
    
    # Set xlim to include entire session
    trial_matrix = MCwatch.behavior.db.get_trial_matrix(session)
    plotter.graphics_handles['ax'].set_xlim((0, len(trial_matrix)))
    
    # label
    plotter.graphics_handles['f'].text(0, 0, session)
    
    plt.show()
    return plotter.graphics_handles['f']

