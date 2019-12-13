"""Module for generating overlays of shape positions"""
from __future__ import print_function
from __future__ import division
from builtins import str
from past.utils import old_div
import os
import numpy as np
import pandas
import ArduFSM
import scipy.misc
import my
from ArduFSM import TrialMatrix, TrialSpeak, mainloop
import matplotlib.pyplot as plt
import my.plot
import MCwatch


def mean_frames_by_choice(trials_info, trialnum2frame):
    # Keep only those trials that we found images for
    trials_info = trials_info.ix[sorted(trialnum2frame.keys())]

    # Dump those with a spoiled trial
    trials_info = my.misc.pick_rows(trials_info, choice=[0, 1], bad=False)

    # Split on choice
    res = []
    gobj = trials_info.groupby('choice')
    for choice, subti in gobj:
        meaned = np.mean([trialnum2frame[trialnum] for trialnum in subti.index],
            axis=0)
        res.append({'choice': choice, 'meaned': meaned})
    resdf_choice = pandas.DataFrame.from_records(res)

    return resdf_choice

def make_overlay(sess_meaned_frames, ax, meth='all'):
    """Plot various overlays
    
    sess_meaned_frames - df with columns 'meaned', 'rewside', 'servo_pos'
    meth -
        'all' - average all with the same rewside together
        'L' - take closest L and furthest R
        'R' - take furthest R and closest L
        'close' - take closest of both
        'far' - take furthest of both
    """
    # Hack: replace nogo with L
    sess_meaned_frames['rewside'].replace(
        {'nogo': 'left'}, inplace=True)
    
    if not np.all(np.in1d(['left', 'right'], sess_meaned_frames['rewside'].values)):
        return None
    
    # Split into L and R
    if meth == 'all':
        L = np.mean(sess_meaned_frames['meaned'][
            sess_meaned_frames.rewside == 'left'], axis=0)
        R = np.mean(sess_meaned_frames['meaned'][
            sess_meaned_frames.rewside == 'right'], axis=0)
    elif meth == 'L':
        closest_L = my.pick_rows(sess_meaned_frames, rewside='left')[
            'servo_pos'].min()
        furthest_R = my.pick_rows(sess_meaned_frames, rewside='right')[
            'servo_pos'].max()
        L = my.pick_rows(sess_meaned_frames, rewside='left', 
            servo_pos=closest_L)['meaned'].iloc[0]
        R = my.pick_rows(sess_meaned_frames, rewside='right', 
            servo_pos=furthest_R)['meaned'].iloc[0]
    elif meth == 'R':
        closest_R = my.pick_rows(sess_meaned_frames, rewside='right')[
            'servo_pos'].min()
        furthest_L = my.pick_rows(sess_meaned_frames, rewside='left')[
            'servo_pos'].max()
        L = my.pick_rows(sess_meaned_frames, rewside='left', 
            servo_pos=furthest_L)['meaned'].iloc[0]
        R = my.pick_rows(sess_meaned_frames, rewside='right', 
            servo_pos=closest_R)['meaned'].iloc[0]     
    elif meth == 'close':
        closest_L = my.pick_rows(sess_meaned_frames, rewside='left')[
            'servo_pos'].min()
        closest_R = my.pick_rows(sess_meaned_frames, rewside='right')[
            'servo_pos'].min()
        L = my.pick_rows(sess_meaned_frames, rewside='left', 
            servo_pos=closest_L)['meaned'].iloc[0]
        R = my.pick_rows(sess_meaned_frames, rewside='right', 
            servo_pos=closest_R)['meaned'].iloc[0]     
    elif meth == 'far':
        furthest_L = my.pick_rows(sess_meaned_frames, rewside='left')[
            'servo_pos'].max()            
        furthest_R = my.pick_rows(sess_meaned_frames, rewside='right')[
            'servo_pos'].max()
        L = my.pick_rows(sess_meaned_frames, rewside='left', 
            servo_pos=furthest_L)['meaned'].iloc[0]
        R = my.pick_rows(sess_meaned_frames, rewside='right', 
            servo_pos=furthest_R)['meaned'].iloc[0]     
    else:
        raise ValueError("meth not understood: " + str(meth))
            
    # Color them into the R and G space, with zeros for B
    C = np.array([L, R, np.zeros_like(L)])
    C = old_div(C.swapaxes(0, 2).swapaxes(0, 1), 255.)

    my.plot.imshow(C, ax=ax, axis_call='image', origin='upper')
    ax.set_xticks([]); ax.set_yticks([])

    return C

def timedelta_to_seconds1(val):
    """Often it ends up as a 0d timedelta array.
    
    This especially happens when taking a single row from a df, which becomes
    a series. Then you sometimes cannot divide by np.timedelta64(1, 's')
    or by 1e9
    """
    ite = val.item() # in nanoseconds
    return old_div(ite, 1e9)

def timedelta_to_seconds2(val):
    """More preferred ... might have been broken in old versions."""
    return old_div(val, np.timedelta64(1, 's'))

def make_overlays_from_fits_for_day(overwrite_frames=False, savefig=True,
    date=None):
    """Makes overlays for date
    
    First chooses sessions to process: those taht have manual sync and 
    are from the target date.
    Then calls make_overlays_from_fits on each session
    """
    # Load data
    sbvdf = MCwatch.behavior.db.get_synced_behavior_and_video_df()
    msdf = MCwatch.behavior.db.get_manual_sync_df()
    sbvdf_dates = sbvdf['dt_end'].apply(lambda dt: dt.date())

    # Set to most recent date in database if None
    if date is None:
        date = sbvdf_dates.max()
    
    # Choose the ones to process
    display_dates = sbvdf.ix[sbvdf_dates == date]

    # Join all the dataframes we need
    jdf = display_dates.join(msdf, on='session', how='inner')

    # Do each
    for session in jdf.session:
        make_overlays_from_fits(session, overwrite_frames=overwrite_frames,
            savefig=savefig)

def make_overlays_from_fits(session, overwrite_frames=False, savefig=True,
    verbose=True, ax=None, ax_meth='all'):
    """Given a session name, generates overlays.

    Compare this with db_plot.display_overlays_by_rig_from_day

    If savefig: then it will save the figure in behavior_db/overlays
        However, if that file already exists, it will exit immediately.
    If overwrite_frames: then it will always redump the frames
    
    If ax is not None:
        Then we assume a figure has already been created and we
        just plot into that ax. In that case we pass the parameter 'ax_meth'
        to make_overlay, which determines the type of overlay.
    If ax is None: we make a figure with various types of 'meth'
        savefig only applies in this case
    """
    # Load data
    sbvdf = MCwatch.behavior.db.get_synced_behavior_and_video_df()
    msdf = MCwatch.behavior.db.get_manual_sync_df()
    PATHS = MCwatch.behavior.db.get_paths()

    # Choose the savename and skip if it exists
    if ax is None and savefig:
        savename = os.path.join(PATHS['database_root'], 'overlays',
            session + '.png')
        if os.path.exists(savename):
            if verbose:
                print("overlay image already exists, returning:", savename)
            return
    
    # Join all the dataframes we need and check that session is in there
    jdf = sbvdf.join(msdf, on='session', how='inner').set_index('session')
    if session not in jdf.index:
        raise ValueError("no syncing information for %s" % session)
    
    # Set the filename for the cached trial_number2frame
    cache_filename = os.path.join(PATHS['database_root'], 'frames', 
        session + '.trial_number2frame.pickle')
    
    # Generate or reload the cache
    if not overwrite_frames and os.path.exists(cache_filename):
        if verbose:
            print("reloading", cache_filename)
        trial_number2frame = my.misc.pickle_load(cache_filename)
    else:
        if jdf.loc[session, 
            ['filename', 'filename_video', 'fit0', 'fit1']].isnull().any():
            raise ValueError("not enough syncing information for %s" % session)
        if verbose:
            print("generating", cache_filename)
        trial_number2frame = extract_frames_at_retraction_times(
            behavior_filename=jdf.loc[session, 'filename'], 
            video_filename=jdf.loc[session, 'filename_video'],
            b2v_fit=(jdf.loc[session, 'fit0'], jdf.loc[session, 'fit1']),
            verbose=False)
        my.misc.pickle_dump(trial_number2frame, cache_filename)

    # Keep only those trials that we found images for
    trial_matrix = MCwatch.behavior.db.get_trial_matrix(session)
    trial_matrix = trial_matrix.ix[sorted(trial_number2frame.keys())]

    # Split on side, servo_pos, stim_number
    res = []
    gobj = trial_matrix.groupby(['rewside', 'servo_pos', 'stepper_pos'])
    for (rewside, servo_pos, stim_number), subti in gobj:
        meaned = np.mean([trial_number2frame[trialnum] for trialnum in subti.index],
            axis=0)
        res.append({'rewside': rewside, 'servo_pos': servo_pos, 
            'stim_number': stim_number, 'meaned': meaned})
    resdf = pandas.DataFrame.from_records(res)

    # Make the various overlays
    if ax is None:
        f, axa = plt.subplots(2, 3, figsize=(13, 6))
        make_overlay(resdf, axa[0, 0], meth='all')
        make_overlay(resdf, axa[1, 1], meth='L')
        make_overlay(resdf, axa[1, 2], meth='R')
        make_overlay(resdf, axa[0, 1], meth='close')
        make_overlay(resdf, axa[0, 2], meth='far')
        f.suptitle(session)
        
        # Save or show
        if savefig:
            f.savefig(savename)
            plt.close(f)
        else:
            plt.show()    
    else:
        # plotting a certain meth into a certain ax
        make_overlay(resdf, ax, meth=ax_meth)

def extract_frames_at_retraction_times(behavior_filename, video_filename, 
    b2v_fit, verbose=False):
    """Extract the frame at each servo retraction time
    
    Returns: dict from trial number to frame
    """
    # Get the state change times
    state_change_times = ArduFSM.TrialSpeak.identify_servo_retract_times(
        behavior_filename)

    # Fit to video times
    state_change_times_vbase = pandas.Series(
        index=state_change_times.index,
        data=np.polyval(b2v_fit, old_div(state_change_times.values, 1000.))
        )
    
    # Mask out any frametimes that are before or after the video
    video_duration = my.video.get_video_duration2(video_filename)
    state_change_times_vbase.ix[
        (state_change_times_vbase < 1) |
        (state_change_times_vbase > video_duration - 1)
        ] = np.nan
    
    # Extract frames
    trial_number2frame = {}
    for trial_number, retract_time in state_change_times_vbase.dropna().items():
        if verbose:
            print(trial_number)
        frame, stdout, stderr = my.video.get_frame(
            video_filename, frametime=retract_time, pix_fmt='gray')
        trial_number2frame[trial_number] = frame
    
    # Save
    return trial_number2frame

def calculate_sess_meaned_frames(trial_number2frame, trial_matrix):
    """Mean frames based on side, servo_pos, and stepper_pos
    
    trial_number2frame : dict from trial number to a frame at the retraction
        time on that trial
    trial_matrix : DataFrame containing info about each trial
    
    trial_matrix is grouped on side, servo_pos, and stepper_pos, and then
    the frames for each group of trial numbers are averaged together.
    
    Returns: sess_meaned_frames
        A DataFrame with columns rewside, servo_pos, stim_number, and
        meaned
    """
    # Ensure we only count frames that we have data for
    trial_matrix = trial_matrix.ix[sorted(trial_number2frame.keys())]    
    
    # Split on side, servo_pos, stim_number
    res = []
    gobj = trial_matrix.groupby(['rewside', 'servo_pos', 'stepper_pos'])
    for (rewside, servo_pos, stim_number), subti in gobj:
        meaned = np.mean([trial_number2frame[trialnum] for trialnum in subti.index],
            axis=0)
        res.append({'rewside': rewside, 'servo_pos': servo_pos, 
            'stim_number': stim_number, 'meaned': meaned})
    resdf = pandas.DataFrame.from_records(res)    
    return resdf