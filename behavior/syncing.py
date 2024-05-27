"""Module for syncing behavioral and video files"""
from __future__ import print_function
from __future__ import division
from builtins import zip
from builtins import range
from past.utils import old_div
import os
import numpy as np
import pandas
import ArduFSM
import my
from ArduFSM import TrialMatrix, TrialSpeak, mainloop
import MCwatch.behavior


## These got moved to my.syncing and the calls need to be replaced
def extract_onsets_and_durations(*args, **kwargs):
    print("warning: replace "
        "MCwatch.behavior.syncing.extract_onsets_and_durations with "
        "my.syncing.extract_onsets_and_durations in your code")
    return my.syncing.extract_onsets_and_durations(*args, **kwargs)

def longest_unique_fit(*args, **kwargs):
    print("warning: replace "
        "MCwatch.behavior.syncing.longest_unique_fit with "
        "my.syncing.longest_unique_fit")
    return my.syncing.longest_unique_fit(*args, **kwargs)

def drop_refrac(*args, **kwargs):
    print("warning: replace "
        "MCwatch.behavior.syncing.drop_refrac with "
        "my.syncing.drop_refrac in your code")
    return my.syncing.drop_refrac(*args, **kwargs)

def extract_duration_of_onsets(*args, **kwargs):
    print("warning: replace "
        "MCwatch.behavior.syncing.extract_duration_of_onsets with "
        "my.syncing.extract_duration_of_onsets in your code")
    return my.syncing.extract_duration_of_onsets(*args, **kwargs)

def extract_duration_of_onsets2(*args, **kwargs):
    print("warning: replace "
        "MCwatch.behavior.syncing.extract_duration_of_onsets2 with "
        "my.syncing.extract_duration_of_onsets2 in your code")
    return my.syncing.extract_duration_of_onsets2(*args, **kwargs)


## Other functions
def index_of_biggest_diffs_across_arr(ser, ncuts_total=3):
    """Return indices of biggest diffs in various segments of arr"""
    # Cut the series into equal length segments, not including NaNs
    ser = ser.dropna()
    cuts = [ser.index[len(ser) * ncut / ncuts_total] 
        for ncut in range(ncuts_total)]
    cuts.append(ser.index[-1])

    # Iterate over cuts and choose the index preceding the largest gap in the cut
    res = []
    for ncut in range(len(cuts) - 1):
        subser = ser.loc[cuts[ncut]:cuts[ncut+1]]
        res.append(subser.diff().shift(-1).argmax())
    return np.asarray(res)

def generate_test_times_for_user(times, max_time, initial_guess=(.9991, 7.5), 
    N=3, buffer=30):
    """Figure out the best times for a user to identify in the video
    
    times: Series of times in the initial time base.
    initial_guess: linear poly to apply to times as a first guess
    N: number of desired times, taken equally across video
    
    Returns the best times to check (those just before a large gap),
    in the guessed timebase.
    """
    # Apply the second guess, based on historical bias of above method
    new_values = np.polyval(initial_guess, times)
    times = pandas.Series(new_values, index=times.index)
    
    # Mask trials too close to end
    mask_by_buffer_from_end(times, max_time, buffer=buffer)

    # Identify the best trials to use for manual realignment
    test_idxs = index_of_biggest_diffs_across_arr(
        times, ncuts_total=N)
    test_times = times.loc[test_idxs]
    test_next_times = times.shift(-1).loc[test_idxs]
    
    return test_times, test_next_times
    

def mask_by_buffer_from_end(ser, end_time, buffer=10):
    """Set all values of ser to np.nan that occur within buffer of the ends"""
    # Avoiding setting with copy warning since we don't care
    ser.iscopy = False
    
    ser[ser < buffer] = np.nan
    ser[ser > end_time - buffer] = np.nan

def generate_mplayer_guesses_and_sync(metadata, 
    user_results=None, guess=(1., 0.), N=4, pre_time=10):
    """Generates best times to check video, and potentially also syncs.
    
    metadata : a row from bv_files to sync. Needs to specify the following:
        'filename' : behavioral filename
        'guess_vvsb_start'
        'duration_video'
        'filename_video'
    
    The fit is between these datasets:
        X : time of retraction from behavior file, minus the test_guess_vvsb
            in the metadata.
        Y : user-supplied times of retraction from video
    The purpose of 'initial_guess' is to generate better guesses for the user
    to look in the video, but the returned data always use the combined fit
    that includes any initial guess. However, test_guess_vvsb is not
    accounted for in the returned value.
    
    N times to check in the video are printed out. Typically this is run twice,
    once before checking, then check, then run again now specifying the 
    video times in `user_results`.

    If the initial guess is very wrong, you may need to find a large
    gap in the video and match it up to trials info manually, and use this
    to fix `guess` to be closer.
    """
    initial_guess = np.asarray(guess)
    
    # Load trials info
    trials_info = TrialMatrix.make_trial_matrix_from_file(metadata['filename'])
    splines = TrialSpeak.load_splines_from_file(metadata['filename'])
    lines = TrialSpeak.read_lines_from_file(metadata['filename'])
    parsed_df_split_by_trial = \
        TrialSpeak.parse_lines_into_df_split_by_trial(lines)

    # Insert servo retract time
    trials_info['time_retract'] = TrialSpeak.identify_servo_retract_times(
        parsed_df_split_by_trial)

    # Apply the delta-time guess to the retraction times
    test_guess_vvsb = metadata['guess_vvsb_start'] #/ np.timedelta64(1, 's')
    trials_info['time_retract_vbase'] = \
        trials_info['time_retract'] - test_guess_vvsb

    # Choose test times for user
    video_duration = old_div(metadata['duration_video'], np.timedelta64(1, 's'))
    test_times, test_next_times = generate_test_times_for_user(
        trials_info['time_retract_vbase'], video_duration,
        initial_guess=initial_guess, N=N)

    # Print mplayer commands
    for test_time, test_next_time in zip(test_times, test_next_times):
        pre_test_time = int(test_time) - pre_time
        print('mplayer -ss %d %s # guess %0.1f, next %0.1f' % (pre_test_time, 
            metadata['filename_video'], test_time, test_next_time))

    # If no data provided, just return
    if user_results is None:
        return {'test_times': test_times}
    if len(user_results) != N:
        print("warning: len(user_results) should be %d not %d" % (
            N, len(user_results)))
        return {'test_times': test_times}
    
    # Otherwise, fit a correction to the original guess
    new_fit = np.polyfit(test_times.values, user_results, deg=1)
    resids = np.polyval(new_fit, test_times.values) - user_results

    # Composite the two fits
    # For some reason this is not transitive! This one appears correct.
    combined_fit = np.polyval(np.poly1d(new_fit), np.poly1d(initial_guess))

    # Diagnostics
    print(os.path.split(metadata['filename'])[-1])
    print(os.path.split(metadata['filename_video'])[-1])
    print("combined_fit: %r" % np.asarray(combined_fit))
    print("resids: %r" % np.asarray(resids))    
    
    return {'test_times': test_times, 'resids': resids, 
        'combined_fit': combined_fit}


## Begin house light syncing
def get_light_times_from_behavior_file(trial_matrix):
    """Return time light goes on and off in logfile from session
    
    Currently the light turns off for 133ms at the beginning of the
    TRIAL_START state. Two caveats:
    * The actual ST_CHG2 line is before the TRL_START token, so it is
    technically assigned to the previous trial. So don't look for a state
    change.
    * The Arduino runs communications before pulsing the light, so the 
    reported time can be jittered. The only way around this would be to
    move the pulse command before commmunications.
    
    Okay, so just return 'start_time'. Keep this function for consistency.
    
    Returns: array
        time that the backlight turned off on each trial
    """
    if not hasattr(trial_matrix, 'columns'):
        raise ValueError("provide trial matrix, not bfile")
    return trial_matrix['start_time'].values

def refit_to_maximum_overlap(xdata, ydata, fitdata):
    """Refit results from longest_unique_fit to max window
    
    longest_unique_fit will terminate when it runs out of data on either
    end, or at a bad sample point. This function uses the best xvy 
    determined by longest_unique_fit, identifies the largest window of
    overlap between xdata and ydata, and refits over this largest window.
    
    This will fail badly if there are extraneous data points in xdata or
    ydata, for instance from another concatenated session!
    
    Returns: dict
        A copy of `fitdata` with extra items about the refit
    """

    # How many extra samples on the left
    extra_left = np.min([fitdata['x_start'], fitdata['y_start']])
    extra_right = np.min([
        len(xdata) - fitdata['x_stop'],
        len(ydata) - fitdata['y_stop'],
    ])
    
    # Store these maximized windows
    refitdata = fitdata.copy()
    refitdata['refit_x_start'] = refitdata['x_start'] - extra_left
    refitdata['refit_y_start'] = refitdata['y_start'] - extra_left
    refitdata['refit_x_stop'] = refitdata['x_stop'] + extra_right
    refitdata['refit_y_stop'] = refitdata['y_stop'] + extra_right
    refitdata['xlen'] = len(xdata)
    refitdata['ylen'] = len(ydata)
    refitdata['nfit'] = refitdata['refit_x_stop'] - refitdata['refit_x_start']
    
    # Refit over maximum window
    x_over_max_window = xdata[refitdata['refit_x_start']:refitdata['refit_x_stop']]
    y_over_max_window = ydata[refitdata['refit_y_start']:refitdata['refit_y_stop']]
    refitdata['refit_best_poly'] = np.polyfit(
        y_over_max_window, x_over_max_window, deg=1)
    refitdata['refit_x_pred_from_y'] = np.polyval(
        refitdata['refit_best_poly'], y_over_max_window)
    refitdata['refit_resids'] = (refitdata['refit_x_pred_from_y'] - 
        x_over_max_window)
    refitdata['refit_mse'] = np.mean(refitdata['refit_resids'] ** 2)    
    
    return refitdata

def get_95prctl_r_minus_b(frame):
    """Gets the 95th percentile of the distr of Red - Blue in the frame
    
    Spatially downsample by 2x in x and y to save time.
    """
    vals = (
        frame[::2, ::2, 0].astype(int) - 
        frame[::2, ::2, 2].astype(int)).flatten()
    return np.sort(vals)[int(.95 * len(vals))]

def get_or_save_lums(session, lumdir=None, meth='gray', verbose=True, 
    image_w=320, image_h=240):
    """Load lum for session from video or if available from cache
    
    Sends kwargs to my.video.process_chunks_of_video
    """    
    PATHS = MCwatch.behavior.db.get_paths()
    if lumdir is None:
        lumdir = os.path.join(PATHS['database_root'], 'lums')
    
    # Get metadata about session
    sbvdf = MCwatch.behavior.db.get_synced_behavior_and_video_df().set_index('session')
    session_row = sbvdf.loc[session]
    guess_vvsb_start = session_row['guess_vvsb_start']
    vfilename = session_row['filename_video']
    
    # New style filenames
    new_lum_filename = os.path.join(lumdir, 
        os.path.split(vfilename)[1] + '.lums')
    
    # If new exists, return
    if os.path.exists(new_lum_filename):
        print("cached lums found")
        lums = my.misc.pickle_load(new_lum_filename)
        return lums    

    # Get the lums ... this takes a while
    if verbose:
        print("calculating lums..")
    if meth == 'gray':
        lums = my.video.process_chunks_of_video(vfilename, n_frames=np.inf,
            verbose=verbose, image_w=image_w, image_h=image_h)
    elif meth == 'r-b':
        lums = my.video.process_chunks_of_video(vfilename, n_frames=np.inf,
            func=get_95prctl_r_minus_b, pix_fmt='rgb24', verbose=verbose)
    
    # Save
    my.misc.pickle_dump(lums, new_lum_filename)
    
    return lums
    
def sync_video_with_behavior(trial_matrix, lums=None,
    light_delta=75, diffsize=2, refrac=50,
    assumed_fps=30., error_if_no_fit=False, verbose=False,
    return_all_data=False, refit_data=True,
    video_frame_range_start=None, video_frame_range_stop=None):
    """Sync video with behavioral file
    
    Uses decrements in luminance and the backlight signal to do the sync.
    Assumes the backlight decrement is at the time of entry to state 1.
    
    The luminance signal will be inverted in order to detect decrements.
    Assumes video frame rates is 30fps, regardless of actual frame rate.
    And fits the behavior to the video based on that.
    
    bfile : behavior log, used to extract state change times
    lums : luminances by frame
    error_if_no_fit : if True and no fit is found, raises Exception
    verbose : passed to process_chunks_of_video to print out frame
        number for each chunk
        And also sent to longest_unique_fit
    
    video_frame_range_start, video_frame_range_stop : int or None
        Any flashes in the video data outside of this range will be ignored
        These are interpreted Pythonically (half-open)
    
    See my.syncing.extract_onsets_and_durations for details on
    light_delta, diffsize, and refrac.
    
    Returns:
        if not return_all_data:
            returns b2v_fit
        if return_all_data:
            returns dict with b2v_fit, lums, behavior_flash_y, video_flash_x, 
            flash_durations
            
            flash_duration_frames will be in frames, whereas video_flash_x
            will be in seconds in the `assumed_fps` timebase
    """    
    # Error check because the function call has changed
    if not hasattr(trial_matrix, 'columns'):
        raise ValueError("must provide trial_matrix, not bfile")
    
    # Get onsets and durations
    onsets, durations = my.syncing.extract_onsets_and_durations(-lums.values, 
        delta=light_delta, diffsize=diffsize, refrac=refrac)

    # Apply video_frame_range_start
    if video_frame_range_start is not None:
        # Discard onsets that are not within the frame range
        keep_mask = (onsets >= video_frame_range_start)
        onsets = onsets[keep_mask]
        durations = durations[keep_mask]

    # Apply video_frame_range_stop
    if video_frame_range_stop is not None:
        # Discard onsets that are not within the frame range
        keep_mask = ((onsets + durations) < video_frame_range_stop)
        
        # Even the last onset should be discarded, because part of its
        # trial will be during the exclusion period
        if not np.any(keep_mask):
            raise ValueError("no onsets left")
        keep_mask[np.where(keep_mask)[0][-1]] = False
        
        onsets = onsets[keep_mask]
        durations = durations[keep_mask]        

    # Convert to seconds in the spurious timebase
    v_onsets = onsets / float(assumed_fps)

    # Find the time of backlight pulse
    backlight_times = get_light_times_from_behavior_file(trial_matrix)

    # Find the fit
    # This will be None if no fit found
    res = my.syncing.longest_unique_fit(
        v_onsets, backlight_times, return_all_data=True,
        refit_data=refit_data, verbose=verbose)    
    
    if res is None and error_if_no_fit:
        raise ValueError("no fit found")

    if return_all_data:
        # In this case, need to return a dict, even if no fit found
        if refit_data:
            # In this case we expect to have refit_best_poly
            if res is None:
                # No fit found
                res = {'refit_best_poly': None}
            
            # Always have a key b2v_fit that is the appropriate fit
            res['b2v_fit'] = res['refit_best_poly']
        else:
            # In this case we expect to have best_fitpoly
            if res is None:
                # No fit found
                res = {'best_fitpoly': None}
            
            # Always have a key b2v_fit that is the appropriate fit
            res['b2v_fit'] = res['best_fitpoly']
        
        # Add in lums and other data sources for clarity
        res['video_flash_x'] = v_onsets
        res['behavior_flash_y'] = backlight_times
        res['flash_duration_frames'] = durations
        
        return res
    else:
        # Return just the fit, or None
        if res is None:
            return None
        else:
            return res['b2v_fit']
