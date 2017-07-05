""""Module for miscellaneous behavior stuff

For example, stuff like extracting lick times or choice times.
TrialSpeak shouldn't depend on stuff like that.


    # Also get the pldf and use that to get lick times
    ldf = ArduFSM.TrialSpeak.read_logfile_into_df(bdf.loc[idx, 'filename']) 
    
    # Get the lick times
    lick_times = ArduFSM.TrialSpeak.get_commands_from_parsed_lines(ldf, 'TCH')
    
    # Group them by trial number and lick type and extract times
    tt2licks = lick_times.groupby(['trial', 'arg0']).groups
    for (trial, lick_type) in tt2licks:
        tt2licks[(trial, lick_type)] = \
            ldf.loc[tt2licks[(trial, lick_type)], 'time'].values / 1000.
    
    # Get response window time as first transition into response window
    state_change_df = ArduFSM.TrialSpeak.get_commands_from_parsed_lines(
        ldf, 'ST_CHG2')
    rwin_open_times = my.pick_rows(state_change_df, 
        arg1=state_name2num['RESPONSE_WINDOW'])
    rwin_open_times_by_trial = rwin_open_times.groupby(
        'trial').first()['time'] / 1000.
    
    # Get choice time as first transition out of response window
    state_change_df = ArduFSM.TrialSpeak.get_commands_from_parsed_lines(
        ldf, 'ST_CHG2')
    rwin_close_times = my.pick_rows(state_change_df, 
        arg0=state_name2num['RESPONSE_WINDOW'])
    rwin_close_times_by_trial = rwin_close_times.groupby(
        'trial').first()['time'] / 1000.
"""
import MCwatch
import ArduFSM
import numpy as np

def get_choice_times(behavior_filename, verbose=False):
    """Calculates the choice time for each trial in the logfile"""
    # Find the state number for response window
    state_num2names = MCwatch.behavior.db.get_state_num2names()    
    resp_win_num = dict([(v, k) for k, v in state_num2names.items()])[
        'RESPONSE_WINDOW']
    
    # Get the lines
    lines = ArduFSM.TrialSpeak.read_lines_from_file(behavior_filename)
    parsed_df_by_trial = \
        ArduFSM.TrialSpeak.parse_lines_into_df_split_by_trial(lines, 
            verbose=verbose)
    
    # Identify times of state change out of response window
    # No sense in warning because there's also multiple state changes on
    # rewarded trials
    choice_times = ArduFSM.TrialSpeak.identify_state_change_times(
        parsed_df_by_trial, state0=resp_win_num, show_warnings=False)
    
    return choice_times    

def get_included_trials(trial_times, data_range, t_start=0, t_stop=0):
    """Identify the trials included in a temporal range.
    
    trial_times : Series of trial times (e.g., rwin times) indexed by
        trial labels
    
    data_range : 2-tuple (start, stop) specifying interval to include
    
    t_start, t_stop : amount of time before (after) each trial time that
        must be within data_range in order for that trial to be included.
    
    Returns: trial_labels that are included    

    Ex:
    ## Get the trial matrix
    tm = MCwatch.behavior.db.get_trial_matrix(vs.bsession_name, True)

    # Include all random trials
    tm = my.pick_rows(tm, isrnd=True, outcome=['hit', 'error'])
    
    # Identify range of trials to include
    video_range_bbase = extras.get_video_range_bbase(vs)
    included_trials = extras.get_included_trials(tm['rwin_time'], 
        data_range=video_range_bbase, t_start=-2, t_stop=0)
    tm = tm.ix[included_trials]
    """
    return trial_times[
        (trial_times + t_start >= data_range[0]) &
        (trial_times + t_stop < data_range[1])
        ].index
