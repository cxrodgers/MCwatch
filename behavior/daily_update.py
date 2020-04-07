"""Module for running daily updates on database"""
from __future__ import print_function
import os
import numpy as np
import glob
import pandas
import ArduFSM
import my
import MCwatch
from ArduFSM import TrialMatrix, TrialSpeak, mainloop


def daily_update():
    """Update the databases with current behavior and video files
    
    This should be run on marvin locale.
    """
    if MCwatch.behavior.db.get_locale() != 'marvin':
        raise ValueError("this must be run on marvin")
    
    daily_update_behavior()
    #~ daily_update_video()
    #~ daily_update_overlap_behavior_and_video()
    daily_update_trial_matrix()
    daily_update_perf_metrics()

def daily_update_behavior(force_reparse=False):
    """Update behavior database
    
    Identifies all sandboxes in the sandbox root directory. Parses them and
    adds to the behavior CSV file.
    
    force_reparse : bool
        If True, always reparse every discovered sandbox, and drop any
        duplicates after concatenating with existing behavior CSV file.
        
        If False, discard sandboxes that are already in the behavior CSV
        file, and error if there somehow are duplicates in the concatenated
        CSV file.
    """
    print("daily_update_behavior: start")
    
    # load the current database
    current_bdf = MCwatch.behavior.db.get_behavior_df()

    # get new records
    PATHS = MCwatch.behavior.db.get_paths()

    # Extract sandbox
    current_bdf['sandbox'] = current_bdf['filename'].apply(
        lambda s: s.split(os.sep)[-4])
    current_bdf.loc[current_bdf['sandbox'] == 'runmice', 'sandbox'] = np.nan

    #~ # Delete the last few for testing
    #~ current_bdf = current_bdf.loc[current_bdf.index[:-50]]

    # Search for new sandboxes
    # TODO: replace this with PATHS
    discovered_sandboxes = MCwatch.behavior.db.search_for_sandboxes()
    print("info: discovered %d sandboxes" % len(discovered_sandboxes))

    # Identify which need to be parsed
    if force_reparse:
        # Reparse all
        new_sandboxes = discovered_sandboxes
    else:
        # Reparse new only
        new_sandboxes = discovered_sandboxes.loc[
            ~discovered_sandboxes['sandbox_name'].isin(
            current_bdf['sandbox']), :]

    # Parse the new sandboxes
    newly_added_bdf = MCwatch.behavior.db.parse_sandboxes(new_sandboxes)
    if newly_added_bdf is None:
        print("info: no new sandboxes found")
    else:
        print("info: parsed %d new sandboxes" % len(newly_added_bdf))

        # Concatenate known and new
        new_bdf = pandas.concat([current_bdf, newly_added_bdf],
            ignore_index=True)
        
        # Deal with duplicates
        if force_reparse:
            # Drop the duplicates
            new_bdf = new_bdf.drop_duplicates(subset='session', 
                keep='last').reset_index(drop=True)
        
        else:
            # Error check: there should be no duplicated sessions
            if new_bdf['session'].duplicated().any():
                dup_mask = new_bdf['session'].duplicated()
                dup_sessions = new_bdf.loc[dup_mask, 'session'].values
                raise ValueError("duplicate sessions after concatenating: %r" % 
                    dup_sessions)

        # store copy for error check
        new_bdf_copy = new_bdf.copy()

        # delocale-ify
        new_bdf['filename'] = new_bdf['filename'].str.replace(
            PATHS['behavior_dir'], '$behavior_dir$')
        new_bdf['filename'] = new_bdf['filename'].str.replace(
            PATHS['presandbox_behavior_dir'], '$presandbox_behavior_dir$')        

        # save
        filename = os.path.join(PATHS['database_root'], 'behavior.csv')
        new_bdf.to_csv(filename, index=False)
    
    print("daily_update_behavior: done")
    
def daily_update_video():
    """Update video database
    
    Finds video files in PATHS['video_dir']
    Extracts timing information from them
    Updates video.csv on disk.
    """
    PATHS = MCwatch.behavior.db.get_paths()
    # find video files
    mp4_files = glob.glob(os.path.join(PATHS['video_dir'], '*.mp4'))
    mkv_files = glob.glob(os.path.join(PATHS['video_dir'], '*.mkv'))
    video_files = mp4_files + mkv_files
    
    # Load existing video file dataframe and use as a cache
    # This way we don't have to reprocess videos we already know about
    vdf = MCwatch.behavior.db.get_video_df()    
    
    # Parse into df
    video_files_df = MCwatch.behavior.db.parse_video_filenames(
        video_files, verbose=True,
        cached_video_files_df=vdf)

    # store copy for error check (to ensure that localeifying and
    # writing to disk didn't corrupt anything)
    video_files_df_local = video_files_df.copy()

    # locale-ify
    video_files_df['filename'] = video_files_df['filename'].str.replace(
        PATHS['video_dir'], '$video_dir$')
    
    # Save
    filename = os.path.join(PATHS['database_root'], 'video.csv')
    video_files_df.to_csv(filename, index=False)    
    
    # Test the reading/writing is working
    # Although if it failed, it's too late
    vdf = MCwatch.behavior.db.get_video_df()
    if not (video_files_df_local == vdf).all().all():
        raise ValueError("read/write error in video database")    

def daily_update_overlap_behavior_and_video():
    """Update the linkage betweeen behavior and video df
    
    Should run daily_update_behavior and daily_update_video first
    """
    PATHS = MCwatch.behavior.db.get_paths()
    # Load the databases
    behavior_files_df = MCwatch.behavior.db.get_behavior_df()
    video_files_df = MCwatch.behavior.db.get_video_df()

    # Load the cached sbvdf so we don't waste time resyncing
    sbvdf = MCwatch.behavior.db.get_synced_behavior_and_video_df()

    # Find the best overlap
    new_sbvdf = MCwatch.behavior.db.find_best_overlap_video(
        behavior_files_df, video_files_df,
        cached_sbvdf=sbvdf,
        always_prefer_mkv=True)
        
    # locale-ify
    new_sbvdf['filename'] = new_sbvdf['filename'].str.replace(
        PATHS['behavior_dir'], '$behavior_dir$')    
    new_sbvdf['filename_video'] = new_sbvdf['filename_video'].str.replace(
        PATHS['video_dir'], '$video_dir$')    
        
    # Save
    filename = os.path.join(PATHS['database_root'], 'behave_and_video.csv')
    new_sbvdf.to_csv(filename, index=False)

def daily_update_trial_matrix(start_date=None, verbose=False):
    """Cache the trial matrix for every session
    
    TODO: use cache
    """
    PATHS = MCwatch.behavior.db.get_paths()
    # Get
    behavior_files_df = MCwatch.behavior.db.get_behavior_df()
    
    # Filter by those after start date
    if start_date is not None:
        behavior_files_df = behavior_files_df[ 
            behavior_files_df.dt_start >= start_date]
    
    # List of existing trial matrix
    existing_list = os.listdir(os.path.join(PATHS['database_root'],
        'trial_matrix'))
    
    # Only process new ones
    new_bdf = behavior_files_df.loc[~behavior_files_df['session'].isin(
        existing_list)]
    
    # Calculate trial_matrix for each
    session2trial_matrix = {}
    for irow, row in new_bdf.iterrows():
        # Form filename
        filename = os.path.join(PATHS['database_root'], 'trial_matrix', 
            row['session'])
        if verbose:
            print(filename)

        # Make it
        try:
            trial_matrix = TrialMatrix.make_trial_matrix_from_file(row['filename'])
        except IOError:
            print(
                "warning: cannot read lines to make trial matrix from {}".format(
                row['filename']))
            # E.g. if the files got deleted or something
            continue
        
        # And store it
        trial_matrix.to_csv(filename)

def daily_update_perf_metrics(start_date=None, verbose=False):
    """Calculate simple perf metrics for anything that needs it.
    
    start_date : if not None, ignores all behavior files before this date
        You can also pass a string like '20150120'
    
    This assumes trial matrices have been cached for all sessions in bdf.
    Should error check for this.
    
    To add: percentage of forced trials. EV of various biases instead
    of FEV
    """
    PATHS = MCwatch.behavior.db.get_paths()
    # Get
    behavior_files_df = MCwatch.behavior.db.get_behavior_df()

    # Filter by those after start date
    if start_date is not None:
        behavior_files_df = behavior_files_df[ 
            behavior_files_df.dt_start >= start_date]

    # Load what we've already calculated
    pmdf = MCwatch.behavior.db.get_perf_metrics()

    # Calculate any that need it
    new_pmdf_rows_l = []
    for idx, brow in behavior_files_df.iterrows():
        # Check if it already exists
        session = brow['session']
        if session in pmdf['session'].values:
            if verbose:
                print("skipping", session)
            continue
        
        # Skip anything that is not TwoChoice
        if brow['protocol'] not in ['TwoChoice', 'TwoChoiceJung', 'TwoChoiceJungLight']:
            continue
        
        # Otherwise run
        try:
            trial_matrix = MCwatch.behavior.db.get_trial_matrix(session)
        except IOError:
            # E.g. if the files were deleted
            print("warning: cannot load trial matrix for {}".format(session))
            continue
            
        if len(trial_matrix) == 0:
            if verbose:
                print("skipping session with no rows: %s" % session)
            continue
        metrics = MCwatch.behavior.db.calculate_perf_metrics(trial_matrix)
        
        # Store
        metrics['session'] = session
        new_pmdf_rows_l.append(metrics)
    
    # Join on the existing pmdf
    new_pmdf_rows = pandas.DataFrame.from_records(new_pmdf_rows_l)
    new_pmdf = pandas.concat([pmdf, new_pmdf_rows],
        verify_integrity=True,
        ignore_index=True)
    
    # Columns are sorted after concatting
    # Re-use original, this should be specified somewhere though
    if new_pmdf.shape[1] != pmdf.shape[1]:
        raise ValueError("missing/extra columns in perf metrics")
    new_pmdf = new_pmdf[pmdf.columns]
    
    # Save
    filename = os.path.join(PATHS['database_root'], 'perf_metrics.csv')
    new_pmdf.to_csv(filename, index=False)

