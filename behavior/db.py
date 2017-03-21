"""Maintaining the database of behavioral data

"""
import os
import datetime
import numpy as np
import glob
import re
import pandas
import subprocess # for ffprobe
import ArduFSM
import scipy.misc
import my
import datetime
from ArduFSM import TrialMatrix, TrialSpeak, mainloop
import socket
import json

# For django ORM
try:
    import runner.models
except ImportError:
    pass

# for get_django_database_path
import sqlalchemy

# for get_whisker_trims_table
import requests
from StringIO import StringIO
import pytz

def get_django_database_path():
    """Return URI to django mouse-cloud database.
    
    This accesses the environment variable DATABASE_URL which should
    have been setup at the time of django.setup. This could also be
    formed from django.conf.settings.DATABASES
    """
    # Connect to the master database
    database_uri = os.environ.get('DATABASE_URL')
    if database_uri is None:
        raise ValueError("cannot get DATABASE_URL from environment")
    return database_uri

def get_django_session_table():
    """Connects to mouse-cloud and extracts session table as DataFrame.
    
    Uses get_django_database_path to access mouse-cloud. Uses pandas
    to read the appropriate tables. Renames and demungs a few columns.
    
    Probably should rewrite this to use the django ORM instead of
    sqlalchemy.
    
    datetimes are returned in NewYork
    
    Returns: DataFrame, with these columns
        mouse, stimulus_set, scheduler, date_time_start
    """
    # Connect to mouse cloud
    database_path = get_django_database_path()
    conn = sqlalchemy.create_engine(database_path)

    # Read the tables using pandas
    session_table = pandas.read_sql_table('runner_session', conn)[[
        'name', 'python_param_stimulus_set', 'python_param_scheduler_name',
        'date_time_start', 'mouse_id', 'board_id', 'box_id']]
    
    # Get other tables to parse id fields
    mouse_table = pandas.read_sql_table('runner_mouse', conn)[['id', 'name']]
    board_table = pandas.read_sql_table('runner_board', conn)[['id', 'name']]
    box_table = pandas.read_sql_table('runner_box', conn)[['id', 'name']]

    # Join mouse, board, and box onto session table
    session_table = session_table.join(
        mouse_table.set_index('id'),
        on='mouse_id', rsuffix='_mouse').drop('mouse_id', 1)
    session_table = session_table.join(
        board_table.set_index('id'),
        on='board_id', rsuffix='_board').drop('board_id', 1)
    session_table = session_table.join(
        box_table.set_index('id'),
        on='box_id', rsuffix='_box').drop('box_id', 1)
    
    # Rename the suffixed columns and the PP columns
    session_table = session_table.rename(
        columns={
            'name_mouse': 'mouse',
            'name_board': 'board',
            'name_box': 'box',
            'python_param_stimulus_set': 'stimulus_set',
            'python_param_scheduler_name': 'scheduler',
        }
    ).set_index('name')

    # Replace Null in stimulus_set and scheduler with ''
    session_table['stimulus_set'].fillna('', inplace=True)
    session_table['scheduler'].fillna('', inplace=True)
    
    # Add timezone
    tz = pytz.timezone('America/New_York')
    session_table['date_time_start'] = session_table['date_time_start'].apply(
        lambda ts: ts.tz_convert(tz))
    
    return session_table

def get_whisker_trims_table():
    """Download the whisker trims from the google doc
    
    Combines the Date and Time columns, using a default time of 11pm.
    Localizes the time to America/New_York
    """
    def combine_date_and_time(dateobj, timeobj):
        """Helper function to combine times that are Timestamp or datetime.time
        
        Returns as timestamp.
        """
        try:
            res = datetime.datetime.combine(dateobj, timeobj)
        except TypeError:
            # Must be a timestamp
            res = datetime.datetime.combine(dateobj, 
                timeobj.to_datetime().time())
        return pandas.Timestamp(res)
    
    # Get the whisker trims
    url = ('https://docs.google.com/spreadsheets/d/'
        '1Dvqw36R2fYTo7iWdTHOf27HONbI78nOcHaqvSEk5Bes/export?format=csv&gid=0')
    r = requests.get(url)
    trims = pandas.read_csv(StringIO(r.content), 
        parse_dates=['Date', 'Time (def 11pm)'],
        ).rename(
        columns={'Time (def 11pm)' : 'Time'})
    trims['Time'].fillna(datetime.time(hour=23), inplace=True)

    # Combine Date and Time
    trims['dt'] = trims.apply(lambda x: combine_date_and_time(
        x['Date'], x['Time']), axis=1)

    # Localize the times to Eastern
    # The times from the google doc are timezone-naive
    # The times from django are timezone-aware and in UTC
    # I think they'll compare correctly?
    tz = pytz.timezone('America/New_York')
    trims['dt'] = trims['dt'].apply(lambda ts: ts.tz_localize(tz))
    trims = trims.drop(['Date', 'Time'], axis=1)
    
    return trims

def calculate_perf_by_training_stage(partition_params=(
    'stimulus_set', 'scheduler', 'trim', 'board', 'box',)):
    """Calculate perf on each day and split by training stage
    
    Splits on: whisker trims (from google doc), scheduler and stim set
    (from mouse-cloud)
    
    Returns: session_table, change_table
        session_table : DataFrame, with 'partition' column
        change_table : DataFrame, boolean, where each entry reflects
            where the partition occurred for that parameter
    """
    gets = getstarted()
    partition_params = list(partition_params)

    # Get the session table from django
    session_table = get_django_session_table()
    
    # Drop non-active mice
    session_table = session_table[
        session_table.mouse.isin(gets['active_mice'])]
    
    # Drop old data
    session_table = session_table[session_table.date_time_start >
        datetime.date.today() - datetime.timedelta(days=70)]

    # Get the trims table
    trims = get_whisker_trims_table()

    # Mark each trim
    session_table['trim'] = 'All'
    for idx in trims.index:
        session_table.loc[
            (session_table.mouse == trims.loc[idx, 'Mouse']) &
            (session_table.date_time_start >= trims.loc[idx, 'dt']),
            'trim'] = trims.loc[idx, 'Which Spared']

    # Join on perf metrics
    pmdf = get_perf_metrics()
    session_table = session_table.join(pmdf.set_index('session')[[
        'n_trials', 'spoil_frac', 'perf_unforced', 'perf_all']])

    # Set perf to be perf_unforced except for FA where it's perf_all
    session_table['perf'] = session_table['perf_unforced'].copy()
    msk = session_table.scheduler == 'ForcedAlternation'
    session_table.loc[msk, 'perf'] = session_table.loc[msk, 'perf_all']

    # Test to apply to each partition param
    def shift_test(ser):
        """Return True where a change occurred, but was not null"""
        return (ser != ser.shift()) & (~ser.isnull())
    
    # Concat
    change_table_l = []
    partition_l = []
    for mouse, msessions in session_table.groupby('mouse'):
        mchange_table = msessions[partition_params].apply(shift_test)
        partition_l.append(mchange_table.any(axis=1).cumsum() - 1)
        change_table_l.append(mchange_table)
    change_table = pandas.concat(change_table_l).ix[session_table.index]
    session_table['partition'] = pandas.concat(partition_l)

    return session_table, change_table


def get_locale():
    """Return the hostname"""
    return socket.gethostname()

def get_paths():
    """Return the data directories on this locale"""
    LOCALE = get_locale()
    if LOCALE == 'chris-pyramid':
        PATHS = {
            'database_root': '/home/chris/mnt/marvin/dev/behavior_db',
            'presandbox_behavior_dir': '/home/chris/mnt/nas2_home/behavior/runmice',
            'behavior_dir': '/home/chris/mnt/nas2_home/behavior/sandbox_root',
            'video_dir': '/home/chris/mnt/nas2_home/ps3eye/marvin/compressed_eye',
            }

    elif LOCALE == 'gamma':
        PATHS = {
            'database_root': '/home/jack/mnt/marvin/dev/behavior_db',
            'presandbox_behavior_dir': '/home/jack/mnt/nas2_home/behavior/runmice',
            'behavior_dir': '/home/jack/mnt/nas2_home/behavior/sandbox_root',
            'video_dir': '/home/jack/mnt/nas2_home/ps3eye/marvin/compressed_eye',
            }

    elif LOCALE == 'marvin':
        PATHS = {
            'database_root': '/home/mouse/dev/behavior_db',
            'presandbox_behavior_dir': '/home/mouse/mnt/nas2_home/behavior/runmice',
            'behavior_dir': '/home/mouse/mnt/nas2_home/behavior/sandbox_root',
            'video_dir': '/home/mouse/mnt/nas2_home/ps3eye/marvin/compressed_eye',
            }

    elif LOCALE == 'nivram':
        PATHS = {
            'database_root': '/home/chris/mnt/marvin/dev/behavior_db',
            'presandbox_behavior_dir': '/home/chris/mnt/nas2_home/behavior/runmice',
            'behavior_dir': '/home/chris/mnt/nas2_home/behavior/sandbox_root',
            'video_dir': '/home/chris/mnt/nas2_home/ps3eye/marvin/compressed_eye',
            }

    elif LOCALE == 'lumps' or LOCALE == 'lumpy':
        PATHS = {
            'database_root': '/home/jack/mnt/marvin/dev/behavior_db',
            'presandbox_behavior_dir': '/home/jack/mnt/nas2_home/behavior/runmice',
            'behavior_dir': '/home/jack/mnt/nas2_home/behavior/sandbox_root',
            'video_dir': '/home/jack/mnt/nas2_home/ps3eye/marvin/compressed_eye',
            }

    else:
        raise ValueError("unknown locale %s" % LOCALE)
    
    return PATHS

def getstarted():
    """Return a dict of data about locale, paths, and mice.
    
    This information is extracted from the django database.
    
    Keys:
        'locale' : name of locale
        'paths' : dict of paths
        'mice' : list of all mice in the database
        'rigs' : deprecated because it uses bdf rig names instead of django
            box names
        'boxes' : hard-coded list of box names in use
        'cohorts' : list of lists of mouse names
        'active_mice' : all mice for which in_training is True
    """
    res = {
        'locale': get_locale(),
        'paths': get_paths(),
        }

    # Get all mouse names and cohorts
    qs = runner.models.Mouse.objects.filter(in_training=True)
    cohort_df = pandas.DataFrame.from_records(list(qs.values_list(
        'name', 'training_cohort')), columns=['mouse', 'cohort'])
    
    # Replace all missing cohorts with -1
    cohort_df.loc[cohort_df.cohort.isnull(), 'cohort'] = -1
    
    # Group the mice by cohort
    cohort2mouse_names = dict([(cohort, list(ser.values)) 
        for cohort, ser in cohort_df.groupby('cohort')['mouse']])

    # Mouse names
    res['mice'] = list(
        runner.models.Mouse.objects.values_list('name', flat=True))
    res['cohorts'] = cohort2mouse_names.values()
    res['active_mice'] = list(np.concatenate(res['cohorts']))
    
    # Hard-coded because these rarely change, except in testing
    res['boxes'] = ['CR0', 'CR1', 'CR2', 'CR3', 'CR4']
    res['rigs'] = ['B1', 'B2', 'B3', 'B4']
    
    return res

def check_ardulines(logfile):
    """Error check the log file.
    
    Here are the things that would be useful to check:
    * File can be loaded with the loading function without error.
    * Reported lick times match reported choice
    * Reported choice and stimulus matches reported outcome
    * All lines are (time, arg, XXX) where arg is known
    * All state transitions are legal, though this is a lot of work
    
    Diagnostics to return
    * Values of params over trials
    * Various stats on time spent in each state
    * Empirical state transition probabilities
    
    Descriptive metrics of each trial
    * Lick times locked to response window opening
    * State transition times locked to response window opening
    """
    # Make sure the loading functions work
    lines = TrialSpeak.read_lines_from_file(logfile)
    pldf = TrialSpeak.parse_lines_into_df(lines)
    plst = TrialSpeak.parse_lines_into_df_split_by_trial(lines)

def get_perf_metrics():
    """Return the df of perf metrics over sessions"""
    PATHS = get_paths()
    filename = os.path.join(PATHS['database_root'], 'perf_metrics.csv')

    try:
        pmdf = pandas.read_csv(filename)
    except IOError:
        raise IOError("cannot find perf metrics database at %s" % filename)
    
    return pmdf

def flush_perf_metrics():
    """Create an empty perf metrics file"""
    PATHS = get_paths()
    filename = os.path.join(PATHS['database_root'], 'perf_metrics.csv')
    columns=['session', 'n_trials', 'spoil_frac',
        'perf_all', 'perf_unforced',
        'fev_corr_all', 'fev_corr_unforced',
        'fev_side_all', 'fev_side_unforced',
        'fev_stay_all','fev_stay_unforced',
        ]

    pmdf = pandas.DataFrame(np.zeros((0, len(columns))), columns=columns)
    pmdf.to_csv(filename, index=False)

def get_logfile_lines(session):
    """Look up the logfile for a session and return it"""
    # Find the filename
    bdf = get_behavior_df()
    rows = bdf[bdf.session == session]
    if len(rows) != 1:
        raise ValueError("cannot find unique session for %s" % session)
    filename = rows.irow(0)['filename']
    
    # Read lines
    lines = TrialSpeak.read_lines_from_file(filename)
    
    # Split by trial
    #~ splines = split_by_trial(lines)
    
    return lines

def get_trial_matrix(session, add_rwin_and_choice_times=False):
    """Return the (cached) trial matrix for a session
    
    add_rwin_and_choice_times : if True, reads the behavior file,
        and adds these times using 
        ArduFSM.TrialMatrix.add_rwin_and_choice_times_to_trial_matrix
        This takes a bit of time.
    """
    PATHS = get_paths()
    filename = os.path.join(PATHS['database_root'], 'trial_matrix', session)
    res = pandas.read_csv(filename)
    
    if add_rwin_and_choice_times:
        # Get the behavior filename
        bdf = get_behavior_df()
        rows = bdf[bdf.session == session]
        if len(rows) != 1:
            raise ValueError("cannot find unique session for %s" % session)
        filename = rows['filename'].iat[0]
        
        # Add the columns
        res = ArduFSM.TrialMatrix.add_rwin_and_choice_times_to_trial_matrix(
            res, filename)
    
    return res

def get_all_trial_matrix():
    """Return a dict of all cached trial matrices"""
    PATHS = get_paths()
    all_filenames = glob.glob(os.path.join(
        PATHS['database_root'], 'trial_matrix', '*'))
    
    session2trial_matrix = {}
    for filename in all_filenames:
        session = os.path.split(filename)[1]
        trial_matrix = pandas.read_csv(filename)
        session2trial_matrix[session] = trial_matrix
    
    return session2trial_matrix

def get_behavior_df():
    """Returns the current behavior database"""
    PATHS = get_paths()
    filename = os.path.join(PATHS['database_root'], 'behavior.csv')

    try:
        behavior_files_df = pandas.read_csv(filename, 
            parse_dates=['dt_end', 'dt_start', 'duration'])
    except IOError:
        raise IOError("cannot find behavior database at %s" % filename)
    
    # de-localeify
    behavior_files_df['filename'] = behavior_files_df['filename'].str.replace(
        '\$behavior_dir\$', PATHS['behavior_dir'])

    # de-localeify
    behavior_files_df['filename'] = behavior_files_df['filename'].str.replace(
        '\$presandbox_behavior_dir\$', PATHS['presandbox_behavior_dir'])
    
    # Alternatively, could store as floating point seconds
    behavior_files_df['duration'] = pandas.to_timedelta(
        behavior_files_df['duration'])
    
    # Force empty strings to be '' not NaN
    behavior_files_df['stimulus_set'].fillna('', inplace=True)
    behavior_files_df['protocol'].fillna('', inplace=True)
    
    return behavior_files_df
    
def get_video_df():
    """Returns the current video database"""
    PATHS = get_paths()
    filename = os.path.join(PATHS['database_root'], 'video.csv')

    try:
        video_files_df = pandas.read_csv(filename,
            parse_dates=['dt_end', 'dt_start'])
    except IOError:
        raise IOError("cannot find video database at %s" % filename)

    # de-localeify
    video_files_df['filename'] = video_files_df['filename'].str.replace(
        '\$video_dir\$', PATHS['video_dir'])
    
    # Alternatively, could store as floating point seconds
    video_files_df['duration'] = pandas.to_timedelta(
        video_files_df['duration'])    
    
    return video_files_df

def get_synced_behavior_and_video_df():
    """Return the synced behavior/video database"""
    PATHS = get_paths()
    filename = os.path.join(PATHS['database_root'], 'behave_and_video.csv')
    
    try:
        synced_bv_df = pandas.read_csv(filename, parse_dates=[
            'dt_end', 'dt_start', 'dt_end_video', 'dt_start_video'])
    except IOError:
        raise IOError("cannot find synced database at %s" % filename)
    
    # Alternatively, could store as floating point seconds
    synced_bv_df['duration'] = pandas.to_timedelta(
        synced_bv_df['duration'])    
    synced_bv_df['duration_video'] = pandas.to_timedelta(
        synced_bv_df['duration_video'])    

    # de-localeify
    synced_bv_df['filename_video'] = synced_bv_df['filename_video'].str.replace(
        '\$video_dir\$', PATHS['video_dir'])
    synced_bv_df['filename'] = synced_bv_df['filename'].str.replace(
        '\$behavior_dir\$', PATHS['behavior_dir'])        
    
    return synced_bv_df    

def get_manual_sync_df():
    PATHS = get_paths()
    filename = os.path.join(PATHS['database_root'], 'manual_bv_sync.csv')
    
    try:
        manual_bv_sync = pandas.read_csv(filename).set_index('session')
    except IOError:
        raise IOError("cannot find manual sync database at %s" % filename)    
    
    return manual_bv_sync

def set_manual_bv_sync(session, sync_poly):
    """Store the manual behavior-video sync for session
    
    TODO: also store guess_vvsb, even though it's redundant with
    the main sync df. These fits are relative to that.
    """
    PATHS = get_paths()
    
    # Load any existing manual results
    manual_sync_df = get_manual_sync_df()
    
    sync_poly = np.asarray(sync_poly) # indexing is backwards for poly
    
    # Add
    if session in manual_sync_df.index:
        raise ValueError("sync already exists for %s" % session)
    
    manual_sync_df = manual_sync_df.append(
        pandas.DataFrame([[sync_poly[0], sync_poly[1]]],
            index=[session],
            columns=['fit0', 'fit1']))
    manual_sync_df.index.name = 'session' # it forgets
    
    # Store
    filename = os.path.join(PATHS['database_root'], 'manual_bv_sync.csv')
    manual_sync_df.to_csv(filename)

def interactive_bv_sync():
    """Interactively sync behavior and video"""
    # Load synced data
    sbvdf = get_synced_behavior_and_video_df()
    msdf = get_manual_sync_df()
    sbvdf = sbvdf.join(msdf, on='session')

    # Choose session
    choices = sbvdf[['session', 'dt_start', 'best_video_overlap', 'rig', 'fit1']]
    choices = choices.rename(columns={'best_video_overlap': 'vid_overlap'})

    print "Here are the most recent sessions:"
    print choices[-20:]
    choice = None
    while choice is None:
        choice = raw_input('Which index to analyze? ')
        try:
            choice = int(choice)
        except ValueError:
            pass
    test_row = sbvdf.ix[choice]

    # Run sync
    N_pts = 3
    sync_res0 = generate_mplayer_guesses_and_sync(test_row, N=N_pts)

    # Get results
    n_results = []
    for n in range(N_pts):
        res = raw_input('Enter result: ')
        n_results.append(float(res))

    # Run sync again
    sync_res1 = generate_mplayer_guesses_and_sync(test_row, N=N_pts,
        user_results=n_results)

    # Store
    res = raw_input('Confirm insertion [y/N]? ')
    if res == 'y':
        set_manual_bv_sync(test_row['session'], 
            sync_res1['combined_fit'])
        print "inserted"
    else:
        print "not inserting"    



## End of database stuff
def get_state_num2names():
    """Return dict of state number to name.
    
    TODO: make this read or write directly from States
    """
    return dict(enumerate([
        'WAIT_TO_START_TRIAL',
        'TRIAL_START',
        'ROTATE_STEPPER1',
        'INTER_ROTATION_PAUSE',
        'ROTATE_STEPPER2',
        'MOVE_SERVO',
        'WAIT_FOR_SERVO_MOVE',
        'RESPONSE_WINDOW',
        'REWARD_L',
        'REWARD_R',
        'POST_REWARD_TIMER_START',
        'POST_REWARD_TIMER_WAIT',
        'START_INTER_TRIAL_INTERVAL',
        'INTER_TRIAL_INTERVAL',
        'ERROR',
        'PRE_SERVO_WAIT',
        'SERVO_WAIT',
        'POST_REWARD_PAUSE',
        ]))

def get_state_num2names_dbg():
    """moveout_shared branch"""
    return {
        1 :"STATE_ID_WAIT_TO_START_TRIAL",
        2 :"STATE_ID_TRIAL_START",
        3 :"STATE_ID_FINISH_TRIAL",
        10 : "STATE_ROTATE_STEPPER1",#             10
        11: "STATE_INTER_ROTATION_PAUSE",
        12: "STATE_ROTATE_STEPPER2",
        13: "STATE_MOVE_SERVO"     ,#             13
        14:"STATE_WAIT_FOR_SERVO_MOVE" ,#        14
        15:"STATE_RESPONSE_WINDOW"     ,#        15
        16:"STATE_REWARD_L"          ,#          16
        17:"STATE_REWARD_R"            ,#        17
        18:"STATE_POST_REWARD_TIMER_START",#     18
        19:"STATE_POST_REWARD_TIMER_WAIT"   ,#   19
        20: "STATE_ERROR_TIMEOUT"           ,#    22
        23:"STATE_PRE_SERVO_WAIT"            ,#  23
        24:"STATE_SERVO_WAIT"                 ,# 24
        25:"STATE_POST_REWARD_PAUSE"           ,#25        
        }

def check_logfile(logfile, state_names='original'):
    """Read the logfile and collect stats on state transitions"""
    # Read
    rdf = ArduFSM.TrialSpeak.read_logfile_into_df(logfile)

    # State numbering
    if state_names == 'original':
        state_num2names = get_state_num2names()  
    elif state_names == 'debug':
        state_num2names = get_state_num2names_dbg()  
    else:
        raise ValueError("unknown state names: %r" % state_names)

    # Extract state change times
    st_chg = ArduFSM.TrialSpeak.get_commands_from_parsed_lines(rdf, 'ST_CHG2')
    st_chg['time'] = st_chg['time'] / 1000.

    # Get duration that it was in the state in 'arg0' column
    st_chg['duration'] = st_chg['time'].diff()

    # Drop the first row, with nan duration
    st_chg = st_chg.drop(st_chg.index[0])

    # Stats on duration by "node", that is, state
    node_stats = {}
    node_all_durations = {}
    node_recs_l = []
    for arg0, subdf in st_chg.groupby('arg0'):
        # Get name and store all durations
        node_name = state_num2names[arg0]
        node_all_durations[arg0] = subdf['duration'].values
        
        # Min, max, mean
        node_recs = {}
        node_recs['node_num'] = arg0
        node_recs['node_name'] = node_name
        node_recs['min'] = subdf['duration'].min()
        node_recs['max'] = subdf['duration'].max()
        node_recs['mean'] = subdf['duration'].mean()
        node_recs['range'] = node_recs['max'] - node_recs['min']
        
        # Note the ones that are widely varying
        extra = ''
        if node_recs['range'] > 1.1 * node_recs['mean']:
            extra = '*'
        
        # Form the string
        node_stats[arg0] = "%d:%s%s\n%0.2f" % (arg0, 
            node_name.lower().replace('_', "\n"), 
            extra, node_recs['mean'])
        
        node_recs_l.append(node_recs)
    node_stats_df = pandas.DataFrame.from_records(node_recs_l).set_index('node_num')

    # Count the number of times that each state transitioned into each other state
    # The sum of this matrix equals the length of st_chg
    state_transition_matrix = st_chg.pivot_table(
        index='arg0', columns='arg1', values='trial', aggfunc=len)
    norm_stm = state_transition_matrix.divide(state_transition_matrix.sum(1), 0)

    return {
        'node_stats_df': node_stats_df,
        'norm_stm': norm_stm,
        'node_all_durations': node_all_durations,
        'node_labels': node_stats,
        }

def calculate_pivoted_performances(start_date=None, delta_days=15,
    drop_perfect=True, display_missing=False, stop_date=None):
    """Returns pivoted performance metrics
    
    start_date : when to start calculating
    delta_days : if start_date is None, do this many recent days
    drop_perfect : assume days with perfect performance are artefactual
        and drop them
    stop_date  : inclusive
    """
    # Choose start date
    if start_date is None:
        start_date = datetime.datetime.now() - \
            datetime.timedelta(days=delta_days)    
    
    # Get data and add the mouse column
    bdf = get_behavior_df()
    pmdf = get_perf_metrics()
    
    # Add start date and mouse
    pmdf = pmdf.join(bdf.set_index('session')[['mouse', 'dt_start']], on='session')

    # Filter by stop date
    if stop_date is not None:
        pmdf = pmdf.ix[pmdf.dt_start <= stop_date]
    
    # Filter by start date and drop the start date column
    pmdf = pmdf.ix[pmdf.dt_start >= start_date].drop('dt_start', 1)
    #pmdf.index = range(len(pmdf))

    # always sort on session
    pmdf = pmdf.sort('session')

    # add a "date_s" column which is just taken from the session for now
    pmdf['date_s'] = pmdf['session'].str[2:8]
    pmdf['date_s'] = pmdf['date_s'].apply(lambda s: s[:2]+'-'+s[2:4]+'-'+s[4:6])

    # Check for duplicate sessions for a given mouse
    # This gives you the indices to drop, so confusingly, take_last means
    # it returns idx of the first
    # We want to keep the last of the day (??) so take_first
    dup_idxs = pmdf[['date_s', 'mouse']].duplicated(take_last=False)
    if dup_idxs.sum() > 0:
        print "warning: dropping %d duplicated sessions" % dup_idxs.sum()
        print "\n".join(pmdf['session'][dup_idxs].values)
        pmdf = pmdf.drop(pmdf.index[dup_idxs])

    if drop_perfect:
        mask = (pmdf.perf_all == 1.0) | (pmdf.perf_unforced == 1.0)
        if np.sum(mask) > 0:
            print "warning: dropping %d perfect sessions" % np.sum(mask)
            pmdf = pmdf[~mask]

    # pivot on all metrics
    piv = pmdf.drop('session', 1).pivot_table(index='mouse', columns='date_s')

    # Find missing data
    missing_data = piv['n_trials'].isnull().unstack()
    missing_data = missing_data.ix[missing_data].reset_index()
    missing_rows = []
    for idx, row in missing_data.iterrows():
        missing_rows.append(row['date_s'] + ' ' + row['mouse'])
    if len(missing_rows) > 0 and display_missing:
        print "warning: missing the following sessions:"
        print "\n".join(missing_rows)
    
    return piv

def calculate_pivoted_perf_by_rig(start_date=None, delta_days=15, 
    drop_mice=None):
    """Pivot performance by rig and day"""
    # Choose start date
    if start_date is None:
        start_date = datetime.datetime.now() - \
            datetime.timedelta(days=delta_days)    
    
    # Get behavior data
    bdf = get_behavior_df()
    
    # Get perf columns of interest and join on rig and date
    pmdf = get_perf_metrics()[[
        'session', 'perf_unforced', 'n_trials', 'fev_side_unforced']]
    pmdf = pmdf.join(bdf.set_index('session')[['rig', 'dt_start', 'mouse']], 
        on='session')
    pmdf = pmdf.ix[pmdf.dt_start >= start_date].drop('dt_start', 1)

    # always sort on session
    pmdf = pmdf.sort_values(by='session')

    # add a "date_s" column which is just taken from the session for now
    pmdf['date_s'] = pmdf['session'].str[2:8]
    pmdf['date_s'] = pmdf['date_s'].apply(lambda s: s[:2]+'-'+s[2:4]+'-'+s[4:6])

    # drop by mice
    if drop_mice is not None:
        pmdf = pmdf[~pmdf['mouse'].isin(drop_mice)]
    pmdf = pmdf.drop('mouse', 1)

    # pivot on all metrics, and mean over replicates
    piv = pmdf.drop('session', 1).pivot_table(index='rig', columns='date_s')
    
    return piv

def calculate_perf_metrics(trial_matrix):
    """Calculate simple performance metrics on a session"""
    rec = {}
    
    # Trials and spoiled fraction
    rec['n_trials'] = len(trial_matrix)
    rec['spoil_frac'] = float(np.sum(trial_matrix.outcome == 'spoil')) / \
        len(trial_matrix)

    # Calculate performance
    rec['perf_all'] = float(len(my.pick(trial_matrix, outcome='hit'))) / \
        len(my.pick(trial_matrix, outcome=['hit', 'error']))
    
    # Calculate unforced performance, protecting against low trial count
    n_nonbad_nonspoiled_trials = len(
        my.pick(trial_matrix, outcome=['hit', 'error'], isrnd=True))
    if n_nonbad_nonspoiled_trials < 10:
        rec['perf_unforced'] = np.nan
    else:
        rec['perf_unforced'] = float(
            len(my.pick(trial_matrix, outcome='hit', isrnd=True))) / \
            n_nonbad_nonspoiled_trials

    # Anova with and without remove bad
    for remove_bad in [True, False]:
        # Numericate and optionally remove non-random trials
        numericated_trial_matrix = TrialMatrix.numericate_trial_matrix(
            trial_matrix)
        if remove_bad:
            suffix = '_unforced'
            numericated_trial_matrix = numericated_trial_matrix.ix[
                numericated_trial_matrix.isrnd == True]
        else:
            suffix = '_all'
        
        # Run anova
        aov_res = TrialMatrix._run_anova(numericated_trial_matrix)
        
        # Parse FEV
        if aov_res is not None:
            rec['fev_stay' + suffix], rec['fev_side' + suffix], \
                rec['fev_corr' + suffix] = aov_res['ess'][
                ['ess_prevchoice', 'ess_Intercept', 'ess_rewside']]
        else:
            rec['fev_stay' + suffix], rec['fev_side' + suffix], \
                rec['fev_corr' + suffix] = np.nan, np.nan, np.nan    
    
    return rec

def calculate_perf_by_rewside_and_servo_pos(trial_matrix):
    gobj = trial_matrix.groupby(['rewside', 'servo_pos'])
    rec_l = []
    for (rwsd, sp), subdf in gobj:
        ntots = len(subdf)
        nhits = np.sum(subdf.outcome == 'hit')
        rec_l.append({'rewside': rwsd, 'servo_pos': sp, 
            'nhits': nhits, 'ntots': ntots})
    resdf = pandas.DataFrame.from_records(rec_l)
    resdf['perf'] = resdf['nhits'] / resdf['ntots']
    return resdf

def search_for_behavior_files(
    behavior_dir='~/mnt/behave/runmice',
    clean=True):
    """Load behavior files into data frame.
    
    behavior_dir : where to look
    clean : see parse_behavior_filenames
    
    See also search_for_behavior_and_video_files
    """
    gets = getstarted()
    
    # expand path
    behavior_dir = os.path.expanduser(behavior_dir)
    
    # Acquire all behavior files in the subdirectories
    # Find all saved sandboxes
    # Then extract logfiles (and probably parameters / results) from each
    # sandbox_name-saved/Script/logfiles/ardulines.*
    saved_directories = sorted(glob.glob(os.path.join(behavior_dir, '*-saved')))
    
    if len(saved_directories) == 0:
        print "warning: no saved directories in %s" % behavior_dir
    
    # Ensure there is only one saved ardulines for each
    all_behavior_files = []
    for sd in saved_directories:
        # Skip if not TwoChoice or LickTrain
        if not (
            os.path.exists(os.path.join(sd, 'Script', 'TwoChoice.py')) or
            os.path.exists(os.path.join(sd, 'Script', 'LickTrain.py'))):
            continue
        logfiles = glob.glob(os.path.join(sd, 'Script', 'logfiles', 'ardulines.*'))
        assert len(logfiles) == 1
        all_behavior_files.append(logfiles[0])
    
    # Parse out metadata for each
    behavior_files_df = parse_behavior_filenames(all_behavior_files, 
        clean=clean)    
    
    # Sort and reindex
    behavior_files_df = behavior_files_df.sort_values(by='dt_start')
    behavior_files_df.index = range(len(behavior_files_df))
    
    return behavior_files_df

def search_for_behavior_and_video_files(
    behavior_dir='~/mnt/behave/runmice',
    video_dir='~/mnt/bruno-nix/compressed_eye',
    cached_video_files_df=None,
    ):
    """Get a list of behavior and video files, with metadata.
    
    Looks for all behavior directories in behavior_dir/rignumber.
    Looks for all video files in video_dir (using cache).
    Gets metadata about video files using parse_video_filenames.
    Finds which video file maximally overlaps with which behavior file.
    
    Returns: joined, video_files_df
        joined is a data frame with the following columns:
            u'dir', u'dt_end', u'dt_start', u'duration', u'filename', 
            u'mouse', u'rig', u'best_video_index', u'best_video_overlap', 
            u'dt_end_video', u'dt_start_video', u'duration_video', 
            u'filename_video', u'rig_video'
        video_files_df is basically used only to re-cache
    """
    # expand path
    behavior_dir = os.path.expanduser(behavior_dir)
    video_dir = os.path.expanduser(video_dir)

    # Search for behavior files
    behavior_files_df = search_for_behavior_files(behavior_dir)

    # Acquire all video files
    video_files = glob.glob(os.path.join(video_dir, '*.mp4'))
    if len(video_files) == 0:
        print "warning: no video files found"
    video_files_df = parse_video_filenames(video_files, verbose=True,
        cached_video_files_df=cached_video_files_df)

    # Find the best overlap
    new_behavior_files_df = find_best_overlap_video(
        behavior_files_df, video_files_df)
    
    # Join video info
    joined = new_behavior_files_df.join(video_files_df, 
        on='best_video_index', rsuffix='_video')    
    
    return joined, video_files_df

def find_best_overlap_video(behavior_files_df, video_files_df,
    cached_sbvdf=None, always_prefer_mkv=True):
    """Find the video file with the best overlap for each behavior file.
    
    cached_sbvdf: if not None, then will skip processing anything
        in the cache, and just add the new stuff
    
    Returns : behavior_files_df, but now with a best_video_index and
        a best_video_overlap columns. Suitable for the following:
        behavior_files_df.join(video_files_df, on='best_video_index', 
            rsuffix='_video')
    """
    # Operate on a copy
    behavior_files_df = behavior_files_df.copy()
    
    # Find behavior files that overlapped with video files
    behavior_files_df['best_video_index'] = -1
    behavior_files_df['best_video_overlap'] = 0.0
    
    # Find the very first video file so we don't waste time analyzing
    # behavior from earlier
    earliest_video_start = video_files_df['dt_start'].min()
    
    # Something is really slow in this loop
    for bidx, brow in behavior_files_df.iterrows():
        if brow['dt_start'] < earliest_video_start:
            # No video old enough
            continue
        
        if cached_sbvdf is not None and brow['session'] in cached_sbvdf.session.values:
            # Already synced
            continue
        
        # Find the overlap between this behavioral session and video sessions
        # from the same rig
        latest_start = video_files_df[
            video_files_df.rig == brow['rig']]['dt_start'].copy()
        latest_start[latest_start < brow['dt_start']] = brow['dt_start']
            
        earliest_end = video_files_df[
            video_files_df.rig == brow['rig']]['dt_end'].copy()
        earliest_end[earliest_end > brow['dt_end']] = brow['dt_end']
        
        # If no videos found, continue
        if len(earliest_end) == 0:
            continue
        
        # Find the video with the most overlap
        overlap = (earliest_end - latest_start)
        positive_overlaps = overlap[overlap > datetime.timedelta(0)]
        if len(positive_overlaps) == 0:
            # ie, no overlapping videos
            continue
        
        # Prefer any MKV over any MP4 files
        if always_prefer_mkv:
            # Find out if any positive overlaps are from mkv
            file_extensions = video_files_df.ix[
                positive_overlaps.index]['filename'].apply(
                lambda s: os.path.splitext(s)[1])
            
            # If so, keep only those
            if np.any(file_extensions == '.mkv'):
                overlap = overlap[positive_overlaps.index[
                    file_extensions == '.mkv']]

        # Extract the index into video_files_df of the best match
        vidx_max_overlap = overlap.argmax()
        
        # Convert from numpy timedelta64 to a normal number
        max_overlap_sec = overlap.ix[vidx_max_overlap] / np.timedelta64(1, 's')
        
        # Store if it's more than zero
        if max_overlap_sec > 0:
            behavior_files_df.loc[bidx, 'best_video_index'] = vidx_max_overlap
            behavior_files_df.loc[bidx, 'best_video_overlap'] = max_overlap_sec

    # Join video info
    joined = behavior_files_df.join(video_files_df, 
        on='best_video_index', rsuffix='_video')

    # Drop on unmatched
    joined = joined.dropna()

    # Concat with cache, if necessary
    if cached_sbvdf is not None:
        if len(joined) == 0:
            new_sbvdf = cached_sbvdf
        else:
            new_sbvdf = pandas.concat([cached_sbvdf, joined], axis=0, 
                ignore_index=True, verify_integrity=True)
    else:
        new_sbvdf = joined

    if len(new_sbvdf) == 0:
        raise ValueError("synced behavior/video frame is empty")

    return new_sbvdf

def parse_behavior_filenames(all_behavior_files, clean=True):
    """Given list of ardulines files, extract metadata and return as df.
    
    Each filename is matched to a pattern which is used to extract the
    rigname, date, and mouse name. Non-matching filenames are discarded.
    
    clean : if True, also clean up the mousenames by upcasing.
        Finally, drop the ones not in the official list of mice.
    """
    gets = getstarted()
    
    # Extract info from filename
    # directory, rigname, datestring, mouse
    # date (with hyphens) - mouse - board - box - saved
    pattern = '(\S+)-(\S+)-(\S+)-(\S+)-saved/Script/logfiles/ardulines\.(\d+)'
    rec_l = []
    for filename in all_behavior_files:
        # Match filename pattern
        m = re.match(pattern, os.path.abspath(filename))
        if m is not None:
            sandbox_date_s, mouse, board, box, ardulines_date_s = m.groups()

            # The start time is parsed from the filename
            date = datetime.datetime.strptime(ardulines_date_s, '%Y%m%d%H%M%S')
            
            # The end time is parsed from the file timestamp
            behavior_end_time = datetime.datetime.fromtimestamp(
                my.misc.get_file_time(filename))
            
            # Get the stimulus set
            json_file = os.path.normpath(
                os.path.join(filename, '../../parameters.json'))
            with file(json_file) as fi:
                params = json.load(fi)
            stimulus_set = params.get('stimulus_set', '')
            
            # Hack, because right now all rigs are using reversed, but
            # not stored
            if not stimulus_set.endswith('_r'):
                stimulus_set += '_r'
            
            # Get the protocol name
            # Not stored as a param, have to get it from the script name
            script_files = os.listdir(os.path.split(json_file)[0])
            if 'TwoChoice.py' in script_files:
                protocol = 'TwoChoice'
            elif 'LickTrain.py' in script_files:
                protocol = 'LickTrain'
            else:
                protocol = ''
            
            # Store
            rec_l.append({'rig': box, 'mouse': mouse,
                'dt_start': date, 'dt_end': behavior_end_time,
                'duration': behavior_end_time - date,
                'protocol': protocol, 'stimulus_set': stimulus_set,
                'filename': filename})
    behavior_files_df = pandas.DataFrame.from_records(rec_l)

    if len(behavior_files_df) == 0:
        raise IOError("no behavior files found")

    elif clean:
        # Clean the behavior files by upcasing
        behavior_files_df.mouse = behavior_files_df.mouse.apply(str.upper)

        # Drop any that are not in the list of accepted mouse names
        behavior_files_df = behavior_files_df.ix[
            behavior_files_df.mouse.isin(gets['mice'])]

    # Add a session name based on the date and cleaned mouse name
    behavior_files_df['session'] = behavior_files_df['filename'].apply(
        lambda s: os.path.split(s)[1].split('.')[1]) + \
        '.' + behavior_files_df['mouse']

    return behavior_files_df

def parse_video_filenames(video_filenames, verbose=False, 
    cached_video_files_df=None):
    """Given list of video files, extract metadata and return df.

    For each filename, we extract the date (from the filename) and duration
    (using ffprobe).
    
    If cached_video_files_df is given:
        1) Skips the probing of any video file already present in 
        cached_video_files_df
        2) Concatenates the new video files info with cached_video_files_df
        and returns.
    
    Returns:
        video_files_df, a DataFrame with the following columns: 
            dt_end dt_start duration filename rig
    """
    # Extract info from filename
    # directory, rigname, datestring, extension
    pattern = '(\S+)/(\S+)[\.|-](\d+)\.(\S+)'
    rec_l = []

    for video_filename in video_filenames:
        if cached_video_files_df is not None and \
            video_filename in cached_video_files_df.filename.values:
            continue
        
        if verbose:
            print video_filename
        
        # Match filename pattern
        m = re.match(pattern, os.path.abspath(video_filename))
        if m is None:
            continue
        dir, rig, date_s, video_ext = m.groups()
        
        if video_ext == 'mp4':
            ## Old way, the datestring is the end time
            # Parse the end time using the datestring
            video_end_time = datetime.datetime.strptime(date_s, '%Y%m%d%H%M%S')

            # Video duration and hence start time
            proc = subprocess.Popen(['ffprobe', video_filename],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            res = proc.communicate()[0]

            # Check if ffprobe failed, probably on a bad file
            if 'Invalid data found when processing input' in res:
                # Just store what we know so far and warn
                rec_l.append({'filename': video_filename, 'rig': rig,
                    'dt_end': video_end_time,
                    })            
                if verbose:
                    print "Invalid data found by ffprobe in %s" % video_filename
                continue

            # Parse out start time
            duration_match = re.search("Duration: (\S+),", res)
            assert duration_match is not None and len(duration_match.groups()) == 1
            video_duration_temp = datetime.datetime.strptime(
                duration_match.groups()[0], '%H:%M:%S.%f')
            video_duration = datetime.timedelta(
                hours=video_duration_temp.hour, 
                minutes=video_duration_temp.minute, 
                seconds=video_duration_temp.second,
                microseconds=video_duration_temp.microsecond)
            video_start_time = video_end_time - video_duration
        elif video_ext == 'mkv':
            ## We don't know the modification time so just use the time that
            ## the behavior was initiated as the start
            video_start_time = datetime.datetime.strptime(date_s, '%Y%m%d%H%M%S')
            
            try:
                video_duration = my.video.get_video_duration2(video_filename,
                    return_as_timedelta=True)
            except ValueError:
                # eg, corrupted file
                if verbose:
                    print "cannot get duration, corrupted?: %s" % video_filename
                continue
                
            video_end_time = video_start_time + video_duration
        
        # Store
        rec_l.append({'filename': video_filename, 'rig': rig,
            'dt_end': video_end_time,
            'duration': video_duration,
            'dt_start': video_start_time,
            })

    resdf = pandas.DataFrame.from_records(rec_l)
    
    # Join with cache, if necessary
    if cached_video_files_df is not None:
        if len(resdf) == 0:
            resdf = cached_video_files_df
        else:
            resdf = pandas.concat([resdf, cached_video_files_df], axis=0, 
                ignore_index=True, verify_integrity=True)
    
    if len(resdf) == 0:
        raise ValueError("video data frame is empty")
    
    # Sort and reindex
    resdf = resdf.sort_values(by='dt_start')
    resdf.index = range(len(resdf))    
    
    return resdf

