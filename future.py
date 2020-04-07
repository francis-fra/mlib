
#-------------------------------------------------------------------------
# optus 
#-------------------------------------------------------------------------
def extract_and_count(df, field, value, target):
    """count target values for a particular value in a particular field"""
    return df[df[field] == value][target].value_counts()

# FIXME: parametize groupby field
def extract_and_plot(df, field, value):
    time_grp = df[df[field] == value].groupby(df.time_in_hour)
    num_visits = time_grp.count().bytes
    plt.plot(num_visits.index, num_visits)
    plt.gcf().autofmt_xdate()
    
# FIXME: parametize sort function
def get_top_counts(data, field, limits):
    """show the values in descending order"""
    arr = data[field]
    arr.sort(key=lambda x: -x[1])
    return arr[:limits]

def get_segments(series, max_gap=600):
    "partition into lists if a value greater than max_gap is found"
    time_diff = series.diff().apply(lambda x: x.seconds)
    segments = time_diff > max_gap
    result = []
    sessions = []
    for idx, x in segments.iteritems():
        if (x == True):
            sessions.append(result)
            result = [idx]
        else:
            result.append(idx)
    return sessions

def get_session(df, hosts=None):
    "partition get lists of browsing sessions for given hosts"
    result = []
    if hosts is None:
        hosts = df.hosts.unique
        
    for host in hosts:
        tmp = df[df.host == host]
        sessions = get_segments(tmp.datetime)
        # put each session into an array
        host_session = [tmp.loc[pd.Index(idx)] for idx in sessions]
        result = result + host_session
        
    return result

def df_select(df, reducer=np.logical_or, *args):
    "select df based on OR conditions"
    sel = args[0]
    if len(args) > 1:
        for arg in args[1:]:
                sel = reducer(sel, arg)
    return df[sel]

def df_by_index(df, index):
    "select rows with the given index array"
    return df.loc[pd.Index(index)]

def session_stat(df, sessions):
    time_diff_stat = [0] * len(sessions)
    num_request_stat = [0] * len(sessions)
    first_time_stat = [0] * len(sessions)
    longest_time_stat = [0] * len(sessions)
    landing_duration_stat = [0] * len(sessions)
    longest_page = [''] * len(sessions)
    first_page = [''] * len(sessions)
    last_page = [''] * len(sessions)
    for idx, s in enumerate(sessions):
        sdf = df_by_index(df, s)
        # get time diff
        time_diff_stat[idx] = (sdf.datetime[-1] - sdf.datetime[0]).seconds
        # get num request
        num_request_stat[idx] = len(sdf)
        # find page spent the most time 
        time_diff_series = sdf.datetime.diff()
        # time landing the longest page
        landing_page_idx = get_prev_index_of_max(time_diff_series)
        # time finish reading the longest page
        page_idx = get_current_index_of_max(time_diff_series)
        longest_page[idx] = sdf.loc[page_idx].url
        # first page
        first_page[idx] = sdf.loc[sdf.index[0]].url
        # last page
        last_page[idx] = sdf.loc[sdf.index[-1]].url
        # first page time
        first_time_stat[idx] = None if len(time_diff_series) <= 1  else time_diff_series.iloc[1].seconds
        # longest page time
        longest_time_stat[idx] = time_diff_series.loc[page_idx].seconds
        # time to land the longest page
        landing_duration_stat[idx] = (sdf.loc[landing_page_idx].datetime - sdf.datetime[0]).seconds
        
    res = [0 if j == 0 else i / j  for i, j in zip(landing_duration_stat, time_diff_stat)]
        
    return {'total_duration': time_diff_stat, 'num_request': num_request_stat,
           'longest': longest_page, 'first': first_page, 'last': last_page,
           'first_page_duration': first_time_stat, 'longest_page_duration': longest_time_stat,
           'landing_duration': landing_duration_stat, 'reaching_time_prop': res}


def analyze_sessions(sessions):
    time_diff_stat = [0] * len(sessions)
    num_request_stat = [0] * len(sessions)
    first_time_stat = [0] * len(sessions)
    longest_time_stat = [0] * len(sessions)
    landing_duration_stat = [0] * len(sessions)
    longest_page = [''] * len(sessions)
    first_page = [''] * len(sessions)
    last_page = [''] * len(sessions)
    for idx, sdf in enumerate(sessions):
        # sdf = df_by_index(df, s)
        # get time diff
        time_diff_stat[idx] = (sdf.datetime.iloc[-1] - sdf.datetime.iloc[0]).seconds
        # get num request
        num_request_stat[idx] = len(sdf)
        # find page spent the most time 
        time_diff_series = sdf.datetime.diff()
        # time landing the longest page
        # landing_page_idx = get_prev_index_of_max(time_diff_series)
        landing_page_idx = get_current_index_of_max(time_diff_series)
        # time finish reading the longest page
        page_idx = get_current_index_of_max(time_diff_series)
        longest_page[idx] = sdf.loc[page_idx].url
        # first page
        first_page[idx] = sdf.loc[sdf.index[0]].url
        # last page
        last_page[idx] = sdf.loc[sdf.index[-1]].url
        # first page time
        first_time_stat[idx] = None if len(time_diff_series) <= 1  else time_diff_series.iloc[1].seconds
        # longest page time
        longest_time_stat[idx] = time_diff_series.loc[page_idx].seconds
        # time to land the longest page
        landing_duration_stat[idx] = (sdf.loc[landing_page_idx].datetime - sdf.datetime.iloc[0]).seconds
        
    res = [0 if j == 0 else i / j  for i, j in zip(landing_duration_stat, time_diff_stat)]
        
    return {'total_duration': time_diff_stat, 'num_request': num_request_stat,
           'longest': longest_page, 'first': first_page, 'last': last_page,
           'first_page_duration': first_time_stat, 'longest_page_duration': longest_time_stat,
           'landing_duration': landing_duration_stat, 'reaching_time_prop': res}


def get_prev_index_of_max(s):
    "get index of prev index of the given series"
    s_index_list = list(s.index)
    if len(s) == 1:
        return s_index_list[0]
    else:
        idx = s_index_list.index(s.idxmax()) - 1
        return s_index_list[idx]

def get_current_index_of_max(s):
    "get index of prev index of the given series"
    s_index_list = list(s.index)
    if len(s) == 1:
        return s_index_list[0]
    else:
        idx = s_index_list.index(s.idxmax())
        return s_index_list[idx]

# FIXME: rename
def is_target_found(df, field, target):
    "return true if there exists a value in df[field] == target"
    return (df[field].isin([target]).any())

def filter_df_by_value(sessions, target, field):
    "filter list of df if a column contain the target" 
    return list(filter(lambda df: is_target_found(df, field, target), sessions))

def extract_column_from_dfs(dfs, field):
    "get columns from list of df"
    return [list(df[field]) for df in dfs]

def flatten(lsts):
    "flatten a list of lists"
    return [item for sublist in lsts for item in sublist]

def remove_seq_duplicate(lst):
    return [x[0] for x in groupby(lst)]




def impact_analysis(stat, pivot, target, limit=10):
    "stat are generated by analyze_session"

    cnt = pd.Series(stat[pivot]).value_counts()
    toplist = list(cnt.head(limit).index)
    tmp = pd.concat([pd.Series(stat[pivot], name=pivot), 
            pd.Series(stat[target], name=target)], axis=1)
    # retain pivots from the top list only 
    topdf = tmp[tmp[pivot].isin(toplist)]
    return topdf.groupby(topdf[pivot])[target].mean()


def summarize_session(df):
    "turn session into a single record"

    return {
        "host": df.host.iloc[0],
        "bytes": df.bytes.sum(),
        "hour": df.hour.iloc[0],
        "weekday": time.strptime(df.weekday.iloc[0], "%A").tm_wday,
        # "pages": df[df.urltype == 'file'].basename.str.cat(sep=' '),
        "pages": ' '.join(list(set(df[df.urltype == 'file'].basename))),
        "extension": ' '.join(list(set(df[df.urltype == 'file'].extension)))
    }

def multi_dicts_to_single(dicts, cols):
    "combine dict into df, cols is the keys of the dictionaries"

    d = {}
    for c in cols:
        d[c] = [r[c] for r in dicts]
    return d

def build_session_record(sessions):
    "build one record per session"

    # sumarize statistics
    stat = analyze_sessions(sessions)
    statdf = pd.DataFrame.from_dict(stat)

    dicts = list(map(lambda df: summarize_session(df), sessions))
    d = multi_dicts_to_single(dicts, list(dicts[0].keys()))
    df = pd.DataFrame(d)

    return pd.concat([df, statdf], axis=1, sort=False)

def page_visit_duration(df):
    "visit duration per session"
    time_diff_series = df.datetime.diff()
    diff_seconds = time_diff_series.apply(lambda x: x.seconds)
    return list(zip(df.url[:-1], diff_seconds[1:]))


def all_page_visit_duration(dfs):
    "find all page visit duration for all sessions in list"
    d = defaultdict(list)
    data = [page_visit_duration(s) for s in dfs]
    data = flatten(data)
    for k, v in data:
        d[k].append(v)
    stat = [np.mean(d[k]) for k in d.keys()]
    d = dict(zip(d.keys(), stat))
    return pd.DataFrame.from_dict(d, orient='index', columns=['counts'])

def search_item_position(df, src, target, isduration=True):
    s = df[src]
    lst = list(s)
    result = [i for i,x in enumerate(lst) if x == target]
    position = result[0] if len(result) > 0 else None
    if isduration == False:
        return None if position is None else position + 1
    else:
        return None if position is None else (df.datetime.iloc[position] - df.datetime.iloc[0]).seconds

def search_all_item_positions(dfs, src, target, isduration):
    lst = [search_item_position(df, src, target, isduration) for df in dfs]
    return list(filter(lambda x: x != None, lst))