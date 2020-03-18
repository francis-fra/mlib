"""
Author: Francis Chan

Data exploring functions for model building

"""
import utility as ut
import pandas as pd
import numpy as np
from itertools import groupby
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import time

# from . import utility as ut

#----------------------------------------------------------------------
# Data Types
#----------------------------------------------------------------------
def get_type_dict(df):
    """Return a dictionary {dtype: [list of variable names]}"""
    
    uniquetypes = set(df.dtypes)
    typeDict = {}
    for dt in uniquetypes:
        typeDict[str(dt)] = [c for c in df.columns if df[c].dtype == dt]
    return (typeDict)

def get_column_type(df, col):
    """Return dtype of the specified column"""
    return (df[col].dtype)


def get_type_tuple(df):
    """Return a list [(colunm name, datatype)]"""
    return([(c, df[c].dtype) for c in df.columns])


def get_categorical_column(df, exclusions=None, index=False, max_distinct=15, cutoff = 0.01):
    """
        Return [list of categorical column]
        
        Categorical column is either:
        1) non-numerical
        2) numeric but small number of finite values
    
    """
    
    if exclusions is None:
        exclusions = []
        
    if (isinstance(exclusions, str)):
        exclusions = [exclusions]
        
    result = []
    result += get_non_numerical_column(df)
    
    # dictionary of unique values proportion
    d = count_unique_values(df, prop=True)
    c = count_unique_values(df, prop=False)
    
    small_prop_set = [k for (k, v) in d.items() if v < cutoff]
    small_finite_set = [k for (k, v) in c.items() if v < max_distinct]

    # AND condition
    result += list(set(small_prop_set) & set(small_finite_set))
    
    result = list(set(result) - set(exclusions))

    if index == False:
        return (result)
    else:
        return [df.columns.get_loc(x) for x in result]


def get_non_categorical_column(df, exclusions=None, index=False, max_distinct=15, cutoff = 0.05):
    """Return [list of non categorical column]"""
    
    if exclusions is None:
        exclusions = []
        
    if (isinstance(exclusions, str)):
        exclusions = [exclusions]
        
    categorical = get_categorical_column(df, exclusions, False, max_distinct, cutoff)
    all_columns = df.columns
    result = list(set(all_columns) - set(categorical))
    result = list(set(result) - set(exclusions))
    
    if index == False:
        return (result)
    else:
        return [df.columns.get_loc(x) for x in result]
    
def get_numerical_column(df, index=False):
    """Return [list of numerical column]"""
    
    cols = df._get_numeric_data().columns.tolist()
    if index == True:
        return [df.columns.get_loc(x) for x in cols]
    else:
        return (cols)
    
def get_non_numerical_column(df, index=False):
    """Return [list of numerical column]"""
    
    cols = list(set(df.columns) - set(df._get_numeric_data().columns.tolist()))
    if index == True:
        return [df.columns.get_loc(x) for x in cols]
    else:
        return (cols)

def get_column_type(df):
    """Return dict of type: binary, categorial or Numerical"""

    unique_map = count_unique_values(df)
    # binary col are categorical but has only two distinct values
    categorical_col = get_categorical_column(df)
    numerical_col = list(set(df.columns) - set(categorical_col))
    
    binary_col = [ col for col in categorical_col if unique_map[col] == 2]
    categorical_col = list(set(categorical_col) - set(binary_col))

    return {'binary': binary_col, 'categorical': categorical_col, 'numerical': numerical_col}


#----------------------------------------------------------------------
# Count
#----------------------------------------------------------------------
def get_distinct_value(df, col):
    """Return distinct values of the given column"""
    return (df[col].unique())

def count_unique_values(df, prop=False, subset=None, dropna=True):
    """Return a dictionary {column name, num_unique_values}"""
    
    if subset is None:
        subset = df.columns
        
    if (isinstance(subset, str)):
        subset = [subset]
        
    if prop == False:
        f = lambda x: x.nunique(dropna)
    else:
        f = lambda x: x.nunique(dropna) / df.shape[0]
    
    return (df[subset].T.apply(f, axis=1).to_dict())

def count_levels(df, cols=None, prop=False, dropna=False):
    """Return a dictionary {column_name: list((unique_value, count)}"""
    
    result = {}
    if cols is None:
        cols = df.columns
    elif (isinstance(cols, str)):
        cols = [cols]
           
    g = lambda pair: pair[0]
    for c in cols:
        x = df[c].value_counts(dropna = dropna)
        if prop:
            total = sum(x.values)
            result[c] = ut.robustsort(list(zip(x.index, x.values / total)), g)
        else:
            result[c] = ut.robustsort(list(zip(x.index, x.values)), g)
        
    return result

    
def count_missing(df, prop=False): 
    """Count a dictionary of the number of missing data in each column"""
    
    if prop == False:
        return (df.isnull().sum().to_dict())
    else:
        result = df.isnull().sum() / df.shape[0]
        return (result.to_dict())
    
#-------------------------------------------------------------------------
# optus 
#-------------------------------------------------------------------------
def extract_and_count(df, field, value, target):
    return df[df[field] == value][target].value_counts()

def extract_and_plot(df, field, value):
    time_grp = df[df[field] == value].groupby(df.time_in_hour)
    num_visits = time_grp.count().bytes
    plt.plot(num_visits.index, num_visits)
    plt.gcf().autofmt_xdate()
    
def get_top_counts(data, field, limits):
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

def df_select(df, *args):
    "select df based on OR conditions"
    sel = args[0]
    if len(args) > 1:
        for arg in args[1:]:
            sel = np.logical_or(sel, arg)
    return df[sel]

def df_by_index(df, index):
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

def is_target_found(df, field, target):
    return (df[field].isin([target]).any())

def filter_df_by_value(sessions, target, field):
    "filter list of df if a column contain the target" 
    return list(filter(lambda df: is_target_found(df, field, target), sessions))

def extract_column_from_dfs(dfs, field):
    "get columns from list of df"
    return [list(df[field]) for df in dfs]

def flatten(lsts):
    return [item for sublist in lsts for item in sublist]

def remove_seq_duplicate(lst):
    return [x[0] for x in groupby(lst)]

# TODO: assertion function
# target cannot be missing

def get_links(lst):
    "get adjacent pair"
    if len(lst) == 0:
        return None
    else:
        return list(zip(lst[:-1], lst[1:]))

def build_counter_links(lsts):
    "return counter object of all links"
    all_links = flatten([get_links(lst) for lst in lsts])
    return Counter(all_links)

def get_nodes_from_counter(cnts, limit):
    "get all unique nodes"
    lsts = cnts.most_common(limit)
    # list of tuples of links
    tup_lst = [lst[0] for lst in lsts]
    return list(set(flatten(tup_lst)))

def build_d3_node_lists(lst):
    return [{"name": x} for x in lst]
    
def build_d3_links(cnt, limit):
    lsts = cnt.most_common(limit)
    return [{"source": key[0], "target": key[1], "value": value} for (key, value) in lsts]

def append_prefix_to_list_items(lsts):
    "append idx to front of item"
    return [str(idx) + lst for idx, lst in enumerate(lsts)]

def generate_d3_dict(cnt, limit):
    "generate json file for d3 sankey"
    node_list = get_nodes_from_counter(cnt, limit)
    node_list = build_d3_node_lists(node_list)
    link_list = build_d3_links(cnt, limit)
    return {"links": link_list, "nodes": node_list}



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
    
def plot_top_counts(data, ishor=False):
    keys = [item[0] for item in data]
    values = [item[1] for item in data]
    if ishor == False:
        df = pd.DataFrame({'categories': keys, 'counts': values})
        ax = df.plot.bar(x='categories', y='counts', rot=0)
    else:
        keys.reverse()
        values.reverse()
        df = pd.DataFrame({'categories': keys, 'counts': values})
        ax = df.plot.barh(x='categories', y='counts', rot=0)

def plot_series(s, ishor=False):
    "bar chart for series"
    if ishor == True:
        s = s.sort_values(ascending=True)
        df = pd.DataFrame(s).plot.barh()
    else:
        df = pd.DataFrame(s).plot.bar()


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