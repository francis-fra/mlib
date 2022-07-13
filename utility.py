"""
Author: Francis Chan 

Utility functions for model building

"""
import os, sys, random
import bisect
import pyodbc
# from zlib import crc32
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import functools
from datetime import datetime
import dateutil.relativedelta
import matplotlib.pyplot as plt
import scipy.stats as stat
# from __future__ import division
import scipy.stats.kde as kde

from sklearn.preprocessing import label_binarize, LabelBinarizer
import explore as ex

#-------------------------------------------------------------------------
# Loading Data
#-------------------------------------------------------------------------

def load_td(source, cnxn, maxNumRows=None):
    """load single table from td

    Parameters
    ----------
    source : data base schema and table name
    cnxn : db connector
    maxNumRows : query row limit

    Returns
    -------
    df : data frame
    
    """
    if maxNumRows is not None:
        sql = "select top {} * from {}".format(maxNumRows, source)
    else:
        sql = "select * from {}".format(source)

    df = pd.read_sql(sql, cnxn)
    return df

def connect_td(uid=None, passwd=None, dsn="tdp5"):
    """get td connector

    Parameters
    ----------
    uid : user id
    passwd : password
    dsn : db name

    Returns
    -------
    cnxn : TDP5 connector
    
    """

    try:
        if passwd is None:
            passwd = os.environ['TDPASS']
        if uid is None:
            # uid = getpass.getuser()
            uid = os.environ['USERNAME']
    except:
        return None

    connect_param = "DSN={};UID={};PWD={}".format(dsn, uid, passwd)
    cnxn = pyodbc.connect(connect_param)
    return cnxn

#------------------------------------------------------------
# statistics
#------------------------------------------------------------
def hpd_grid(sample, alpha=0.05, roundto=2):
    """Calculate highest posterior density (HPD) of array for given alpha. 
    The HPD is the minimum width Bayesian credible interval (BCI). 
    The function works for multimodal distributions, returning more than one mode
    Parameters
    ----------
    
    sample : Numpy array or python list
        An array containing MCMC samples
    alpha : float
        Desired probability of type I error (defaults to 0.05)
    roundto: integer
        Number of digits after the decimal point for the results
    Returns
    ----------
    hpd: list with the highest density interval
    x: array with grid points where the density was evaluated
    y: array with the density values
    modes: list listing the values of the modes
          
    """
    sample = np.asarray(sample)
    sample = sample[~np.isnan(sample)]
    # get upper and lower bounds
    l = np.min(sample)
    u = np.max(sample)
    density = kde.gaussian_kde(sample)
    x = np.linspace(l, u, 2000)
    y = density.evaluate(x)
    #y = density.evaluate(x, l, u) waitting for PR to be accepted
    xy_zipped = zip(x, y/np.sum(y))
    xy = sorted(xy_zipped, key=lambda x: x[1], reverse=True)
    xy_cum_sum = 0
    hdv = []
    for val in xy:
        xy_cum_sum += val[1]
        hdv.append(val[0])
        if xy_cum_sum >= (1-alpha):
            break
    hdv.sort()
    diff = (u-l)/20  # differences of 5%
    hpd = []
    hpd.append(round(min(hdv), roundto))
    for i in range(1, len(hdv)):
        if hdv[i]-hdv[i-1] >= diff:
            hpd.append(round(hdv[i-1], roundto))
            hpd.append(round(hdv[i], roundto))
    hpd.append(round(max(hdv), roundto))
    ite = iter(hpd)
    hpd = list(zip(ite, ite))
    modes = []
    for value in hpd:
         x_hpd = x[(x > value[0]) & (x < value[1])]
         y_hpd = y[(x > value[0]) & (x < value[1])]
         modes.append(round(x_hpd[np.argmax(y_hpd)], roundto))
    return hpd, x, y, modes

def get_categorical_interval(cat_intervals):
    "get middle point from categorical interval"
    return [np.mean([x.left, x.right]) for x in cat_intervals]

def get_actual_pred_ci(df, target, pred, bins, alpha=0.95):
    """get confidence interval from model predictions

        Parameters
        ----------
        df: data frame
        target : name of the target column
        pred : numae of the predicted column
        bins : bin boundaries
        alpha: level of confidence

        Returns:
        --------
        return intervals for all bins
    """
    dfx, _ = pd.cut(df[pred], bins, retbins=True)
    grp = df.groupby(dfx)[target]
    interval_keys = list(grp.groups.keys())

    # mean is taken from prediction and std error from actual
    result = []
    for k in interval_keys:
        idx = grp.groups[k]
        pred_values = df.loc[idx][pred]
        values = df.loc[idx][target]
        # interval = stat.t.interval(alpha=alpha, df=len(values)-1, loc=np.mean(pred_values), scale=stat.sem(values)) 
        interval = stat.norm.interval(alpha=alpha, loc=np.mean(pred_values), scale=stat.sem(values))
        result.append(interval)
    return result

def get_conditional_avg(df, x, y, numbins=10):
    """
        Get avg value of y conditioned on x

        Parameters
        ----------
        sdf    : data frame
        x      : string
        y      : string
        numbins : num bins
    """

    # check if binning is required
    if not ex.is_column_categorical(df, x):
        dfx, bins = pd.cut(df[x], numbins, retbins=True)
        # mean of y conditional on x
        avg_line = df.groupby(dfx)[y].mean()
        xx = (bins[:-1] + bins[1:]) / 2
    else:
        cnts = df[x].value_counts().sort_index()
        avg_line = df.groupby(df[x])[y].mean()
        xx = cnts.index
        # no binning - reset numbins to natural unique values
        numbins = len(xx)
    
    return (xx, avg_line)

#------------------------------------------------------------
# Data Frame
#------------------------------------------------------------
def replace_df_value(sdf, idx, colname, value):
    "immutable function to replace vaue in dataframe"
    df = sdf.copy()
    df.loc[idx, colname] = value
    return df

def extract_and_count(df, sel_col, value, target_col):
    """count target values for a particular value in a particular field

    counts values from the target column conditioned
    on rows which contain the specific value in a specific column
    
    Parameters
    ----------
    df: data frame
    sel_col: column name for filtering
    value : value to be selected from sel_col
    target_col : name of column to be counted

    Returns:
    --------
    histogram of values in the targeted column
    
    """
    return df[df[sel_col] == value][target_col].value_counts()

def widedf_to_talldf(self, df, id_vars=None):
    "convert wide to tall data frame"

    if id_vars is None:
        id_vars = self.target_col
    df.reset_index(level=0, inplace=True)
    df = pd.melt(df, id_vars=[id_vars])
    return df


def discretize_column(df, colname, append_column=True, num_bins=3, suffix ='GRP', group_labels=None):
    """bin single column into groups and append to df
    
    Parameters
    ----------
    df: data frame
    colname : column name in string
    append_column : return only the binned column if False
    num_bins : num of bins
    suffix : suffix name given to the new column if appended
    group_labels : labels given to the categorical bins

    Returns
    -------
    df : data frame
    """
    
    if group_labels is None:
        labels = False
    else:
        labels = group_labels

    tmp = pd.qcut(df[colname], num_bins, retbins=False, duplicates='drop', labels=labels)
        
    if not append_column:
        return tmp
    else:
        newcolname = colname + '_' + suffix
        df[newcolname] = tmp
        return df

def discretize_dataframe(df, varList=None, append_column=False, num_bins=3, suffix ='GRP', 
                            group_labels=None):
    """discretize subset of columns
    
    Parameters
    ----------
    df: data frame
    varList : list of columns to bin
    append_column : return only the binned column if False
    num_bins : num of bins
    suffix : suffix name given to the new column if appended
    group_labels : labels given to the categorical bins

    Returns
    -------
    df : data frame
    """
    
    if varList is None:
        varList = df.columns
        
    categorical_cols = ex.get_categorical_column(df[varList])
    frames = []
    
    for var in varList:
        if var in categorical_cols:
            frames.append(df[var])
        else:
            tmp = discretize_column(df, var, False, num_bins, suffix, group_labels)
            tmp.name = var + '_' + suffix
            frames.append(tmp)    

    # return new frames
    result = pd.concat(frames, axis=1)
    if not append_column:
        return result
    else:
        return pd.concat([df, result], axis=1)

def sort_and_rank(df, col, ascending=False, bypct=True, zero_base=False):
    """
        sort data frame by column and append additional rank column

        Parameters
        ----------
        df : data frame
        col : column name
        bypct : boolean rank in pctile, otherwise in decile

        Returns
        ----------
        df : sorted data frame with additional rank column
    """
    length = df.shape[0]
    df = df.sort_values(by=col, ascending=ascending, ignore_index=True)
    if bypct:
        scaler = 100
    else:
        scaler = 10

    if zero_base:
        pad = 0
    else:
        pad = 1

    df['rank'] = np.floor(df.index / length * scaler) + 1
    return df
    
#------------------------------------------------------------
# Series
#------------------------------------------------------------
def unravel(s):
    return s.to_numpy().reshape(len(s),1)

#------------------------------------------------------------
# Date
#------------------------------------------------------------
def datestr2date(x, date_format='%Y-%m-%d'):
    """
        convert date string to date object

        Parameters
        ----------
        x: string or datetime object
        date_format : date format in string

        Returns
        -------
        datetime object
    """

    if isinstance(x, list):
        return [datetime.strptime(item, date_format).date() for item in x]
    else:
        return datetime.strptime(x, date_format).date()

def shift_by_month(current_month, latency, format='%Y-%m-%d'):
    """
        shift the month by the latency

        Parameters
        ----------
        current_month : base month shifting from (datetime or string object)
        latency : integer, num month to shift
        format : date format in string

        Returns
        -------
        datetime object
    """
    if isinstance(current_month, str):
        current_month = datetime.strptime(current_month, format).date()

    shifted_month = current_month + dateutil.relativedelta.relativedelta(months=latency)
    return (shifted_month)

class DiscreteSampler:
    """ Discrete value sampler

        Parameters
        ----------
        populations: list of finite values
        weights: weight assigned to each possible value

        Returns
        -------
        random choice based on the pdf
    """

    def __init__(self, population, weights):
        self.weights = weights
        self.population = population

    def _cdf(self):
        total = sum(self.weights)
        result = []
        cumsum = 0
        for w in self.weights:
            cumsum += w
            result.append(cumsum / total)
        return result

    def choice(self):
        assert len(self.population) == len(self.weights)
        cdf_vals = self._cdf()
        x = random.random()
        idx = bisect.bisect(cdf_vals, x)
        return self.population[idx]


def create_artificial_col(df, val, px, colname):
    """Create an artificial data column

        Parameters
        ----------
                df     : data frame
                val    : list of possible value to choose from
                px     : list of prob of each value
                colname: name of the new column
        Returns
        -------
                df     : a new data frame with the appended column

    """

    # assertion
    assert (colname not in df.columns)

    newdf = df.copy()
    num_rows = newdf.shape[0]
    sampler = DiscreteSampler(val, px)
    newdf[colname] = [sampler.choice() for k in range(num_rows)]

    return (newdf)

def create_missing_columns(df, cols=None, p=0.05):
    """Return a new data frame by filling the specified columns with missing values
    
        Parameters
        ----------
        df : data frame
        cols : columns to be filled with missing values
        p : prob of missing values

        Returns
        -------
        df : new data frame
    """

    if cols is None:
        cols = df.columns

    if (isinstance(cols, str)):
        cols = [cols]

    newdf = df.copy()
    num_rows = newdf.shape[0]
    num_missing = round(num_rows * p)
    for cl in cols:
        idx = np.random.choice(num_rows, num_missing, replace=False).tolist()
        # FIXME: deprecated
        # newdf = newdf.set_value(idx, cl, None)
        newdf.at[idx,cl] = None

    return (newdf)

def select_and_sort(x, select=lambda x: x):
    """Sort a list with priority given to non None and non NaN values

        Parameters
        ----------
        x      : list of item
        select : selector of the items in x

        Returns
        -------
        x_new  : sorted list

    """
    clean = list(filter(lambda item:
                        (select(item) is not None and select(item) is not np.nan), x))

    dirty = list(filter(lambda item: (select(item) is  None or select(item) is np.nan), x))
    result = sorted(clean) + dirty
    return (result)


#---------------------------------------------------------------------
# TORM: Debug print functions for Debug Transformer
#---------------------------------------------------------------------
def show_row(X, idx, text=None):
    """ show a row of a data frame of ndarray

        Parameters
        ----------
        X: data frame or ndarray
        idx: row index

    """
    if text is not None:
        print(text, "=" * 40)
    if isinstance(X, pd.core.frame.DataFrame):
        print(X.iloc[idx])
    elif isinstance(X, np.ndarray):
        print(X[idx])
    else:
        raise ValueError("input is neither a data frame nor numpy array")

def show_missing(X, text=None):
    """ show number of missing values

        Parameters
        ----------
        X: data frame or ndarray
        text: print header 
    """
    # X is a data frame
    if text is not None:
        print("=" * 20, text, "=" * 20)
    print(ex.count_missing(X))

def remove_values(df, col, values, dropna=True):
    """ filter the given values from a given column

        Parameters
        ----------
        df : data frame 
        col : column name
        values : values to be filtered
        dropna : to drop NA values or not
    """
    # remove if true
    rmidx = df[col].isin(values)
    if dropna == True:
        nanidx = df[col].isna()
        # union of nanidx and rmidx
        rmidx = list(set().union(nanidx, rmidx))
    return df[~rmidx]

#----------------------------------------------------------------------
# Extract
#----------------------------------------------------------------------
def extract_feature_target(df, target, todf=True, exclusions=None):
    """
        Separate feature set and target as ndarrays

        Parameters
        ----------
        df: data frame
        target: column name in string

        Returns
        -------
        X: All columns except target 
        y: target column
        features: list of column names except target

    """

    if exclusions is None:
        exclusions= []
    all_columns = list(set(df.columns) - set(exclusions))
    features = list(set(all_columns) - set([target]))

    # convert as ndarray
    # dtype changed - all object
    if todf == False:
        # y = df[[target]].ix[:, 0].values
        y = df[[target]].to_numpy()
        y = y.reshape(-1)
        X = df[features].values
    else:
        y = df[[target]]
        X = df.loc[:, df.columns != target]

    return (X, y, features)

#----------------------------------------------------------------------
# split training / testing data for data frame
#----------------------------------------------------------------------
def df_train_test_split(df, target, test_size=0.2, dfout=True):
    """
        train test split for data frame

    INPUTS:
        df: data frame
        target: column name of target
        test_size: fraction of test samples
        df: returns df if True

    OUTPUTS:
        tuples of (X_train, y_train, X_test, y_test)
        where
        X_train, X_test: data frame or numpy
        y_train, y_test: numpy

    """
    X = df.iloc[:, df.columns != target]
    columns = X.columns
    X = X.to_numpy()
    y = df[target].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    if dfout:
        return (pd.DataFrame(X_train, columns=columns), y_train, pd.DataFrame(X_test, columns=columns), y_test)
    else:
        return (X_train, y_train, X_test, y_test)

# FIXME: program to the abstraction...
def create_feature_tables(feature_df, target_df, all_df,
                          train_ref_month, test_ref_month,
                          id='customer_id', time_id = 'ref_month', target_col='target_f',
                          train_multiplier=1.0, test_multiplier=None, combine=True):
    "create cohorts for training"

    # determine time frame
    ref_month = train_ref_month + test_ref_month

    target_df[target_col] = 1

    # filter by date: target
    target_train_df = target_df[target_df[time_id].isin(train_ref_month)]
    target_test_df = target_df[target_df[time_id].isin(test_ref_month)]

    non_target_df = all_df[all_df[time_id].isin(ref_month)]
    non_target_df = non_target_df[~non_target_df[id].isin(target_df[id])]

    non_target_df[target_col] = 0

    non_target_train_df = non_target_df[non_target_df[time_id].isin(train_ref_month)]
    non_target_test_df = non_target_df[non_target_df[time_id].isin(test_ref_month)]

    # non-target sample ratio
    if train_multiplier is not None:
        frac = train_multiplier * len(target_train_df) / len(non_target_train_df)
        frac = min(frac, 1.0)
        non_target_train_df = non_target_train_df.sample(frac=frac)

    # non-target sample ratio
    if test_multiplier is not None:
        frac = test_multiplier * len(target_test_df) / len(non_target_test_df)
        frac = min(frac, 1.0)
        non_target_test_df = non_target_test_df.sample(frac=frac)

    # concatenate
    trainData = pd.concat([target_train_df[[id, time_id, target_col]], non_target_train_df[[id, time_id, target_col]]])
    testData = pd.concat([target_test_df[[id, time_id, target_col]], non_target_test_df[[id, time_id, target_col]]])
    # inner join with feature
    trainData = pd.merge(trainData, feature_df, on=[id, time_id])
    testData = pd.merge(testData, feature_df, on=[id, time_id])

    # join back
    if combine == True:
        df = pd.concat([trainData, testData])
        return df
    else:
        return (trainData, testData)

def sort_dict(d, sort_key=lambda item: item[1]):
    "sort dictionary by values"
    return {k: v for k, v in sorted(d.items(), key=sort_key)}

def reverse_dict(d):
    return dict(zip(d.values(), d.keys()))

def npslice(arr, begin, size):
    """ tf.slice like function for ndarray

        Examples
        --------
        t = tf.constant([[[1, 1, 1], [2, 2, 2]],
                    [[3, 3, 3], [4, 4, 4]],
                    [[5, 5, 5], [6, 6, 6]]])
        tf.slice(t, [1, 0, 0], [1, 1, 3])  # [[[3, 3, 3]]]
        tf.slice(t, [1, 0, 0], [1, 2, 3])  # [[[3, 3, 3],
                                        #   [4, 4, 4]]]
        tf.slice(t, [1, 0, 0], [2, 1, 3])  # [[[3, 3, 3]],
                                        #  [[5, 5, 5]]] 
    """
    ending = [sum(x) for x in zip(begin, size)]
    rng = list(zip(begin, ending))
    idx = [slice(*list(tup)) for tup in rng]   
    return arr[tuple(idx)]

def one_hot_encoded(class_numbers, num_classes=None):
    """
    Generate the One-Hot encoded class-labels from an array of integers.

    For example, if class_number=2 and num_classes=4 then
    the one-hot encoded label is the float array: [0. 0. 1. 0.]

    Parameters
    ----------
    :param class_numbers:
        Array of integers with class-numbers.
        Assume the integers are from zero to num_classes-1 inclusive.

    :param num_classes:
        Number of classes. If None then use max(class_numbers)+1.

    :return:
        2-dim array of shape: [len(class_numbers), num_classes]
    """

    # Find the number of classes if None is provided.
    # Assumes the lowest class-number is zero.
    if num_classes is None:
        num_classes = np.max(class_numbers) + 1

    return np.eye(num_classes, dtype=float)[class_numbers]

def one_hot_decoded(X):
    """
        One Hot Decoded: reverse of one_hot_encoded
        Parameters
        ----------
        X: 2D matrix in which only a single 1 in each row

        Returns
        -------
        list of integers
    """
    return [np.argmax(row, axis=0) for row in X]

def col_upper_case(df, inplace=True):
    """Upper case all column names

        Parameters
        ----------
        df : data frame

        Returns
        --------
        df : data frame

    """
    if inplace==False:
        df = df.copy()
    data_cols = df.columns.values.tolist()
    df.columns = [col.upper() for col in data_cols]

    return df

def get_single_column(df, col):
    """extract a flatten array

        Parameters
        ----------
        df : data frame
        col : column name in string

        Returns
        --------
        ndarray

    """
    return np.ravel(df[col])

def reframe(df, column_names):
    """create data frame with exact columns

       if column is missing in df, create an empty column
       extra columns in df are removed

        Parameters
        ----------
        df : data frame
        column_names : column names in string

        Returns
        --------
        df : data frame

    """
    emptydf = pd.DataFrame(columns=column_names)
    return emptydf.append(df)[column_names]