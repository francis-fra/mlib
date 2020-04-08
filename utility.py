"""
Author: Francis Chan 

Utility functions for model building

"""
import os, sys, random
import bisect
from zlib import crc32
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import functools
from datetime import datetime
import dateutil.relativedelta

from sklearn.preprocessing import label_binarize, LabelBinarizer
import explore as ex

#-------------------------------------------------------------------------
# Loading Data
#-------------------------------------------------------------------------
def load_csv(source, row=None, na_values=['?', '']):
    df = pd.read_csv(source, nrows=row, na_values=na_values)
    return df


def load_td(source, cnxn, row=None):
    # print("Loading TD data from {}...\n".format(source))
    if row is not None:
        sql = "select top {} * from {}".format(row, source)
    else:
        sql = "select * from {}".format(source)

    df = pd.read_sql(sql, cnxn)
    return df

def connect_td():
    pwd = os.environ['TDPASS']
    uid = getpass.getuser()
    connectparam = "DSN=tdp5;UID={};PWD={}".format(uid, pwd)
    cnxn = pyodbc.connect(connectparam)
    return cnxn
        
def widedf_to_talldf(df, id_vars=['Segment']):
    "convert wide to tall data frame"
    
    df.reset_index(level=0, inplace=True)
    df = pd.melt(df, id_vars=id_vars)
    
    return df


def discretize_column(df, colname, append_column=True, num_bins=3, suffix ='GRP', group_labels=['Low', 'Mid', 'High']):
    "bin columns into groups and append to df"
    
    # assume this is not a categorical column
    tmp = pd.qcut(df[colname], num_bins, retbins=False, duplicates='drop')
    if len(tmp.cat.categories) == len(group_labels):
        tmp.cat.categories = group_labels          
    if not append_column:
        return tmp
    else:
        newcolname = colname + '_' + suffix
        df[newcolname] = tmp
        return df

def discretize_dataframe(df, varList=None, return_all=False, num_bins=3, suffix ='GRP', group_labels=['Low', 'Mid', 'High']):
    "discretize subset of columns"
    
    if varList is None:
        varList = df.columns
        
    categorical_cols = ex.get_categorical_column(df[varList])
    frames = []
    
    for var in varList:
        print('Binning {}'.format(var))
        if var in categorical_cols:
            frames.append(df[var])
        else:
            tmp = discretize_column(df, var, False, num_bins, suffix, group_labels)
            tmp.name = var + '_' + suffix
            frames.append(tmp)    

    # new frames
    result = pd.concat(frames, axis=1)
    if not return_all:
        return result
    else:
        return pd.concat([df, result], axis=1)
    

def datestr2date(x, date_format='%Y-%m-%d'):
    "convert date string to date object"

    if isinstance(x, list):
        return [datetime.strptime(item, date_format).date() for item in x]
    else:
        return datetime.strptime(x, date_format).date()


def shift_month(df, col, latency):
    "shift the month of the date column"

    def shift_by_month(current_month, latency, format='%Y-%m-%d'):
        "shift the month by the latency"

        if isinstance(current_month, str):
            current_month = datetime.strptime(current_month, format).date()

        shifted_month = current_month + dateutil.relativedelta.relativedelta(months=latency)
        return (shifted_month)

    shift_by_latency = functools.partial(shift_by_month, latency=latency)
    newdf = df.copy()
    newdf[col] = newdf[col].map(shift_by_latency)
    return (newdf)


class DiscreteSampler:
    """Discrete value sampler"""

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
                px     : prob of each value
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
    """Return a new data frame by filling the specified columns with missing values"""

    if cols is None:
        cols = df.columns

    if (isinstance(cols, str)):
        cols = [cols]

    newdf = df.copy()
    num_rows = newdf.shape[0]
    num_missing = round(num_rows * p)
    for cl in cols:
        idx = np.random.choice(num_rows, num_missing, replace=False).tolist()
        newdf = newdf.set_value(idx, cl, None)

    return (newdf)


def robustsort(x, select=lambda x: x):
    """Sorting: Items with None are placed at the end

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
# Debug print for Debug Transformer
#---------------------------------------------------------------------
def show_row(X, idx, text=None):
    # X is a data frame
    if text is not None:
        print(text, "=" * 40)
    if isinstance(X, pd.core.frame.DataFrame):
        print(X.ix[idx])
    elif isinstance(X, np.ndarray):
        print(X[idx])
    else:
        raise ValueError("input is neither a data frame nor numpy array")

# FIXME:
# def show_count_levels(X, idx=None, text=None):
#     # X is a data frame
#     if text is not None:
#         print("=" * 20, text, "=" * 20)
#     print(ex.count_levels(X))

def show_missing(X, idx=None, text=None):
    # X is a data frame
    if text is not None:
        print("=" * 20, text, "=" * 20)
    print(ex.count_missing(X))

def remove_values(df, col, values, dropna=True):
    # remove if true
    rmidx = df[col].isin(values)
    if dropna == True:
        nanidx = df[col].isna()
        # union
    else:
        return df[~rmidx]

#----------------------------------------------------------------------
# Extract
#----------------------------------------------------------------------
def extract_feature_target(df, target, todf=True, exclusions=None):
    """Separate feature set and target as ndarrays"""

    if exclusions is None:
        exclusions= []
    all_columns = list(set(df.columns) - set(exclusions))
    features = list(set(all_columns) - set([target]))

    # convert as ndarray
    # dtype changed - all object
    if todf == False:
        y = df[[target]].ix[:, 0].values
        X = df[features].values
    else:
        y = df[[target]]
        X = df.loc[:, df.columns != target]

    return (X, y, features)

# def get_unique_values(df, col):
#     "get unique values of the given col"

#     return list(df[col].unique())

#----------------------------------------------------------------------
# split training / testing data for data frame
#----------------------------------------------------------------------

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

def get_LabelBinarizer_index(lb, target):
    "get the encoded index of the specified target"

    classes = lb.classes_
    # to determine the encoded target index
    try:
        idx = int(np.where(lb.classes_ == target)[0])
    except:
        idx = -1

    return (idx)


def split_train_test_by_id(data, test_ratio, id_column):

    def test_set_check(identifier, test_ratio):
        return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32

    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]


def npslice(arr, begin, size):
    "tf.slice for numpy"
    ending = [sum(x) for x in zip(begin, size)]
    rng = list(zip(begin, ending))
    idx = [slice(*list(tup)) for tup in rng]   
    return arr[tuple(idx)]

