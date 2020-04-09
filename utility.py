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
# def load_csv(source, row=None, na_values=['?', '']):
#     df = pd.read_csv(source, nrows=row, na_values=na_values)
#     return df


def load_td(source, cnxn, row=None):
    "load single table from td"
    # print("Loading TD data from {}...\n".format(source))
    if row is not None:
        sql = "select top {} * from {}".format(row, source)
    else:
        sql = "select * from {}".format(source)

    df = pd.read_sql(sql, cnxn)
    return df

def connect_td():
    "get td cursor"
    pwd = os.environ['TDPASS']
    uid = getpass.getuser()
    connectparam = "DSN=tdp5;UID={};PWD={}".format(uid, pwd)
    cnxn = pyodbc.connect(connectparam)
    return cnxn
        
# deprecated: use pd.melt
# def widedf_to_talldf(df, id_vars=['Segment']):
#     """
#         convert wide to tall data frame
#         columns in id_vars are suppressed
#     """
    
#     df.reset_index(level=0, inplace=True)
#     df = pd.melt(df, id_vars=id_vars)
#     return df


# def discretize_column(df, colname, append_column=True, num_bins=3, suffix ='GRP', group_labels=['Low', 'Mid', 'High']):
#     "bin single column into groups and append to df"
    
#     # assume this is not a categorical column
#     tmp = pd.qcut(df[colname], num_bins, retbins=False, duplicates='drop')
#     if len(tmp.cat.categories) == len(group_labels):
#         tmp.cat.categories = group_labels          
#     if not append_column:
#         return tmp
#     else:
#         newcolname = colname + '_' + suffix
#         df[newcolname] = tmp
#         return df

def discretize_column(df, colname, append_column=True, num_bins=3, suffix ='GRP', group_labels=None):
    "bin single column into groups and append to df"
    
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
    "discretize subset of columns"
    
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
    

#------------------------------------------------------------
# Date
#------------------------------------------------------------
def datestr2date(x, date_format='%Y-%m-%d'):
    """
        convert date string to date object

        Parameters
        ----------
        x: string or datetime object

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
        current_month: datetime or string object
        latency: integer
        format: date format

        Returns
        -------
        datetime object
    """
    if isinstance(current_month, str):
        current_month = datetime.strptime(current_month, format).date()

    shifted_month = current_month + dateutil.relativedelta.relativedelta(months=latency)
    return (shifted_month)

# Deprecated: use  df['col'].apply(ut.shift_by_month, args=(2,))
# def shift_month(df, col, latency):
#     "shift the month of the date column"

#     shift_by_latency = functools.partial(shift_by_month, latency=latency)
#     newdf = df.copy()
#     newdf[col] = newdf[col].map(shift_by_latency)
#     return (newdf)


class DiscreteSampler:
    """
        Discrete value sampler
        Parameters
        ----------
        populations: list of finite values
        weights: weight assigned to the populations

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
        # FIXME: deprecated
        # newdf = newdf.set_value(idx, cl, None)
        newdf.at[idx,cl] = None

    return (newdf)

def select_and_sort(x, select=lambda x: x):
    """Sorting: 

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
# Debug print functions for Debug Transformer
#---------------------------------------------------------------------
def show_row(X, idx, text=None):
    # X is a data frame
    if text is not None:
        print(text, "=" * 40)
    if isinstance(X, pd.core.frame.DataFrame):
        print(X.iloc[idx])
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

# deprecated: use transformer
# def get_LabelBinarizer_index(lb, target):
#     "get the encoded index of the specified target"

#     classes = lb.classes_
#     # to determine the encoded target index
#     try:
#         idx = int(np.where(lb.classes_ == target)[0])
#     except:
#         idx = -1
#     return (idx)


# def split_train_test_by_id(data, test_ratio, id_column):

#     def test_set_check(identifier, test_ratio):
#         return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32

#     ids = data[id_column]
#     in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
#     return data.loc[~in_test_set], data.loc[in_test_set]


def npslice(arr, begin, size):
    """
        tf.slice for numpy

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