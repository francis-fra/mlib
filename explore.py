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
    """ Return a dtype dictionary 

        Parameters
        ----------
        df: data frame

        Returns
        -------
        {dtype: [list of variable names]}
    """
    
    uniquetypes = set(df.dtypes)
    typeDict = {}
    for dt in uniquetypes:
        typeDict[str(dt)] = [c for c in df.columns if df[c].dtype == dt]
    return (typeDict)

# def get_column_type(df, col):
#     """Return dtype of the specified column"""
#     return (df[col].dtype)

def get_column_for_type(df, data_type, unselect=False):
    """ get column names of the specified type

        Parameters
        ----------
        df: data frame
        data_type: string
        unselect: inverse selection if True

        Returns
        -------
        [list of column names]
    """
    if unselect == False:
        return [col for col in df.dtypes.index if df.dtypes.get(col) == data_type ]
    else:
        return [col for col in df.dtypes.index if df.dtypes.get(col) != data_type ]

def get_type_tuple(df):
    """dtype tuple

        Parameters
        ----------
        df: data frame

        Returns
        -------
        Return a list [(colunm name, datatype)]

    """
    return([(c, df[c].dtype) for c in df.columns])


def get_categorical_column(df, exclusions=None, index=False, max_distinct=15, cutoff = 0.01):
    """
        Return [list of categorical column]

        Parameters
        ----------
        df: data frame
        exculsions : list of column names to be excluded
        index : boolean, true if return the index of categorical column
        max_distinct : a column is categorical if num distinct values is less than max_distinct
        cutoff : a column is categorical if proportion of distinct values is less than the cutoff
        
        Categorical column is either:
        1) non-numerical
        2) numeric but small number of finite values
        3) including binary

        Returns
        -------
        result : either a list of index or list of columns
    
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
    """
        Return [list of non categorical column]
        
        Parameters
        ----------
        df: data frame
        exculsions : list of column names to be excluded
        index : boolean, true if return the index of categorical column
        max_distinct : a column is categorical if num distinct values is less than max_distinct
        cutoff : a column is categorical if proportion of distinct values is less than the cutoff
        
        Categorical column is either:
        1) non-numerical
        2) numeric but small number of finite values

        Returns
        -------
        result : either a list of index or list of columns
    """
    
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
    """Return [list of numerical column]
    
        Parameters
        ----------
        df: data frame
        index: boolean

        Returns
        -------
        Return [list of numerical column]
    """
    
    # cols = df._get_numeric_data().columns.tolist()
    cols = df.select_dtypes([np.number]).columns
    if index == True:
        return [df.columns.get_loc(x) for x in cols]
    else:
        return (cols)
    
def get_non_numerical_column(df, index=False):
    """Return [list of non numerical column]
    
        Parameters
        ----------
        df: data frame
        index: boolean

        Returns
        -------
        Return [list of numerical column]
    """
    
    cols = list(set(df.columns) - set(df._get_numeric_data().columns.tolist()))
    if index == True:
        return [df.columns.get_loc(x) for x in cols]
    else:
        return (cols)

def get_column_type(df, max_distinct=15, cutoff=0.01):
    """
        Return dict of type: binary, categorial or Numerical
        Parameters
        ----------
        df : dataframe

        Returns
        -------
        {binary: [list of cols], 
        categorical: [list of cols], 
        numerical: [list of cols]}

    """

    unique_map = count_unique_values(df)
    # binary col are categorical but has only two distinct values
    categorical_col = get_categorical_column(df, max_distinct=max_distinct, cutoff=cutoff)
    numerical_col = list(set(df.columns) - set(categorical_col))
    
    binary_col = [ col for col in categorical_col if unique_map[col] == 2]
    categorical_col = list(set(categorical_col) - set(binary_col))

    return {'binary': binary_col, 'categorical': categorical_col, 'numerical': numerical_col}

#----------------------------------------------------------------------
# column type 
#----------------------------------------------------------------------
def is_column_numeric(df, column):
    return column in df.select_dtypes([np.number]).columns

def is_column_categorical(df, column, max_distinct=15, cutoff=0.01):
    is_non_numeric = not is_column_numeric(df, column)
    if is_non_numeric:
        return True
    else:
        num_unique_values = count_unique_values(df, prop=False, subset=[column])[column]
        unique_values_proportion = count_unique_values(df, prop=True, subset=[column])[column]
        if num_unique_values < max_distinct and unique_values_proportion < cutoff:
                return True
    return False

def is_column_binary(df, column):
    # is_non_numeric = not is_column_numeric(df, column)
    num_unique_values = count_unique_values(df, prop=False, subset=[column])[column]
    if num_unique_values == 2:
        return True
    else:
        return False

#----------------------------------------------------------------------
# Count
#----------------------------------------------------------------------

def get_distinct_value(df, cols=None):
    """
        Return list of distinct values of a dataframe
        Parameters
        ----------
        df : dataframe
        col: string

        Returns
        -------
    """

    if not cols:
        cols = df.columns

    return dict(( col, df[col].unique()) for col in cols)

def count_unique_values(df, prop=False, subset=None, dropna=True):
    """
        Return a dictionary of num unique values for each column

        Returns
        -------
        {column name, num_unique_values}
    """
    
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
    """
        Count the number of occurences for each unique value
        for each column
    
        Returns
        -------
        {column_name: list((unique_value, count)}
    """
    
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
            # FIXME: replace robust sort
            result[c] = ut.select_and_sort(list(zip(x.index, x.values / total)), g)
        else:
            result[c] = ut.select_and_sort(list(zip(x.index, x.values)), g)
        
    return result

def any_missing(df):
    """predicate if any column has missing values"""
    cnt = count_missing(df)
    return sum(cnt.values()) > 0

def get_missing_columns(df):
    "get list of columns with missing values"
    cnt = count_missing(df)
    return [k for k in cnt.keys() if cnt[k] > 0]

    
def count_missing(df, prop=False): 
    """Count a dictionary of the number of missing data in each column"""
    
    if prop == False:
        return (df.isnull().sum().to_dict())
    else:
        result = df.isnull().sum() / df.shape[0]
        return (result.to_dict())


def summary(df, max_distinct=15, cutoff=0.05):
    "statistical summary of the data frame"

    result = pd.DataFrame()
    coltype = get_column_type(df)
    numericalCols = coltype['numerical']
    categoricalCols = coltype['categorical']
    binaryCols = coltype['binary']
    d = {
        'unique_values': count_unique_values(df),
        'num_missing': count_missing(df),
        'numerical': dict(zip(df.columns, map(lambda x: x in numericalCols, list(df.columns)))),
        'categorical': dict(zip(df.columns, map(lambda x: x in categoricalCols, list(df.columns)))),
        'binary': dict(zip(df.columns, map(lambda x: x in binaryCols, list(df.columns)))),     
    }
    return pd.DataFrame(data=d)
    