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

def summary(df, max_distinct=15, cutoff=0.05):
    result = pd.DataFrame()
    numericalCols = get_numerical_column(df)
    categoricalCols = get_categorical_column(df, max_distinct=max_distinct, cutoff=cutoff)
    nonCategoricalCols = get_non_categorical_column(df, max_distinct=max_distinct, cutoff=cutoff)
    d = {
        'unique_values': count_unique_values(df),
        'num_missing': count_missing(df),
        'numerical': dict(zip(df.columns, map(lambda x: x in numericalCols, list(df.columns)))),
        'categorical': dict(zip(df.columns, map(lambda x: x in categoricalCols, list(df.columns)))),
        'non_categorical': dict(zip(df.columns, map(lambda x: x in nonCategoricalCols, list(df.columns)))),
    }
    return pd.DataFrame(data=d)

    
#============================================================
# plotting
#============================================================
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


