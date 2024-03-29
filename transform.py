"""
Author: Francis Chan

Data Transformation for building classification models

"""

from sklearn.base import TransformerMixin, BaseEstimator, clone
import random
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, OrdinalEncoder
from functools import reduce
from datetime import datetime
import explore as ex
import utility as ut
import varprofile as vp

# TODO: my own multiclass WoE transformer
# http://contrib.scikit-learn.org/category_encoders/woe.html



# ----------------------------------------------------------------------------
# Imputer
# ----------------------------------------------------------------------------

class DataFrameImputer(BaseEstimator, TransformerMixin):
    """
        Impute missing values

        Parameters
        -----------
        categorical_strategy : fill with most frequent
        numerical_strategy : fill with mean value
        numeric_const : numeric filled value
        categorical_const : categorical filled value
    """

    def __init__(self, categorical_strategy = 'constant', numerical_strategy= 'constant',
                 numeric_const = 0, categorical_const = 'Unknown'):

        self.c_strategy = categorical_strategy
        self.n_strategy = numerical_strategy
        self.categorical_const = categorical_const
        self.numerical_const = numeric_const

    def fit(self, X, y=None):
        """Preparation steps for transformation

        Columns of categorical object are imputed with the most frequent value
        in column.

        Columns of numerical data are imputed with mean of column.

        Parameters
        ----------
        X     : pandas data frame
        """

        # all of these return a single value for all missing
        def fill_most_frequent(x):
            "return the most freq value in this column"
            try:
                value = (x.value_counts().index[0])
            except:
                value = self.categorical_const
            return value
            
        def fill_mean(x):
            "return the mean value in this column"
            return(x.mean())

        def fill_median(x):
            "return the median value in this column"
            return(x.median())

        def fill_categorical_const(x):
            "return the constant categorical value"
            return self.categorical_const

        def fill_numerical_const(x):
            "return the constant numerical value"
            return self.numerical_const

        # strategy selector for categorical
        if self.c_strategy == 'constant':
            self.c_func = fill_categorical_const
        else:
            self.c_func = fill_most_frequent

        # strategy selector for numerical
        if self.n_strategy == 'most_frequent':
            self.n_func = fill_most_frequent
        elif self.n_strategy == 'median':
            self.n_func = fill_median
        elif self.n_strategy == 'constant':
            self.n_func = fill_numerical_const
        else:
            self.n_func = fill_mean

        # X is data frame
        # categorical_columns = ex.get_categorical_column(X)
        # non_categorical_columns = ex.get_non_categorical_column(X)
        categorical_columns = ex.get_column_for_type(X, "object")
        non_categorical_columns = ex.get_column_for_type(X, "object", True)

        # find the values to impute for each column
        self.fill = pd.Series([self.c_func(X[c])
                               if c in categorical_columns
                               else self.n_func(X[c]) for c in X],
                              index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)

# ----------------------------------------------------------------------------
# Combiner
# ----------------------------------------------------------------------------
class Combiner(BaseEstimator, TransformerMixin):
    """Concatenate data frames by columns

        Parameters
        --------
        X : list of data frames
    
        Returns
        --------
        concatenated data frame
    """

    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return reduce((lambda x, y: pd.concat([x, y.reindex(x.index)], axis=1)), X)

class SequentialTransformer(BaseEstimator, TransformerMixin):
    """Sequential pipeline transform

        Transform the same data frame with multiple pipelines

        Parameters
        --------
        pipelines : list of pipelines
    
        Returns
        --------
        list of pipeline outputs
    """
    def __init__(self, pipelines):
        self.pipelines = pipelines
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        out = []
        for pline in self.pipelines:
            out.append(pline.fit_transform(X))
        return out

# TODO: for review
# class DataFrameFeatureUnion(BaseEstimator, TransformerMixin):
#     """ A DataFrame transformer that unites several DataFrame transformers
    
#     Fit several DataFrame transformers and provides a concatenated
#     Data Frame
    
#     Parameters
#     ----------
#     list_of_transformers : list of DataFrameTransformers
        
#     """ 
#     def __init__(self, list_of_transformers):
#         self.list_of_transformers = list_of_transformers
        
#     def transform(self, X, **transformparamn):
#         "Applies the fitted transformers on a DataFrame"
        
#         concatted = pd.concat([transformer.transform(X)
#                             for transformer in
#                             self.fitted_transformers_], axis=1).copy()
#         return concatted


#     def fit(self, X, y=None, **fitparams):
#         self.fitted_transformers_ = []
#         for transformer in self.list_of_transformers:
#             fitted_trans = clone(transformer).fit(X, y=None, **fitparams)
#             self.fitted_transformers_.append(fitted_trans)
#         return self

# ----------------------------------------------------------------------------
# Extractor
# ----------------------------------------------------------------------------
class DateExtractor(BaseEstimator, TransformerMixin):
    """
        Extract Day, Month and Year of the given date columns

        Parameters
        ----------
        datacolumns: list of date columns
        fmt : strptime date format, e.g. "%d-%m-%Y"

        Returns data frame columns
        -------
        _Date: day of month
        _Month: month 
        _Year: year
        _Weekday: Monday is 1
    """

    def __init__(self, datecolumns, fmt):
        self.cols = datecolumns
        self.fmt = fmt
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        "expand date columns into weekday, day, month, year"
        df = X.copy()
        to_day = lambda x: datetime.strptime(x, self.fmt).day
        to_month = lambda x: datetime.strptime(x, self.fmt).month
        to_year = lambda x: datetime.strptime(x, self.fmt).year
        to_weekday = lambda x: datetime.strptime(x, self.fmt).isoweekday()
        funcs = [to_day, to_month, to_year, to_weekday]
        date_val = ['_Day', '_Month', '_Year', "_Weekday"]
        for col in self.cols:
            names = [col + d for d in date_val]
            combo = zip(names, funcs)
            for (field, f) in combo:
                df[field] = df[col].apply(f)
        return df

# ----------------------------------------------------------------------------
# Remover
# ----------------------------------------------------------------------------
class DropColumns(BaseEstimator, TransformerMixin):
    """Drop columns
    
    Parameters
    ----------
    excl : list of columns to be dropped
    """

    def __init__(self, cols=['TARGET_F', 'CUSTOMER_ID', 'REF_MONTH', 'DATA_DT', 'PROCESS_DTTM', 'ACCT_KEY', 'ACCOUNT_ID', 'GCIS_KEY']):
        self._cols = cols 

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # cols_to_drop = list(intersect(set(X.columns), set(self._excl)))
        cols_to_drop = list((set(X.columns).intersection(self._cols)))
        return X.drop(cols_to_drop, axis=1, inplace=False)

class ValueRemover(BaseEstimator, TransformerMixin):
    """
        Remove rows for a given list of values
    
        Parameters
        ----------
        cols: list of columns for filtering
        values: list of values to be removed

        Returns
        -------
        dataframe filtered by the values
    
    """
    def __init__(self, cols, values):
        # both cols and values are lists
        self.cols = cols
        self.values = values
    def fit(self, X, y=None):
        self.X = X
        return self
    def transform(self, X):
        rmidx = self.X[self.cols].isin(self.values)
        return self.X[~rmidx.values]


# ----------------------------------------------------------------------------
# Creator
# ----------------------------------------------------------------------------
class CreateDataFrame(BaseEstimator, TransformerMixin):
    """
        Create Data Frame from ndarray or list

        Parameters
        ----------
        col    : list of column names
        X      : ndarry or list
    """
    def __init__(self, col):
        self.col = col
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        df = pd.DataFrame(X)
        df.columns = self.col
        return df

# ----------------------------------------------------------------------------
# Encoder
# ----------------------------------------------------------------------------
# Encode all categorical columns of data frame

class DummyEncoder(BaseEstimator, TransformerMixin):
    """Transform Categorical variables to Dummy
        Returns
        --------
        data frame
    """
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return pd.get_dummies(X)


class TargetEncoder(BaseEstimator, TransformerMixin):
    """Encode single column 
    
        usually for target column
        same capability as LabelEncoder
        adjust the interface of LabelEncoder to fit(X, y) instead of fit(y)

        Parameters
        ----------
        colname  : name of the column to transform
        mapping : optional (use Label Encoder if not given)

        Returns
        -------
        ndarray
    
    """
    def __init__(self, colname, mapping=None):
        self.encoder = LabelEncoder()
        self.colname = colname
        self.map = mapping
    def fit(self, X, y=None):
        self.encoder.fit(X[self.colname])
        return self
    def transform(self, X):
        if self.map is not None:
            do_transform = lambda x: self.map[x]
            return X[self.colname].apply(do_transform).to_numpy()
        else:
            return self.encoder.transform(X[self.colname])

class CategoricalEncoder(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self
    def transform(self, X):
        enc = OrdinalEncoder()
        categoricalCols = ex.get_column_for_type(X, 'object')
        # x is a series object
        do_transform = lambda x: enc.fit_transform(ut.unravel(x)).ravel() if x.name in categoricalCols else x
        # Encoding and return results
        return X.apply(do_transform)

# FIXME
class WoeMultiClassEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, target_col, num_bins=10):
        self.target_col = target_col
        self.num_bins=num_bins
    def fit(self, X, y=None):
        self.profiler = Profiler(X, self.target_col)
        return self
    def transform(self, X):
        enc = OrdinalEncoder()

# ----------------------------------------------------------------------------
# Selector
# ----------------------------------------------------------------------------
class DataFrameSelector(BaseEstimator, TransformerMixin):
    """Data Frame Selector
    
        Select columns from data frames

        Parameters
        ----------
        col_names : column name in string

        Returns
        --------
        filtered data frame
    """
    def __init__(self, col_names):
        self.col_names = col_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.col_names]

class DTypeSelector(BaseEstimator, TransformerMixin):
    """
        Select columns with specific types:
        Parameters
        ----------
        X : must be a data frame
        col_type : name of numpy dtypes in string
        unselect: reverse selection criteria

        Returns
        --------
        either a data frame of ndarray
    """

    def __init__(self, column_type, unselect=False):
        self.column_type = column_type
        self.unselect = unselect
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        attribute_names = ex.get_column_for_type(X, self.column_type, self.unselect)
        # dftype = X.dtypes
        # if self.unselect == True:
        #     attribute_names = list(dftype.index[dftype != self.column_type])
        # else:
        #     attribute_names = list(dftype.index[dftype == self.column_type])
        return X[attribute_names]

class TypeSelector(BaseEstimator, TransformerMixin):
    """
        Select columns with specific types:
        Parameters
        ----------
        X : data frame
        col_type : binary, categorical, factor (binary + categorical) or numerical
        todf: boolean (the returned type is df or not)

        Returns
        --------
        either a data frame of ndarray
    """

    def __init__(self, column_type, todf=True):
        self.column_type = column_type
        self.todf = todf
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        dftype = ex.get_column_type(X)
        if self.column_type == 'factor':
            attribute_names = dftype['categorical'] + dftype['binary']
        else:
            attribute_names = dftype[self.column_type]
        if self.todf:
            return X[attribute_names]
        else:
            return X[attribute_names].values

# ----------------------------------------------------------------------------
# Transformer
# ----------------------------------------------------------------------------
class HybridTransformer(BaseEstimator, TransformerMixin):
    """Transform conditional on numerical or categorical types
    
        categorical definition is 'object' dtype
        Parameters
        ----------
        X: must be a dataframe (for dtype identification)
        
        Returns
        --------
        data frame
    """
    def __init__(self, ntransformer, ctransformer):
        self.ntransformer = ntransformer
        self.ctransformer = ctransformer
    def fit(self, X, y=None):
        self.categoricalCols = ex.get_column_for_type(X, 'object')
        return self
    def transform(self, X):
        # conditionally transform numeric and categorical variables
        do_transform = lambda x: self.ctransformer().fit_transform(ut.unravel(x)).ravel() \
            if x.name in self.categoricalCols else self.ntransformer().fit_transform(ut.unravel(x)).ravel()
        # Encoding and return results
        return X.apply(do_transform)

class WoeTransformer(BaseEstimator, TransformerMixin):
    """Transform all categorical columns into woe

        Parameters
        ----------
        X: must be a dataframe 
        
        Returns
        --------
        data frame

    """
    def __init__(self, target_col):
        self.target_col = target_col
        self.profiler = None
    def fit(self, X, y=None):
        self.profiler = vp.Profiler(X, self.target_col)
        return self
    def transform(self, X=None):
        return self.profiler.woe_encode()

class SupervisedTransformer(BaseEstimator, TransformerMixin):
    """Wrapped Transformer into a pipeline
    
        Parameters
        ----------
        transformer : instantiated transformer
        X: must be a dataframe (for dtype identification)
        y: binary target (numeric)
        
        Returns
        --------
        data frame
    """
    def __init__(self, transformer, y):
        self.transformer = transformer
        self.y = y
    def fit(self, X, y=None):
        if y is None:
            y = self.y
        self.transformer.fit(X, y)
        return self
    def transform(self, X):
        return self.transformer.transform(X)

# class GuidedHybridTransformer(TransformerMixin):
#     """Transform conditional on numerical or categorical types
    
#         Parameters
#         ----------
#         X: must be a dataframe (for dtype identification)
        
#         Returns
#         --------
#         data frame
#     """
#     def __init__(self, y, ntransformer, ctransformer):
#         self.ntransformer = ntransformer
#         self.ctransformer = ctransformer
#         self.y = y
#     def fit(self, X, y=None):
#         self.categoricalCols = ex.get_column_for_type(X, 'object')
#         if y is None:
#             y = self.y
#         self.ntransformer().fit(X, y)
#         self.ctransformer().fit(X, y)
#         return self
#     def transform(self, X):
#         # conditionally transform numeric and categorical variables
#         do_transform = lambda x: self.ctransformer.transform(ut.unravel(x)).ravel() \
#             if x.name in self.categoricalCols else self.ntransformer.transform(ut.unravel(x)).ravel()
#         # Encoding and return results
#         return X.apply(do_transform)


class UpperCaseColumn(BaseEstimator, TransformerMixin):
    """Upper Case all column names
    
        Returns
        --------
        data frame
    """
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return ut.col_upper_case(X, False)

class Scaler(BaseEstimator, TransformerMixin):
    """Standard Scaler
    
        Returns
        --------
        data frame
    """
    def fit(self, X, y=None):
        if isinstance(X, pd.core.frame.DataFrame):
            self.columns = X.columns
        self.scaler = StandardScaler().fit(X)
        return self
    def transform(self, X):
        out = self.scaler.transform(X)
        if isinstance(X, pd.core.frame.DataFrame):
            return pd.DataFrame(out, columns=self.columns)
        else:
            return out

class FunctionTransformer(BaseEstimator, TransformerMixin):
    """
        Apply a function to all columns of a df
        e.g.
        t = tr.FunctionTransformer(ex.get_distinct_value, 'Cl_Toilets', 'Cl_Scenic_View')

        Parameters
        ----------
        X : dataframe
        fun: function object
        args: argument to the function 

    """

    def __init__(self, fun, *args):
        self.transformer = fun
        self.args = args

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        result = self.transformer(X, self.args)
        return result


class TransformDebugger(BaseEstimator, TransformerMixin):
    """
        Transformer Wrapper
        Embed a transformer and print debug information

        Parameters
        ----------
        transformer: any transformer
        debug_func : print function for debug (see utility lib)
        cols       : columns to show when debug

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)

    """

    def __init__(self, transformer, debug_func, cols=None):
        self.transformer = transformer
        self.debug_func = debug_func
        self.cols = cols

    def set_col(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def _debug_print(self, X):
        "print debug information"
        print(self.transformer.__class__.__name__)
        # select a random row of data
        idx = random.randrange(0, len(X))

        # show only specific columns
        # fill in all columns if not specified
        if isinstance(X, pd.core.frame.DataFrame):
            if self.cols is None:
                cols = X.columns
            else:
                cols = self.cols
            XX = X[cols]
        elif isinstance(X, np.ndarray):
            if self.cols is None:
                self.cols = range(X.shape[1])
            else:
                cols = self.cols
            XX = X[:, cols]
        else:
            raise ValueError("input is neither a data frame nor numpy array")

        # call the debug function
        self.debug_func(XX, idx, "Before")
        XX = self.transformer.transform(XX)
        self.debug_func(XX, idx, "After")

    def transform(self, X):
        "transform routine for pipeline"
        self._debug_print(X)
        return X
