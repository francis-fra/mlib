"""
Author: Francis Chan

Data Transformation for building classification models

"""

from sklearn.base import TransformerMixin, BaseEstimator
import random
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import label_binarize
from sklearn.impute import SimpleImputer
from datetime import datetime
import explore as ex
import utility as ut

class ConstantImputer(TransformerMixin):
    """Imputer Data Frame with contant values"""
    def __init__(self, numerical_columns=None, categorical_columns=None):
        self.num_cols = numerical_columns
        self.cat_cols = categorical_columns
    def fit(self, X, y=None):
        self.type_dict = ex.get_type_dict(X)
        # if self.num_cols is None:
        #     self.num_cols = self.type_dict['float64'] + self.type_dict['int64']
        if self.cat_cols is None:
            self.cat_cols = self.type_dict['object']
        if self.num_cols is None:
            self.num_cols = list(set(X.columns) - set(self.cat_cols))
        return self
    def transform(self, X,):
        num_imputer = SimpleImputer(strategy="constant", fill_value=0)
        cat_imputer = SimpleImputer(strategy="constant")
        df = X.copy()
        df[self.num_cols] = num_imputer.fit_transform(df[self.num_cols])
        df[self.cat_cols] = cat_imputer.fit_transform(df[self.cat_cols])
        return df

class DateExtractor(TransformerMixin):
    """Remove rows for the given list of values"""
    def __init__(self, datecolumns, fmt):
        self.cols = datecolumns
        self.fmt = fmt
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        "expand date columns into day, month, and year"
        df = X.copy()
        to_day = lambda x: datetime.strptime(x, self.fmt).day
        to_month = lambda x: datetime.strptime(x, self.fmt).month
        to_year = lambda x: datetime.strptime(x, self.fmt).year
        funcs = [to_day, to_month, to_year]
        date_val = ['_Day', '_Month', '_Year']
        for col in self.cols:
            names = [col + d for d in date_val]
            combo = zip(names, funcs)
            for (field, f) in combo:
                df[field] = df[col].apply(f)
        return df

class ValueRemover(TransformerMixin):
    """Remove rows for a given list of values"""
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

class ColumnRemover(TransformerMixin):
    """Remove rows for a given list of values"""
    def __init__(self, cols):
        # cols to drop
        self.cols = cols
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X.drop(self.cols, axis=1)

class CreateDataFrame(TransformerMixin):
    """Create Data Frame from numpy array"""
    def __init__(self, col):
        self.col = col
    def fit(self, X, y=None):
        # X is a ndarray
        return self
    def transform(self, X):
        df = pd.DataFrame(X)
        df.columns = self.col
        return df

class DataFrameImputer(TransformerMixin):
    """Impute missing values"""

    def __init__(self, cateogrical_strategy = 'most_frequent', numerical_strategy= 'mean',
                 numeric_const = 0, categorical_const = 'Unknown'):

        self.c_strategy = cateogrical_strategy
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
            return(x.value_counts().index[0])

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

        if self.c_strategy == 'constant':
            self.c_func = fill_categorical_const
        else:
            self.c_func = fill_most_frequent


        if self.n_strategy == 'most_frequent':
            self.n_func = fill_most_frequent
        elif self.n_strategy == 'median':
            self.n_func = fill_median
        elif self.n_strategy == 'constant':
            self.n_func = fill_numerical_const
        else:
            self.n_func = fill_mean

        # X is data frame
        categorical_columns = ex.get_categorical_column(X)
        non_categorical_columns = ex.get_non_categorical_column(X)

        # find the values to impute for each column
        self.fill = pd.Series([self.c_func(X[c])
                               if c in categorical_columns
                               else self.n_func(X[c]) for c in X],
                              index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)


# Encode all categorical columns of data frame
# class DataFrameLabelEncoder(TransformerMixin):
class DataFrameCategoricalEncoder(TransformerMixin):
    def __init__(self, cols=None, onehot=True, exclusion=[]):

        # cols: categorical column (if None, it is determined automatically)
        self.reset(cols)
        self.exclusion = exclusion

    def get_transformed_index(self, col, x):
        "Return the transformed index of the give value in the column"

        d = self.get_transform_map(col)
        return d[x]

    def single_transform(self, col, x):
        """
        transform the given column
        col    : column name in string
        x      : pdf series or list to be transformed

        """
        if col in self.fit_dict.keys():
            rule = self.fit_dict[col]
            return (rule.fit_transform(x))
        else:
            return None

    def get_transform_map(self, col):
        """Return the transformed dictionary of the given column"""
        if col in self.fit_dict.keys():
            rule = self.fit_dict[col]
            return (dict(zip(rule.classes_, range(len(rule.classes_)))))
        else:
            return None

    def single_inverse_transform(self, col, x):
        """Inverse transformed column to original"""
        if col in self.fit_dict.keys():
            rule = self.fit_dict[col]
            return (rule.inverse_transform(x))
        else:
            return None

    def get_all_transform_map(self):
        """Return the transform map of all columns"""
        result = defaultdict(np.array)
        for col in self.fit_dict.keys():
            rule = self.fit_dict[col]
            result[col] = dict(zip(rule.classes_, range(len(rule.classes_))))
        return (dict(result))

    def reset(self, cols=None):
        "reset the state"
        # label encoder (a default dict is used to assoicate an encoder for that column)
        # TODO: stacked encoder with oneHot
        # TODO: one hot encoding
        self.fit_dict = defaultdict(LabelEncoder)
        self.categorical_cols = cols

    def fit(self, X, y=None):
        "dummy fit"
        return self

    # TODO: inverse_transform

    def transform(self, X, y=None):
        '''
        Call LabelEncoder() for each column in data frame X

        Parameters
        ----------
        X     : pandas data frame
        '''
        # lambda function either transform or a columns or return the same column
        do_transform = lambda x: self.fit_dict[x.name].fit_transform(x) \
                    if x.name in self.categorical_cols else x

        result = X.copy()

        # get categorical variables
        if self.categorical_cols is None:
            self.categorical_cols = ex.get_categorical_column(X)

        # apply exclusions
        self.categorical_cols = list(set(self.categorical_cols) - set(self.exclusion))

        # Encoding and return results
        result = result.apply(do_transform)

        return result

    # def fit_transform(self, X):
    #     "fit and transformataion"
    #     return (self.transform(X))

# class DummyEncoder(BaseEstimator, TransformerMixin):
#     "One Hot Encoder"
#     def __init__(self):
#         self.encoder = OneHotEncoder(handle_unknown=='ignore')
#     def fit(self, X, y=None):
#         self.encoder.fit(X.ravel())
#         return self
#     def transform(self, X):
#         shape = X.shape
#         result = self.encoder.transform(X.ravel())
#         return result.reshape(shape)

class Imputer(TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        self.categoricalCols = [col for col in X.dtypes.index if X.dtypes.get(col) == 'object' ]

        do_transform = lambda x: x.fillna('NA') \
                if x.name in self.categoricalCols else x.fillna(0)

        result = X.copy()
        result = result.apply(do_transform)

        return result

class Encoder(TransformerMixin):

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        self.categoricalCols = [col for col in X.dtypes.index if X.dtypes.get(col) == 'object' ]
        self.fit_dict = defaultdict(LabelEncoder)

        # lambda function either transform or a columns or return the same column
        do_transform = lambda x: self.fit_dict[x.name].fit_transform(x) \
                    if x.name in self.categoricalCols else x

        result = X.copy()

        # Encoding and return results
        result = result.apply(do_transform)

        return result

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    "Label encoder"
    def __init__(self):
        self.encoder = LabelEncoder()
    def fit(self, X, y=None):
        self.encoder.fit(X.ravel())
        return self
    def transform(self, X):
        shape = X.shape
        result = self.encoder.transform(X.ravel())
        return result.reshape(shape)

# class Factorizer(BaseEstimator, TransformerMixin):
#     "Label encoder"
#     def __init__(self):
#         pass
#     def fit(self, X, y=None):
#         return self
#     def transform(self, X):
#         codedX, categories = X.factorize()
#         return codedX

class DataFrameSelector(BaseEstimator, TransformerMixin):
    "select columns"
    def __init__(self, column_type="numerical"):
        self.column_type = column_type
        self.attribute_names = None
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        if self.column_type == "numerical":
            self.attribute_names = ex.get_numerical_column(X)
        else:
            self.attribute_names = ex.get_categorical_column(X)
        return X[self.attribute_names].values



# ----------------------------------------------------------------------------
# DEBUG
# ----------------------------------------------------------------------------
# FIXME: transform with or without arguments
class FunctionTransformer(TransformerMixin):

    def __init__(self, fun, args = None):
        self.transformer = fun
        self.args = args

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        result = self.transformer(X, self.args)
        return result

# FIXME: one hot encoder??
# class DataFrameOneHotEncoder(TransformerMixin):

#     def __init__(self, cols=None, exclusion=[]):
#         self.reset(cols)
#         self.exclusion = exclusion

#     def reset(self, cols=None):
#         self.fit_dict = defaultdict(OneHotEncoder)
#         # self.encoder = defaultdict(OneHotEncoder)
#         self.categorical_cols = cols
#         self.shape_dict = None 
#         self.index = None
#         self.column_name = None

#     def record_metadata(self, X):
#         self.shape_dict = X.shape
#         self.index = X.index
#         self.column_name = X.columns

#     def fit(self, X, y=None):
#         return self

#     def inverse_transform(self, X):
#         pass

#     def transform(self, X):
#         """
#         X   : a data frame
#         """

#         # x is a series
#         do_transform = lambda x: self.fit_dict[x.name].fit_transform(x.values.reshape(-1, 1)) \
#             if x.name in self.categorical_cols else x

#         self.record_metadata(X)
#         result = X.copy()

#         # get categorical variables
#         if self.categorical_cols is None:
#             self.categorical_cols = ex.get_categorical_column(X)

#         # apply exclusions
#         self.categorical_cols = list(set(self.categorical_cols) - set(self.exclusion))

#         # Encoding and return results
#         result = result.apply(do_transform)

#         return result


# class DummyEncoder(TransformerMixin):
#     '''
#         Convert the columns to dummy variables
#     '''
#     def __init__(self, excluded_cols=None, cols=None):
#         '''
#             excluded_cols   : columns to be excluded (higher priority)
#             cols            : columns to transform
#         '''
#         self.reset(excluded_cols, cols)

#     def fit(self, X, y=None):
#         """needed for pipeline"""
#         # must return self
#         return self

#     def set_col(self, cols):
#         self.cols = cols

#     def reset(self, excluded_cols, cols=None):
#         "reset the state"
#         # dictionary of label encoder
#         self.categorical_cols = cols
#         if isinstance(excluded_cols, list):
#             self.excluded_cols = excluded_cols
#         else:
#             self.excluded_cols = [excluded_cols]

#     def transform(self, X):
#         """
#             X : pd data frame

#         """
#         # get cols for transformation
#         if self.categorical_cols is None:
#             if self.excluded_cols is not None:
#                 cols = list(set(X.columns) - set(self.excluded_cols))
#             else:
#                 cols = X.columns
#         else:
#             if self.excluded_cols is not None:
#                 cols = list(set(self.categorical_cols) - set(self.excluded_cols))
#             else:
#                 cols = self.categorical_cols

#         dummy_df = pd.get_dummies(X[cols])
#         # columns not for transformation
#         other_cols = list(set(X.columns) - set(cols))
#         if other_cols is not None:
#             df = pd.concat([dummy_df, X[other_cols]], axis=1)
#         else:
#             df = dummy_df

#         return df

class TransformDebugger(TransformerMixin):
    """Embed a transformer and print debug information

        Parameters
        ----------
        transformer: any transformer
        debug_func : print function for debug
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
        # FIXME: just return self?? (call fit function of the embedded transformer)
#         self.transformer.fit(X, y)
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

#============================================================
# Staging Codes
def one_hot_encoded(class_numbers, num_classes=None):
    """
    Generate the One-Hot encoded class-labels from an array of integers.

    For example, if class_number=2 and num_classes=4 then
    the one-hot encoded label is the float array: [0. 0. 1. 0.]

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