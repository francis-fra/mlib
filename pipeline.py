# from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
# from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator
import numpy as np
import pandas as pd

# from functools import reduce
import transform as trf

#---------------------------------------------------------------------
# pipelines
#---------------------------------------------------------------------
# for numerical variables
num_pipeline = Pipeline([
    ('selector', trf.TypeSelector("numerical")),
    ('std_scaler', trf.Scaler()),
])

# for categorical and binary variables
categorical_pipeline = Pipeline([
    ('selector', trf.TypeSelector("factor")),
    ('dummy_encoder', trf.DummyEncoder()),
])

# for numerical type of variables
numeric_pipeline = Pipeline([
    ('selector', trf.DTypeSelector("object", True)),
    ('dummy_encoder', trf.Scaler()),
])

# for object (string) type of variables
object_pipeline = Pipeline([
    ('selector', trf.DTypeSelector("object")),
    ('dummy_encoder', trf.DummyEncoder()),
])


def make_base_pipeline(target_col, exclusions=[]):
    """Basic variable transformation
    
        Parameters
        ----------
        target_col : string
        exclusions : string
    
        Returns 
        ----------
        df
    """
    base_pipeline = Pipeline([
        ('upper_case', trf.UpperCaseColumn()),
        ('drop_columns', trf.DropColumns(exclusions + [target_col])),
        ('imputer', trf.DataFrameImputer()),
    ])
    return base_pipeline

def make_feature_pipeline(ntransformer, ctransformer, target_col, exclusions=[]):
    """feature variable transformation
    
        Parameters
        ----------
        ntransformer : transformer for numeric variable
        ctransformer : transformer for sring (object) variable
        target_col : string
        exclusions : string
    
        Returns 
        ----------
        df
    """
    base_pipeline = make_base_pipeline(target_col, exclusions)
    pipeline = Pipeline([
        ('base_pipeline', base_pipeline),
        ('transformer', trf.HybridTransformer(ntransformer, ctransformer))
    ])
    return pipeline

def make_dummy_pipeline(target_col, exclusions=[]):
    """feature variable transformation
    
        Parameters
        ----------
        ntransformer : transformer for numeric variable
        ctransformer : transformer for sring (object) variable
        target_col : string
        exclusions : string
    
        Returns 
        ----------
        df
    """
    base_pipeline = make_base_pipeline(target_col, exclusions)
    pipeline = Pipeline([
        ('base_pipeline', base_pipeline),
        ('seq_transformer', trf.SequentialTransformer([numeric_pipeline, object_pipeline])),
        ('combiner', trf.Combiner()),
    ])
    return pipeline

def make_target_pipeline(target_name, mapping=None):
    """extract and transform target variable
    
        Parameters
        ----------
        target_name : string
        mapping : dictionary of custom mapping (optional)
    
        Returns 
        ----------
        ndarray 
    """
    return Pipeline([
        ('upper_case', trf.UpperCaseColumn()),
        ('encoder', trf.TargetEncoder(target_name, mapping))
    ])

def dummy_feature_preprocess(df, target_col, exclusions=[], mapping=None):
    """Preprocessing Raw data
    
        Parameters
        ----------
        df : raw data frame
        target_col : string
        mapping : custom target variable transformation map
    
        Returns 
        ----------
        df
    """
    # extract target column
    py = make_target_pipeline(target_col, mapping)
    y = py.fit_transform(df)

    # extract and transform feature
    px = make_dummy_pipeline(target_col, exclusions)
    df = px.fit_transform(df)

    X = df.to_numpy()

    return (X, y, df.columns, py, px)

# http://contrib.scikit-learn.org/category_encoders/woe.html
def make_target_guided_pipeline(transformer, y, target_col, exclusions=[]):
    """feature variable transformation
    
        Parameters
        ----------
        transformer : supervised transformer 
        y : binary target
        target_col : string
        exclusions : string
    
        Returns 
        ----------
        df
    """
    base_pipeline = make_base_pipeline(target_col, exclusions)
    pipeline = Pipeline([
        ('base_pipeline', base_pipeline),
        ('guided_transformer', trf.SupervisedTransformer(transformer(),y))
    ])
    return pipeline

def target_guided_feature_preprocess(df, target_col, transformer, exclusions=[], mapping=None):
    """Preprocessing Raw data
    
        Parameters
        ----------
        df : raw data frame
        target_col : string
        ntransformer: transformer for numeric variables (must act on numpy, e.g. StandardScaler)
        ctransformer: transformer for categorical variables (must act on numpy, e.g. OrdinalEncoder)
        mapping : custom target variable transformation map
    
        Returns 
        ----------
        X : explantory variables
        y : target column
        feature : list of feature names
        py : target variable pipeline
        px : feature variable pipeline
    """

    # extract target column
    py = make_target_pipeline(target_col, mapping)
    y = py.fit_transform(df)

    # extract and transform feature
    px = make_target_guided_pipeline(transformer, y, target_col, exclusions)
    # note that px produces a data frame while X is a ndarray
    df = px.fit_transform(df, y)
    X = df.to_numpy()

    return (X, y, df.columns, py, px)

def standard_feature_preprocess(df, target_col, ntransformer, ctransformer, exclusions=[], mapping=None):
    """Preprocessing Raw data
    
        Parameters
        ----------
        df : raw data frame
        target_col : string
        ntransformer: transformer for numeric variables (must act on numpy, e.g. StandardScaler)
        ctransformer: transformer for categorical variables (must act on numpy, e.g. OrdinalEncoder)
        mapping : custom target variable transformation map
    
        Returns 
        ----------
        X : explantory variables
        y : target column
        feature : list of feature names
        py : target variable pipeline
        px : feature variable pipeline
    """

    # extract target column
    py = make_target_pipeline(target_col, mapping)
    y = py.fit_transform(df)

    # extract and transform feature
    px = make_feature_pipeline(ntransformer, ctransformer, target_col, exclusions)
    # note that px produces a data frame while X is a ndarray
    df = px.fit_transform(df)
    X = df.to_numpy()

    return (X, y, df.columns, py, px)

#---------------------------------------------------------------------
# classifiers
#---------------------------------------------------------------------
class RandomClassifier(BaseEstimator):
    """
        Parameters
        ----------
        prop : base rate
    """
    def __init__(self, prop):
        self.prop = prop
    def fit(self, X, y=None):
        pass
    def predict(self, X):
        nums = np.zeros((len(X), 1), dtype=bool)
        N = math.floor(len(X) * self.prop)
        nums[:N] = True
        np.random.shuffle(nums)
        return nums

class NeverClassifier(BaseEstimator):
    def fit(self, X, y=None):
        pass
    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)
