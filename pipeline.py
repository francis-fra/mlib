# from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline, make_pipeline
from category_encoders.woe import WOEEncoder
# from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator
import numpy as np
import pandas as pd

# from functools import reduce
import transform as trf

#---------------------------------------------------------------------
# pipelines
#---------------------------------------------------------------------
# TORM: for numerical variables
num_pipeline = Pipeline([
    ('selector', trf.TypeSelector("numerical")),
    ('std_scaler', trf.Scaler()),
])

# TORM: for categorical and binary variables
categorical_pipeline = Pipeline([
    ('selector', trf.TypeSelector("factor")),
    ('dummy_encoder', trf.DummyEncoder()),
])

# for numerical type of variables
numeric_pipeline = Pipeline([
    ('selector', trf.DTypeSelector("object", True)),
    ('std_scaler', trf.Scaler()),
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
    """hybrid feature transformation
    
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

        to get component from pipeline: e.g. dpy.steps[1][1].encoder
    
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

def make_target_guided_pipeline(name, transformer, y):
    return Pipeline([
        (name, trf.SupervisedTransformer(transformer(),y))
    ])

# deprecated
# def make_classification_pipeline(transformer, y, target_col, exclusions=[]):
#     """feature variable transformation
    
#         Parameters
#         ----------
#         transformer : supervised transformer 
#         y : binary target
#         target_col : string
#         exclusions : string
    
#         Returns 
#         ----------
#         df
#     """
#     base_pipeline = make_base_pipeline(target_col, exclusions)
#     pipeline = Pipeline([
#         ('base_pipeline', base_pipeline),
#         ('guided_transformer', trf.SupervisedTransformer(transformer(),y)),
#         ('std_scaler', trf.Scaler()),
#     ])
#     return pipeline

def make_classification_pipeline(pipelines, target_col, exclusions=[]):
    """add custom pipeline in additional to basic
    
        Parameters
        ----------
        pipelines : list of additional pipelines
        target_col : string
        exclusions : string
    
        Returns 
        ----------
        df
    """
    base_pipeline = make_base_pipeline(target_col, exclusions)
    px =[('base_pipeline', base_pipeline)]
    px = px + pipelines 
    px = px + [('std_scaler', trf.Scaler())]
    px = Pipeline(px)
    return px

# ------------------------------------------------------------
# Process
# ------------------------------------------------------------
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


def binary_classfication_preprocess(df, target_col, exclusions=[], mapping=None):
    """Preprocessing Raw data
    
        Parameters
        ----------
        df : raw data frame
        target_col : string
        exclusions : list of names to be excluded
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

    # for target guided pipeline, need the target y
    # TODO: add more here
    woe_transformer = make_target_guided_pipeline('WOE', WOEEncoder, y)
    pipelines = [('WOE', woe_transformer)]
    px = make_classification_pipeline(pipelines, target_col, exclusions)

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

