from abc import ABC, abstractmethod 
from sklearn.pipeline import Pipeline, make_pipeline
from category_encoders.woe import WOEEncoder
from sklearn.base import BaseEstimator
import numpy as np
import pandas as pd

# import varprofile as vp
import transform as trf

#---------------------------------------------------------------------
# pipelines
#---------------------------------------------------------------------
# TORM: for numerical variables
# num_pipeline = Pipeline([
#     ('selector', trf.TypeSelector("numerical")),
#     ('std_scaler', trf.Scaler()),
# ])

# TORM: for categorical and binary variables
# categorical_pipeline = Pipeline([
#     ('selector', trf.TypeSelector("factor")),
#     ('dummy_encoder', trf.DummyEncoder()),
# ])

# for numerical type of variables
numeric_pipeline = Pipeline([
    # ('selector', trf.DTypeSelector("object", True)),
    ('selector', trf.DTypeSelector("numerical", True)),
    ('std_scaler', trf.Scaler()),
])

# for object (string) type of variables
object_pipeline = Pipeline([
    ('selector', trf.DTypeSelector("object")),
    ('dummy_encoder', trf.DummyEncoder()),
])

#---------------------------------------------------------------------
# funtional to create predefined pipelines
#---------------------------------------------------------------------

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

# use hybrid instead of sequential
def make_dummy_pipeline(target_col, exclusions=[]):
    """feature variable transformation
    
        Parameters
        ----------
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

def make_target_guided_pipeline(name, transformer, y=None):
    return Pipeline([
        (name, trf.SupervisedTransformer(transformer,y))
    ])


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
# Pipeline class
# ------------------------------------------------------------
class Pipe(ABC):
    """Base class for transformation pipeline

        Parameters
        -------
        target_col : column name of the target
        exclusions : list of columns to be excluded
        mapping : if not given, target is assumed to be encoded as 1
    
    """

    def __init__(self, target_col, exclusions=[], mapping=None):
        self.features = None
        self.X = None
        self.y = None
        self.px = None      # pipeline for explanatory
        self.py = None      # pipeline for target column
        self.mapping = mapping
        self.exclusions = exclusions
        self.target_col= target_col

    def feature_names(self):
        return self.features

    def pipelines(self):
        return (self.px, self.py)

    # for training only
    def fit_transform(self, df):
        "encoder features and target and set feature cols"
        # note that px produces a data frame while X is a ndarray
        y = self.py.fit_transform(df)
        sdf = self.px.fit_transform(df, y)
        X = sdf.to_numpy()
        self.features = sdf.columns
        # assign target mapping
        self.assign_mapping()
        return (X, y)

    # for testing / scoring only
    def transform(self, df):
        "encoder features and target with fitted pipeline"
        if self.px is None or self.py is None:
            raise Exception('Pipeline is not fitted yet')
        else:
            X = self.px.transform(df).to_numpy()
            y = self.py.transform(df)
            return (X, y)

    def __get_target_mappings(self, idx=1):
        """Get target transform map

            Parameters
            ----------
            py is generated by make_target_pipeline
        """

        # label encoder
        encoder = self.py[idx].encoder
        return dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))

    def target_mapping(self):
        return self.mapping

    def assign_mapping(self, idx=1):
        """
            idx : label encoder for the target pipeline
        """
        # assign mapping
        if self.mapping is None:
            self.mapping = self.__get_target_mappings(idx)

    @abstractmethod
    def make_pipeline(self, **kwargs):
        """
            Parameters
            ----------
            df : data frame
            **kwargs: dict (Additional parameters)

            Assignment
            --------
            py : pipeline for target variable
            px : pipeline for feature variable
            mapping : target variable encode map
            features : list of feature names
        """
        pass

class StandardPipeline(Pipe):
    """
        Parameters
        ----------
        ntransformer : transformer for numerical variables
        ctransformer : transformer for categorical variables
    """

    def make_pipeline(self, **kwargs):  
        self.py = make_target_pipeline(self.target_col, self.mapping)

        # extract and transform feature
        ntransformer = kwargs["ntransformer"]
        ctransformer = kwargs["ctransformer"]
        self.px = make_feature_pipeline(ntransformer, ctransformer, self.target_col, self.exclusions)

# TORM
class DummyPipeline(Pipe):
    """Create Dummy variables for all categorical Variables
    """

    def make_pipeline(self, **kwargs):  
        self.py = make_target_pipeline(self.target_col, self.mapping)
        self.px = make_dummy_pipeline(self.target_col, self.exclusions)


class WoEPipeline(Pipe):
    """WOE transform all categorical Variables
        for binary target only

    """

    def make_pipeline(self, **kwargs):  
        self.py = make_target_pipeline(self.target_col, self.mapping)

        woe_transformer = make_target_guided_pipeline('WOE', WOEEncoder())
        pipelines = [('WOE', woe_transformer)]
        self.px = make_classification_pipeline(pipelines, self.target_col, self.exclusions)

class GeneralizedWoEPipeline(Pipe):
    """WOE transform all categorical Variables
    """

    def make_pipeline(self, **kwargs):  
        self.py = make_target_pipeline(self.target_col, self.mapping)

        # FIXME: var profiler requires target column to make it work - see woe_encode()
        woe_transformer = make_target_guided_pipeline('WOE', trf.WoeTransformer(self.target_col))
        pipelines = [('WOE', woe_transformer)]
        self.px = make_classification_pipeline(pipelines, self.target_col, self.exclusions)
