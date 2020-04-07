from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
import numpy as np

import transform as tf

num_pipeline = Pipeline([
        ('selector', tf.DataFrameSelector("numerical")),
        ('imputer', SimpleImputer(strategy="constant", fill_value=0)),
        ('std_scaler', StandardScaler()),
    ])

categorical_pipeline = Pipeline([
        ('selector', tf.DataFrameSelector("categorical")),
        # ('imputer', SimpleImputer(strategy="constant", fill_value=0)),
        # ('dummy_creator', OneHotEncoder(handle_unknown='ignore')),
        ('categorical_encoder', tf.DataFrameCategoricalEncoder()),
        # ('categorical_encoder', tf.CategoricalEncoder()),
    ])

full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", categorical_pipeline),
    ])

# imputation_pipeline = Pipeline([
#         ('imputer', tf.ConstantImputer()),
#         ('encoder', tf.DataFrameCategoricalEncoder(),
#     ])

class RandomClassifier(BaseEstimator):
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

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names]

class MostFrequentImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.most_frequent_ = pd.Series([X[c].value_counts().index[0] for c in X],
                                        index=X.columns)
        return self
    def transform(self, X, y=None):
        return X.fillna(self.most_frequent_)