from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder

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