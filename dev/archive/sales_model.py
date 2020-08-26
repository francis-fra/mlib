import pandas as pd
import os, sys
# sys.path
mlibpath = r"C:\Users\m038402\Documents\myWork\pythoncodes\mlib"
sys.path.append(mlibpath)

from categorize import Categorizer
import utility as ut
import explore as ex 
import model as ml
import numpy as np
import transform as tf
from imp import reload
reload(ut)

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

data_file = "./data/sales_train.csv"

df = ut.load_csv(data_file)
df.shape
df.columns
df.describe()

ex.get_type_dict(df)

# -----------------------------------------------------------------------------
# clean data: remove nan and remove zero target
# find missing values
ex.count_missing(df)

# remove zero contract price
targetcol = 'CL_Contract_Price'
reload(tf)
reload(ml)

to_drop = ['CL_Transfer_Date', 'Cl_History_From', 'Cl_Property_ID']

# ------------------------------------------------------------------------
# impute and encode
# ------------------------------------------------------------------------
ex.count_missing(df)
remover = tf.ValueRemover(targetcol, [0, np.NaN])
cdf = remover.fit_transform(df)
imputer = tf.ConstantImputer()
cdf = imputer.fit_transform(df)
ex.count_missing(cdf)

encoder = tf.DataFrameCategoricalEncoder(exclusion=['CL_Transfer_Date'])
cdf = encoder.fit_transform(cdf)
cdf.head()

encoder.fit_dict

# ------------------------------------------------------------------------
# TODO: scale numerical values
# main pipeline
fmt = "%d/%m/%Y"
pipeline = Pipeline([
    ('dateExtractor', tf.DateExtractor(['CL_Transfer_Date'], fmt)),
    ('columnRemover', tf.ColumnRemover(to_drop))
])

cdf = pipeline.fit_transform(cdf)
cdf.head()
cdf.shape

#----------------------------------------------------------------
(X, y, features) = ut.extract_feature_target(cdf, targetcol)
X.head()

# -----------------------------------------------------------------------------
# modelling

# linear model
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()

# fit the training data
lin_reg.fit(X, y)
lin_reg.coef_, lin_reg.intercept_, 

# -----------------------------------------------------------------------------
# evaluate rmse
y_pred = lin_reg.predict(X)

from sklearn.metrics import mean_squared_error
import math
math.sqrt(mean_squared_error(y, y_pred))
# 510202

# -----------------------------------------------------------------------------
# test data
data_file = "./data/sales_test.csv"
df_test = ut.load_csv(data_file)

cdf_test = remover.transform(df_test)
cdf_test = imputer.transform(cdf_test)

# encoder = tf.DataFrameCategoricalEncoder(exclusion=['CL_Transfer_Date'])
cdf_test = encoder.transform(cdf_test)
cdf_test = pipeline.fit_transform(cdf_test)
cdf_test.head()

(X_test, y_test, features) = ut.extract_feature_target(cdf_test, targetcol)

y_pred = lin_reg.predict(X_test)
math.sqrt(mean_squared_error(y_test, y_pred))
# 556202

# -----------------------------------------------------------------------------
# TODO: more regressor
# TODO: param search
# TODO: key drivers