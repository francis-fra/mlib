
import pandas as pd
import os, sys
# sys.path
# mlibpath = r"C:\Users\m038402\Documents\myWork\pythoncodes\mlib"
mlibpath = '/home/fra/Project/pyProj/mlib'
sys.path.append(mlibpath)

# from categorize import Categorizer
import utility as ut
import explore as ex 
import model as ml
import pipeline as pl

# data_file = "./data/data.csv"

# df = pd.read_csv(data_file)
# df.head()
# ------------------------------------------------------------
# load data
# ------------------------------------------------------------
data_folder = "/home/fra/DataMart/datacentre/westpac/"
train_filename = "W_EXTREF_R_V_MDS_train.csv"
data_file = data_folder + train_filename
df_train = pd.read_csv(data_file)

test_filename = "W_EXTREF_R_V_MDS_test.csv"
data_file = data_folder + test_filename
df_test = pd.read_csv(data_file)

# ------------------------------------------------------------
exclusions=['DATA_DT', 'PROCESS_DTTM', 'GCIS_KEY', 'CUSTOMER_ID', 'PERIOD_ID']
target_col ="TARGET_F"

from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from importlib import reload

stdpl = pl.StandardPipeline(target_col, exclusions)
stdpl.make_pipeline(ntransformer=StandardScaler, ctransformer=OrdinalEncoder)

(X_train, y_train) = stdpl.fit_transform(df_train)
(X_test, y_test) = stdpl.transform(df_test)
# stdpl.target_col
# stdpl.mapping
# stdpl.features
# len(stdpl.features)
# X_train.shape
# y_train.shape


# ------------------------------------------------------------
# WoE encoder
# ------------------------------------------------------------
reload(pl)
woepl = pl.WoEPipeline(target_col, exclusions)
woepl.make_pipeline(target=y_train)
(X_train, y_train) = woepl.fit_transform(df_train)
(X_test, y_test) = woepl.transform(df_test)

# ------------------------------------------------------------
# Model
# ------------------------------------------------------------
import xgboost as xgb
xgb_cls = xgb.XGBClassifier()
model = ml.BinaryClassifier(xgb_cls)
model.fit(X_train, y_train)

xgb_eval = ml.BinaryClassificationEvaluator(model)
xgb_eval.performance_summary(X_test, y_test)
# gini: 0.48 using woe encoder
# gini: 0.39