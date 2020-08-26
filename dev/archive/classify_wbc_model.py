import pandas as pd
import os, sys
# sys.path
mlibpath = r"C:\Users\m038402\Documents\myWork\pythoncodes\mlib"
sys.path.append(mlibpath)

from categorize import Categorizer
import utility as ut
import explore as ex 
import model as ml

data_file = "./data/data.csv"
TARGET_F ="TARGET_F"

df = ut.load_csv(data_file)
df.head()

# normal sklearn split
# train_df, test_df = ut.train_test_split_with_target(df, test_size=0.2)
# df must be a data frame
train_df, test_df, train_target, test_target = ut.train_test_split_with_target(df, test_size=0.2, target=TARGET_F)

train_df.shape
test_df.shape
train_target.shape
test_target.shape

train_df.columns

trainX = ml.full_pipeline.fit_transform(train_df)
testX = ml.full_pipeline.fit_transform(test_df)

