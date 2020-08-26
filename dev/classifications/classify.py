# data_source = "/home/fra/Project/pyProj/mlib/data/"
# data_file = "abalone.csv"
# "extref_osdData.csv"
# "extref_trainData.csv"
# "mtcars_data.csv"

import pandas as pd

data_source = "/home/fra/DataMart/datacentre/olddata/"
train_file = "adult_train.csv"
test_file = "adult_test.csv"

# load data
trainData = pd.read_csv(data_source + train_file)
trainData.head()

# transform

# 