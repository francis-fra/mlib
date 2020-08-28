import pandas as pd
import sys
mylib = '/home/fra/Project/pyProj/mlib'

sys.path.append(mylib)

data_folder = "/home/fra/DataMart/datacentre/olddata/"
train_filename = 'ccbt_train.csv'

df = pd.read_csv(data_folder + train_filename)
df.head()

import utility as util
import explore as ex
import pipeline as pl
import transform as trf

target_col = 'TARGET_F'

categorical_columns = ex.get_categorical_column(df)
non_categorical_columns = ex.get_non_categorical_column(df)

ex.count_missing(df)

# ------------------------------------------------------------
# label encoder
from sklearn import preprocessing
le = preprocessing.LabelEncoder()

# X = df.to_numpy()
# le.fit(X)
from imp import reload
reload(trf)

trf.SimpleEncoder().fit_transform(df)

cdf = trf.DataFrameImputer().fit_transform(df)
# ex.any_missing(df)
# ex.any_missing(cdf)
trf.SimpleEncoder().fit_transform(cdf[categorical_columns])
le.fit(cdf.iloc[:,202])
# trf.SimpleEncoder().fit_transform(cdf.iloc[:,202])

cdf.iloc[:,200:205].head()
cdf.iloc[:,200:205].columns



# ------------------------------------------------------------
# TODO: test categorical encoder / LabelTransformer
px = pl.make_base_pipeline(target_col, exclusions=['DATA_DT', 'PROCESS_DTTM', 'GCIS_KEY'])
out = px.fit_transform(df)