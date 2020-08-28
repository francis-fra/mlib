import pandas as pd
import pyodbc
import pickle

mylib = 'C:\\Users\\m038402\\Documents\\myWork\\Codes\\palantir\\pythonlib'
sys.path.append(mylib)
import explore as ex
import transform as tf
import utility as ut
import evaluate as ev

#------------------------------------------------------------------------- 
# get data
#------------------------------------------------------------------------- 
cnxn = pyodbc.connect('DSN=tdp1;UID=m038402;PWD=Nov17#402')
cnxn

query = 'select * from c4ustpmk.binary_si_segment_extra_MDS'
data = pd.read_sql(query, cnxn)

#------------------------------------------------------------------------- 
# save data
#------------------------------------------------------------------------- 
outFile = 'C:\\Users\\m038402\\Documents\\HomeDir\\workspace\\myModels\\saving\\savings201609.csv'
data.to_csv(outFile, index=False)

#------------------------------------------------------------------------- 
# load data
#------------------------------------------------------------------------- 
df = pd.read_csv(outFile, index_col=None)
df.columns
df.shape

categorical_cols = ex.get_categorical_column(df)
categorical_cols
len(categorical_cols)
# 48

#------------------------------------------------------------------------- 
# imputation
#-------------------------------------------------------------------------
ex.count_missing(df)
 
imputer = tf.DataFrameImputer()
imputer.fit(df)
newdf = imputer.transform(df)
ex.count_missing(newdf)

#------------------------------------------------------------------------- 
# data transformation
#------------------------------------------------------------------------- 
label_encoder = tf.DataFrameLabelEncoder()
labeled_df = label_encoder.transform(newdf)
labeled_df.head()

target = 'TARGET_F'

# split data
# exclusions = ['CUSTOMER_ID', 'REF_MONTH']
exclusions = ['NEW_QUADRANT_NO', 'REF_MONTH']
# random train/test split
X_train, X_test, y_train, y_test, features = ut.df_train_test_split(labeled_df, target, exclusions = exclusions, test_size=0.3)
X_train.shape, X_test.shape
y_train.shape, y_test.shape
features
len(features)
# 146

# find out the target transform map
label_encoder.get_all_transform_map()[target]
label_encoder.get_all_transform_map()

#------------------------------------------------------------------------- 
# modelling - binary classification
#-------------------------------------------------------------------------
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
gbm = GradientBoostingClassifier()
rf = RandomForestClassifier()

from imp import reload
import evaluate
reload(evaluate)
import evaluate as ev

evaluator = ev.ClassificationEvaluator(rf, X_train, y_train, X_test, y_test, 'CUSTOMER_ID', 
                                       features, cv=10, target_f=0, nsizes=10, n_jobs=-1)

evaluator.plot_roc_curve()
evaluator.get_gini()
evaluator.plot_importance_features(20)
evaluator.show_confusion_matrix()

# save model

saveDir = 'C:\\Users\\m038402\\Documents\\HomeDir\\workspace\\myModels\\wbcModels\\data\\'
with open(saveDir + 'model.pickle', 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(evaluator, f, pickle.HIGHEST_PROTOCOL)
    
    
#------------------------------------------------------------------------- 
# score
#-------------------------------------------------------------------------
evaluator.train().score()
# scoring table
score_table = evaluator.get_score_table()
score_table.head()

#------------------------------------------------------------------------- 
# TODO: find which quadrant is problematic
#-------------------------------------------------------------------------
y_pred = evaluator.estimator.predict(X_test)
y_pred[:5]

# compare with true value and customer quadrant
scoredf = score_table
scoredf['target_f'] = y_test 
scoredf['pred'] = y_pred
# add quadrant
otherdf = df[['CUSTOMER_ID', 'NEW_QUADRANT_NO']]
otherdf.head()

joindf = pd.merge(scoredf, otherdf, 'inner', on='CUSTOMER_ID')
joindf.head()

# check if the prediction is right
joindf['matched'] = joindf['target_f']  == joindf['pred']
joindf.head()

saveDir = 'C:\\Users\\m038402\\Documents\\HomeDir\\workspace\\myModels\\wbcModels\\data\\'
with open(saveDir + 'outcome.pickle', 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(joindf, f, pickle.HIGHEST_PROTOCOL)

#------------------------------------------------------------------------- 
# statistics about miss-classifications
#------------------------------------------------------------------------- 
joindf.columns
ex.count_levels(joindf, cols='matched')
ex.count_levels(joindf, cols='NEW_QUADRANT_NO')

ex.count_levels(joindf[joindf['matched']==True], cols='NEW_QUADRANT_NO')
ex.count_levels(joindf[joindf['matched']==False], cols='NEW_QUADRANT_NO')

ex.count_levels(joindf, cols='NEW_QUADRANT_NO')
#------------------------------------------------------------------------- 
# TODO: grid search
#-------------------------------------------------------------------------


#------------------------------------------------------------------------- 
# TODO: ensemble
#-------------------------------------------------------------------------



#-------------------------------------------------------------------------
# DEBUG
#-------------------------------------------------------------------------
y_pred = evaluator.estimator.predict_proba(X_test)
y_pred[:5]

X_test.shape
len(features)
'CUSTOMER_ID' in features
import numpy as np
# features == 'CUSTOMER_ID'
ix = np.isin(features, 'CUSTOMER_ID')
ix
np.where(ix).values()
colidx = np.where(ix)[0][0]
features[127]

X_test[colidx]

ix = np.isin(evaluator.features, evaluator.id)
colidx = np.where(ix)[0][0]
y_pred = evaluator.estimator.predict_proba(X_test)
idx = evaluator.target_f
score = y_pred[:,idx]
score
pd.DataFrame({'id': X_test[:,127], 'score': score})

X_test.shape
len(X_test[127])
len(score)