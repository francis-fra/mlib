import pandas as pd
import pyodbc

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

# query = 'select * from c4ustpmk.binary_si_segment_extra_MDS'
# query = 'select * from c4ustpmk.si_segment_all_MDS_REDUCED'
query = 'select top 1000 * from c4ustpmk.si_segment_all_MDS_REDUCED'
data = pd.read_sql(query, cnxn)
data.columns

df = data
df.shape
# (1000, 169)
#------------------------------------------------------------------------- 
# save data
#------------------------------------------------------------------------- 
outFile = 'C:\\Users\\m038402\\Documents\\HomeDir\\workspace\\myModels\\saving\\savings201609.csv'
data.to_csv(outFile, index=False)


outFile = 'C:\\Users\\m038402\\Documents\\HomeDir\\workspace\\myModels\\saving\\savings_reduced.csv'
data.to_csv(outFile, index=False)

#------------------------------------------------------------------------- 
# load data
#------------------------------------------------------------------------- 
# outFile = 'C:\\Users\\m038402\\Documents\\HomeDir\\workspace\\myModels\\saving\\data\\savings201609.csv'
outFile = 'C:\\Users\\m038402\\Documents\\HomeDir\\workspace\\myModels\\saving\\data\\savings_reduced.csv'
df = pd.read_csv(outFile, index_col=None)
df.columns
df.shape

#------------------------------------------------------------------------- 
categorical_cols = ex.get_categorical_column(df)
categorical_cols
len(categorical_cols)
# 48

# load evaluator
# with open('data.pickle', 'rb') as f:
#     data = pickle.load(f)
    
    
#------------------------------------------------------------------------- 
# imputation
#-------------------------------------------------------------------------
ex.count_missing(df)
 
imputer = tf.DataFrameImputer()
imputer.fit(df)
newdf = imputer.transform(df)
ex.count_missing(newdf)

#------------------------------------------------------------------------- 
# convert binary to multiclass
#-------------------------------------------------------------------------
# 6 classes
def defineClass(df, col='NEW_QUADRANT_NO'):
    if df[col] in [2, 4]:
        return 1
    if df[col] in [1, 3]:
        return 2
    if df[col] in [5]:
        return 3
    if df[col] in [6, 7, 8]:
        return 4
    elif df[col] in [9, 10, 11]:
        return 5
    elif df[col] in [13]:
        return 6
    elif df[col] in [14]:
        return 7
    elif df[col] in [17]:
        return 8
    else:
        return 9
    
# def defineClass(df, col='NEW_QUADRANT_NO'):
#     if df[col] in [1, 2, 3]:
#         return 0
#     if df[col] in [4, 5]:
#         return 1
#     if df[col] in [6, 7, 9, 10]:
#         return 2
#     if df[col] in [8, 11]:
#         return 3
#     elif df[col] in [12, 13, 14]:
#         return 4
#     else:
#         return 5
# 
# def defineClass(df, col='NEW_QUADRANT_NO'):
#     if df[col] in [1, 2, 3, 4, 5]:
#         return 0
#     elif df[col] in [6, 7, 8, 9, 10, 11]:
#         return 1
#     else:
#         return 2
    
# all classes    
# newdf['TARGET_F'] = newdf['NEW_QUADRANT_NO']
# force the label encoder to transform
# newdf['TARGET_F'] = newdf['NEW_QUADRANT_NO'].apply(str)

# grouped class
newdf['TARGET_F'] = newdf.apply(defineClass, axis=1)
# check    
newdf[['TARGET_F', 'NEW_QUADRANT_NO']][:20]

ex.count_levels(newdf, cols='TARGET_F')
# {'TARGET_F': [(1, 64674),
#   (2, 82019),
#   (3, 3194),
#   (4, 14093),
#   (5, 151079),
#   (6, 31763),
#   (7, 31784),
#   (8, 38058),
#   (9, 83336)]}

newdf.head()

#------------------------------------------------------------------------- 
# data transformation
#------------------------------------------------------------------------- 
label_encoder = tf.DataFrameLabelEncoder()
labeled_df = label_encoder.transform(newdf)
# labeled_df.head()
target = 'TARGET_F'

# split data
# exclusions = ['CUSTOMER_ID', 'REF_MONTH']
exclusions = ['NEW_QUADRANT_NO', 'REF_MONTH']
X_train, X_test, y_train, y_test, features = ut.df_train_test_split(labeled_df, target, exclusions = exclusions, test_size=0.3)
X_train.shape, X_test.shape
y_train.shape, y_test.shape

len(features)
# 166
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
rf = RandomForestClassifier(n_estimators=100)

evaluator = ev.MultiClassEvaluator(gbm, X_train, y_train, X_test, y_test, 'CUSTOMER_ID', 
                                   features, cv=2, target_f=1, target_column='TARGET_F', 
                                   label_encoder = label_encoder)

# evaluator.train()
evaluator.score()
evaluator.plot_roc_curve()
evaluator.plot_importance_features(40)
evaluator.show_confusion_matrix()

len(y_test)
confusionMatrix = evaluator.get_confusion_matrix()
# evaluator.is_fitted()
# evaluator.train()

#-------------------------------------------------------------------------
# get a list of var importance
importancedf = evaluator.get_importance_features()
importancedf.head(40)
outFile = 'M:\\MKTAN\\_Analytics and Modelling Dept\\05 Production Models\\01. Savings\\WBC_Savings_Segmentation\\06 Output\\var.csv'
importancedf.to_csv(outFile)

#-------------------------------------------------------------------------
# find miss-classified quadrant
#-------------------------------------------------------------------------
def examine_class(df, evaluator, X, y):
    "produce a new df to examine the mis-classifications"
    
    # get soft probabilities
    pred = evaluator.get_probability()
    scoredf = pd.DataFrame(pred)
    colname = ['target_' + str(col) for col in scoredf.columns.values.tolist()]
    scoredf.columns = colname
    # true target label
    scoredf['target_f'] = y
    # hard decision
    y_pred = evaluator.estimator.predict(X)
    scoredf['pred'] = y_pred
    
    # probability score with customer id
    score_table = evaluator.score().get_score_table()
    tmp = pd.concat([score_table, scoredf], axis=1)

    # quadrant
    otherdf = df[['CUSTOMER_ID', 'NEW_QUADRANT_NO']]
    joindf = pd.merge(tmp, otherdf, 'inner', on='CUSTOMER_ID')
    
    # check if the hard decisino is right
    joindf['matched'] = joindf['target_f']  == joindf['pred']
    
    return (joindf)
    
resultdf = examine_class(df, evaluator, X_test, y_test)
resultdf.head()
resultdf.shape

# confusionMatrix.shape
# confusionMatrix[0]
# score_table = evaluator.score().get_score_table()
# score_table[:5]
#------------------------------------------------------------------------- 
# statistics about miss-classifications
#------------------------------------------------------------------------- 
ex.count_levels(resultdf, cols='matched')
# quadrant distribution
ex.count_levels(resultdf, cols='NEW_QUADRANT_NO')

ex.count_levels(resultdf[resultdf['matched']==True], cols='NEW_QUADRANT_NO')
ex.count_levels(resultdf[resultdf['matched']==False], cols='NEW_QUADRANT_NO')

ex.count_levels(resultdf, cols='NEW_QUADRANT_NO')

ex.count_levels(resultdf, cols='pred')
# quadrant mis-classified
missclassified = resultdf[resultdf['matched']==False]


# calcuate capture rate:
tmp = ex.count_levels(resultdf[resultdf['pred']==8], cols='target_f')['target_f']
# number of prediction for this label
sum([count for label, count in tmp])
# actual count of target
ex.count_levels(resultdf, cols='target_f')



ex.count_levels(missclassified[missclassified['NEW_QUADRANT_NO']==2], cols='pred')
# {'pred': [(2, 3), (13, 21)]}
ex.count_levels(missclassified[missclassified['NEW_QUADRANT_NO']==3], cols='pred')



#-------------------------------------------------------------------------
# save model
#------------------------------------------------------------------------- 
from imp import reload
reload(ev)
import pickle

saveDir = 'C:\\Users\\m038402\\Documents\\HomeDir\\workspace\\myModels\\wbcModels\\data\\'
with open(saveDir + 'outcome.pickle', 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(resultdf, f, pickle.HIGHEST_PROTOCOL)
 
    
# save model
saveDir = 'C:\\Users\\m038402\\Documents\\HomeDir\\workspace\\myModels\\wbcModels\\data\\'
with open(saveDir + 'gbm_model.pickle', 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(evaluator, f, pickle.HIGHEST_PROTOCOL)
        
#------------------------------------------------------------------------- 
# Scoring test
#-------------------------------------------------------------------------
# load saved model
savedDir = 'C:\\Users\\m038402\\Documents\\HomeDir\\workspace\\myModels\\wbcModels\\data\\'
segmentationModel = pickle.load( open(savedDir + "gbm_model.pickle", "rb" ) )

segmentationModel.is_fitted()


# Make sure the feature columns are aligned!
pd.DataFrame({'model': segmentationModel.features, 'new': features})

segmentationModel.score(X)
score_table = segmentationModel.get_score_table()

# check
score_table.loc[0:10,]
score_table
sum(score_table['CUSTOMER_ID'] == kdf['CUSTOMER_ID'])


#------------------------------------------------------------------------- 
# TODO: bayesian net demo data sets
#-------------------------------------------------------------------------


#------------------------------------------------------------------------- 
# TODO: additional var
#-------------------------------------------------------------------------

#------------------------------------------------------------------------- 
# KNN clustering
#-------------------------------------------------------------------------
from sklearn.neighbors import NearestNeighbors
nbrs = NearestNeighbors(n_neighbors=6).fit(X_train)
distances, indices = nbrs.kneighbors(X_train)
indices[:25]
X_train.shape
indices.shape

from sklearn.cluster import KMeans
km = KMeans(n_clusters=6)
km.fit(X_train)
km.labels_[:5]
len(km.labels_)

# cluster
clusterdf = pd.DataFrame(km.labels_, columns=['cluster'])
clusterdf.head()

# append customer id
clusterdf['CUSTOMER_ID'] = df.CUSTOMER_ID
clusterdf['NEW_QUADRANT_NO'] = df.NEW_QUADRANT_NO

# count quadrant (distribution) for each cluster
ex.count_levels(clusterdf[clusterdf['cluster']==1], cols='NEW_QUADRANT_NO')
ex.count_levels(clusterdf[clusterdf['cluster']==2], cols='NEW_QUADRANT_NO')
ex.count_levels(clusterdf[clusterdf['cluster']==3], cols='NEW_QUADRANT_NO')

