#-------------------------------------------------------------------------
# my library
#-------------------------------------------------------------------------
# mylib = 'C:\\Users\\m038402\\Documents\\HomeDir\\workspace\\myModels\\pylib'
mylib = 'C:\\Users\\m038402\\Documents\\myWork\\Codes\\palantir\\pythonlib'
sys.path.append(mylib)
import explore as ex
import transform as tf
import utility as ut
import evaluate as ev

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from imp import reload
import transform
reload(transform)
import transform as tf

#------------------------------------------------------------------
# Load a multi-label dataset
#------------------------------------------------------------------
yeast = fetch_mldata('yeast')
X = yeast['data']
Y = yeast['target'].transpose().toarray()
X.shape, Y.shape
Y[:5]
X[0]

# split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2)
X_test.shape, Y_test.shape

#-------------------------------------------------------------------------
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import label_binarize, LabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

# algorithms
gbm = GradientBoostingClassifier()
gbm_ovr = OneVsRestClassifier(gbm)
dt = DecisionTreeClassifier()
dt_ovr = OneVsRestClassifier(dt)

# multi-label classification
gbm_ovr.fit(X_train, Y_train)
Y_pred_ovr = gbm_ovr.predict(X_test)
ovr_jaccard_score = jaccard_similarity_score(Y_test, Y_pred_ovr)
ovr_jaccard_score
# 0.5

Y_pred_ovr[:5]
Y_test[:5]

#------------------------------------------------------------------
# using chain classifier ensemble
#------------------------------------------------------------------
from sklearn.multioutput import ClassifierChain

chains = [ClassifierChain(GradientBoostingClassifier(), order='random', random_state=i)
          for i in range(10)]

for chain in chains:
    chain.fit(X_train, Y_train)
    
   
Y_pred_chains = np.array([chain.predict(X_test) for chain in chains])

chain_jaccard_scores = [jaccard_similarity_score(Y_test, Y_pred_chain >= .5)
                        for Y_pred_chain in Y_pred_chains]


Y_pred_ensemble = Y_pred_chains.mean(axis=0)
# check
Y_pred_ensemble[:3]

ensemble_jaccard_score = jaccard_similarity_score(Y_test, Y_pred_ensemble >= .5)

model_scores = [ovr_jaccard_score] + chain_jaccard_scores
model_scores.append(ensemble_jaccard_score)

# ensemble scores is the highest
model_scores


#------------------------------------------------------------------
# abalone data set
#------------------------------------------------------------------
import pandas as pd
import numpy as np

 # name of the column containing the target
target = 'sex'
dataDir = 'C:\\Users\\m038402\\Documents\\myWork\\Codes\\sparkProj\\data\\'
data_file = 'abalone.csv'
df = pd.read_csv(dataDir + data_file)

target = 'sex'
ex.count_levels(df, 'sex')

#------------------------------------------------------------------------- 
# split train / test
label_encoder = tf.DataFrameLabelEncoder()
newdf = label_encoder.transform(df)
newdf.head()

exclusions = []
X_train, X_test, y_train, y_test, features = ut.df_train_test_split(newdf, target, exclusions=exclusions, test_size=0.3)

X_train[:5]
y_train[:5]

#------------------------------------------------------------------------- 
# TODO: convert multiclass to multilabel
#------------------------------------------------------------------------- 




