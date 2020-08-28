#-------------------------------------------------------------------------
# my library
#-------------------------------------------------------------------------
mylib = 'C:\\Users\\m038402\\Documents\\HomeDir\\workspace\\myModels\\pylib'
sys.path.append(mylib)
from mlearn import explore as ex
from mlearn import transform as tf
from mlearn import util as ut
from mlearn import evaluate as ev

target = 'income'
#-------------------------------------------------------------------------
# adult
#-------------------------------------------------------------------------
reload(tf)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
dataDir = 'C:\\Users\\m038402\\Documents\\myWork\\Codes\\sparkProj\\data\\'
data_file = 'adult.csv'
df = pd.read_csv(dataDir + data_file, header=None)
column_names = ["age","workclass","fnlwgt","education","education_num","marital_status",
               "occupation","relationship","race", "sex", "capital_gain", "capital_loss", 
               "hours_per_week", "native_country", "income"]


df.columns = column_names
df.head()

#-------------------------------------------------------------------------
# encode whole data frame
#-------------------------------------------------------------------------
from imp import reload
reload(tf)
label_encoder = tf.DataFrameLabelEncoder()
newdf = label_encoder.transform(df)
newdf.head()

# label_encoder.fit_transform(df)
# label_encoder.get_all_transform_map()
# label_encoder.get_transform_map(target)
target_f = label_encoder.get_transformed_index(target, ' >50K')
target_f

# categorical_column = ex.get_categorical_column(df)
# categorical_column

#-------------------------------------------------------------------------
# split train / text data frame
#-------------------------------------------------------------------------
exclusions = []
# X, y = ut.extract_feature_target(newdf, target, exclusions)
# X_train, X_test, y_train, y_test = \
#         train_test_split(X, y, test_size=0.20, random_state=1)

X_train, X_test, y_train, y_test = ut.df_train_test_split(newdf, target, exclusions=exclusions, test_size=0.3)

X_train.shape, y_train.shape, X_test.shape, y_test.shape
X_train[:5]
y_train[:5]




#--------------------------------------------------------------------
# Pipeline test
#--------------------------------------------------------------------
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipe_lr = Pipeline([('scl', StandardScaler()),
                    ('clf', LogisticRegression(random_state=1))])

pipe_lr.fit(X_train, y_train)

# prediction: testing set
y_pred = pipe_lr.predict(X_test)
y_pred[:10]


# target count
# df.groupby(target).count()
df.groupby(target)[target].count()

#--------------------------------------------------------------------
reload(ev)
label_encoder.get_transformed_index(target, ' <=50K')

low_income_evaluator = ev.ClassificationEvaluator(pipe_lr, X_train, y_train, X_test, y_test, 
                                       cv=10, target_f=0, nsizes=10, n_jobs=-1)

low_income_evaluator.get_gini()
# unbalanced sample: low lift
low_income_evaluator.get_lift_chart()
low_income_evaluator.plot_lift_chart()
low_income_evaluator.plot_lift_chart(cumulative=True)


low_income_evaluator.plot_roc_curve()
low_income_evaluator.show_confusion_matrix()
low_income_evaluator.target_f

#--------------------------------------------------------------------
label_encoder.get_transformed_index(target, ' >50K')

high_income_evaluator = ev.ClassificationEvaluator(pipe_lr, X_train, y_train, X_test, y_test, 
                                       cv=10, target_f=1, nsizes=10, n_jobs=-1)
high_income_evaluator.get_gini()

high_income_evaluator.plot_lift_chart()
high_income_evaluator.plot_lift_chart(cumulative=True)
high_income_evaluator.plot_roc_curve()
high_income_evaluator.show_confusion_matrix()
high_income_evaluator.kfold_validate()
high_income_evaluator.get_performance_metrics()
high_income_evaluator.target_f

high_income_evaluator.plot_learning_curve()

# low_income_evaluator.score_df.head(100)
# low_income_evaluator.kfold_validate()
# low_income_evaluator.score_df.tail(100)






#--------------------------------------------------------------------
# Ensemble Evaluator
#--------------------------------------------------------------------
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from mlearn import ensemble as en

clf2 = DecisionTreeClassifier(max_depth=1, criterion='entropy', random_state=0)
clf3 = KNeighborsClassifier(n_neighbors=1, p=2, metric='minkowski')

# majority vote classifier
eclf1 = en.MajorityVoteClassifier([clf2, clf3])

evaluator = ev.EnsembleEvaluator(eclf1, X_train, y_train, X_test, y_test, 
                    cv=10, target_f=1, nsizes=10, n_jobs=-1)

evaluator.get_all_gini()
evaluator.plot_all_roc_curve()

#--------------------------------------------------------------------
# Low Lift 
#--------------------------------------------------------------------
# sorted lift score
low_income_evaluator.score_df.head()
numSamples = low_income_evaluator.score_df.shape[0]
numSamples # 6513
low_income_evaluator.score_df.groupby('true')['true'].count()
# true
# 0    5026
# 1    1487
upper_pct, lower_pct = 0.1, 0
int(numSamples * upper_pct) # 651
# model rate (646)
sum(low_income_evaluator.score_df['true'][0:651] == low_income_evaluator.target_f)
# base rate (502)
sum(low_income_evaluator.score_df['true'] == low_income_evaluator.target_f) * upper_pct - sum(low_income_evaluator.score_df['true'] == low_income_evaluator.target_f) * lower_pct
# max lift
651 / 502



#--------------------------------------------------------------------
# Decision tree grid search
#--------------------------------------------------------------------
from imp import reload
reload(ev)
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=0)
param_grid=[{'max_depth': [ 3, 5, 10, None]}]

evaluator = ev.ClassificationEvaluator(dt, X_train, y_train, X_test, y_test, cv=10, 
                                       target_f=1, nsizes=10, n_jobs=1)

evaluator.grid_search_validate(param_grid)
evaluator.get_gini()
evaluator.plot_roc_curve()

#--------------------------------------------------------------------
# random forest
#--------------------------------------------------------------------
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf_evaluator = ev.ClassificationEvaluator(rf, X_train, y_train, X_test, y_test, cv=10, 
                                          target_f=1, nsizes=10, n_jobs=1)

rf_evaluator.get_gini() # 0.76752873403582966
rf_evaluator.plot_roc_curve()
rf_evaluator.get_lift_chart()
rf_evaluator.score_df.head()

#--------------------------------------------------------------------
# get probabilty score
#--------------------------------------------------------------------
