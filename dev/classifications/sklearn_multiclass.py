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

from imp import reload
import transform
reload(transform)
import transform as tf


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#-------------------------------------------------------------------------
# iris
#-------------------------------------------------------------------------
target = 'species'
dataDir = 'C:\\Users\\m038402\\Documents\\myWork\\Codes\\sparkProj\\data\\'
data_file = 'iris.csv'
df = pd.read_csv(dataDir + data_file)

target = 'species'
ex.count_levels(df, 'species')
#-------------------------------------------------------------------------
# abalone
#-------------------------------------------------------------------------
# name of the column containing the target
target = 'sex'
dataDir = 'C:\\Users\\m038402\\Documents\\myWork\\Codes\\sparkProj\\data\\'
data_file = 'abalone.csv'
df = pd.read_csv(dataDir + data_file)

target = 'sex'
ex.count_levels(df, 'sex')
#-------------------------------------------------------------------------
# explore
#-------------------------------------------------------------------------
df.columns
df.head()

# categorical_columns = ex.get_categorical_column(df, target)
# categorical_columns
ut.get_unique_values(df, target)
ex.count_unique_values(df, subset=['species'])

#-------------------------------------------------------------------------
# encode whole data frame (including target)
#-------------------------------------------------------------------------
label_encoder = tf.DataFrameLabelEncoder()
newdf = label_encoder.transform(df)
newdf.head()
# if no categorical
# newdf = df

#------------------------------------------------------------------------- 
# split train / test
exclusions = []
X_train, X_test, y_train, y_test, features = ut.df_train_test_split(newdf, target, exclusions=exclusions, test_size=0.3)

X_train[:5]
y_train[:5]

# to find out the encoded index
label_encoder.get_transformed_index(target, 'M')
label_encoder.get_transformed_index(target, 'I')
label_encoder.get_transformed_index(target, 'F')

label_encoder.get_all_transform_map()
# {'sex': {'F': 0, 'I': 1, 'M': 2}}
# {'species': {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}}

# g = lambda x: label_encoder.get_transformed_index(target, x)
# g('Iris-setosa')
# list(zip(ut.get_unique_values(df, target), map(g, ut.get_unique_values(df, target))))

#-------------------------------------------------------------------------
# data check
#-------------------------------------------------------------------------
X_train[:5]
X_train.shape, X_test.shape
y_train[:5]
y_train.shape, y_test.shape

np.unique(y_train)
df.shape
# find proportion
df.groupby('sex')['sex'].count() / df.shape[0]
# F    0.312904
# I    0.321283
# M    0.365813

pd.DataFrame({'y': y_train}).y.value_counts() / len(y_train)
# M    0.365720
# I    0.321245
# F    0.313035

newdf.groupby('sex')['sex'].count() / newdf.shape[0]
pd.DataFrame({'y': y_train}).y.value_counts() / len(y_train)
# 2    0.365720
# 1    0.321245
# 0    0.313035

#-------------------------------------------------------------------------
from imp import reload
reload(ut)
reload(tf)
reload(ev)


#-------------------------------------------------------------------------
# OneVsRest
#-------------------------------------------------------------------------
sum(result.VIR)

#-------------------------------------------------------------------------
# test evaluator - OVR
#-------------------------------------------------------------------------
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import label_binarize, LabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier


#####  For OVR - it must use label binarizer!!
lb = LabelBinarizer()
X_train, X_test, y_train, y_test, features = ut.df_train_test_split(df, target, binarizer=lb, test_size=0.3)
X_train.shape, X_test.shape
y_train.shape, y_test.shape

np.unique(y_train)

# target class vs others 
target_idx = ut.get_LabelBinarizer_index(lb, 'F')
target_idx = ut.get_LabelBinarizer_index(lb, 'Iris-setosa')

#-------------------------------------------------------------------------
# algorithms
gbm = GradientBoostingClassifier()
gbm_ovr = OneVsRestClassifier(gbm)
dt = DecisionTreeClassifier()
dt_ovr = OneVsRestClassifier(dt)


gbm_evaluator = ev.MultiClassEvaluator(gbm_ovr, X_train, y_train, X_test, y_test, features,
                                       cv=10, target_f=target_idx, target_column = target, 
                                       label_encoder=lb, nsizes=10, n_jobs=-1)

gbm_evaluator.plot_roc_curve()
gbm_evaluator.get_importance_features()
gbm_evaluator.plot_importance_features()
gbm_evaluator.features

gbm_evaluator.get_gini()

#-------------------------------------------------------------------------
evaluator = ev.MultiClassEvaluator(dt_ovr, X_train, y_train, X_test, y_test, features,
                                       cv=10, target_f=target_idx, target_column = target, 
                                       label_encoder=lb, nsizes=10, n_jobs=-1)

evaluator.plot_lift_chart()
evaluator.plot_lift_chart(cumulative=True)

evaluator.get_importance_features()

evaluator.plot_roc_curve()
evaluator.get_auc()

# evaluator.get_target_transform_map()['F']
evaluator.get_gini()
evaluator.get_gini(target='M')

evaluator.get_inverse_transform_map()
evaluator.get_target_transform_map()

evaluator.is_fitted()

evaluator.show_confusion_matrix()
evaluator.get_confusion_matrix()

evaluator.get_performance_metrics()


#-------------------------------------------------------------------------
# grid search
#-------------------------------------------------------------------------
reload(ev)
evaluator = ev.MultiClassEvaluator(gbm_ovr, X_train, y_train, X_test, y_test, features,
                                       cv=10, target_f=target_idx, target_column = target, 
                                       label_encoder=lb, nsizes=10, n_jobs=-1)

# OVR estimator keys
evaluator.estimator.get_params().keys()
param_grid=[{'estimator__max_depth': [ 3, 5, 10],
            'estimator__max_features': [3, 6]}]
evaluator.grid_search_validate(param_grid)

evaluator.plot_learning_curve()
evaluator.kfold_validate()


#-------------------------------------------------------------------------
# test evaluator - multiclass
#-------------------------------------------------------------------------
reload(ev)
reload(ut)

le = tf.DataFrameLabelEncoder()
newdf = le.transform(df)
X_train, X_test, y_train, y_test, features = ut.df_train_test_split(newdf, target, test_size=0.3)
X_train.shape, X_test.shape
y_train.shape, y_test.shape
features

# get target index
target_idx = le.get_transformed_index(target, 'F')

gbm = GradientBoostingClassifier()
rf = RandomForestClassifier()
ada = AdaBoostClassifier()

evaluator = ev.MultiClassEvaluator(rf, X_train, y_train, X_test, y_test, features,
                                    score_method="accuracy",
                                    cv=10, target_f=target_idx, target_column = 'sex', 
                                    label_encoder=le, nsizes=10, n_jobs=1)

evaluator = ev.MultiClassEvaluator(ada, X_train, y_train, X_test, y_test, features,
                                    score_method="accuracy",
                                    cv=10, target_f=target_idx, target_column = 'sex', 
                                    label_encoder=le, nsizes=10, n_jobs=1)


evaluator = ev.MultiClassEvaluator(gbm, X_train, y_train, X_test, y_test, features,
                                    score_method="accuracy",
                                    cv=10, target_f=target_idx, target_column = 'sex', 
                                    label_encoder=le, nsizes=10, n_jobs=1)

evaluator.plot_lift_chart()
evaluator.plot_lift_chart(cumulative=True)
    
evaluator.get_lift_chart()
evaluator.plot_roc_curve()

evaluator.get_target_transform_map()
le.get_all_transform_map()

evaluator.get_target_levels()
evaluator.get_num_target_levels()
evaluator.get_target_levels()

evaluator.get_importance_features()
evaluator.plot_importance_features()
# tmp.sort_values(by='VIR', ascending=False)
# evaluator.estimator.feature_importances_

# result = evaluator.get_auc()

evaluator.get_gini()
evaluator.get_gini(target='M')
evaluator.get_gini(target=2)
evaluator.get_gini(target=0)

evaluator.train()
evaluator.is_fitted()

# confusion matrix
confmat = evaluator.get_confusion_matrix()
evaluator.get_performance_metrics()
evaluator.show_confusion_matrix()

# takes awhile..
evaluator.plot_learning_curve()
evaluator.kfold_validate()

evaluator.is_fitted()

# import matplotlib.pyplot as plt
result = evaluator.get_importance_features()
result[:5]




#-------------------------------------------------------------------------
# debug
data = {'feature': evaluator.features, 'VIR': evaluator.estimator.feature_importances_}
data
pd.DataFrame(data, columns=['feature', 'VIR'])
# pd.DataFrame(data)
pd.DataFrame([evaluator.estimator.feature_importances_, evaluator.features])

# pd.DataFrame(evaluator.estimator.feature_importances_, index=evaluator.features)
# pd.DataFrame([evaluator.features, evaluator.estimator.feature_importances_])

#-------------------------------------------------------------------------
# grid search
#-------------------------------------------------------------------------
evaluator.estimator.get_params()
evaluator.estimator.get_params().keys()
param_grid=[{'max_depth': [ 3, 5, 10],
            'learning_rate': [0.05, 0.3, 1.0]}]
param_grid=[{'max_depth': [ 3, 5, 10]}]
evaluator.grid_search_validate(param_grid)

# evaluator.assess_regulation(param_grid)

# from sklearn.model_selection import GridSearchCV
# gs = GridSearchCV(estimator=gbm,
#                   param_grid=param_grid,
#                   scoring="roc_auc",
#                   cv=5,
#                   n_jobs=1)
# 
# gs.fit(X_train, y_train)

#-------------------------------------------------------------------------
# TODO: variable importance (VIR) 
#-------------------------------------------------------------------------
evaluator.train()
evaluator.is_fitted()
evaluator.estimator.feature_importances_









#-------------------------------------------------------------------------
# DEBUG
#-------------------------------------------------------------------------


# plt.title("Feature importances")
# result.plot.barh(legend=False).invert_yaxis()
# 
# plt.yticks(result.feature.values)
# plt.yticks(result.feature)

# result.plot.barh(result.feature, result.VIR, legend=False).invert_yaxis()

# left = 0.25
# bottom = 0.05
# width = 0.7
# height = 0.8
# size = (8, 6)
# fig = plt.figure(figsize=size)
# 
# ax = fig.add_axes([left, bottom, width, height])
# 
# y_pos = np.arange(len(result.feature))
# ax.barh(y_pos, result.VIR.values)
# ax.invert_yaxis()
# ax.set_yticks(y_pos)
# ax.set_yticklabels(result.feature)
# plt.title("Feature importances")

# locs, labels = plt.yticks()
# # locs, labels
# plt.yticks( locs, result.feature.values )
# 
# locs




#-------------------------------------------------------------------------
# OVR ROC
#-------------------------------------------------------------------------
y_score = gbm_ovr.decision_function(X_test)
y_score = gbm_ovr.predict(X_test)
y_score = gbm_ovr.predict_proba(X_test)
y_score[:5]
y_test[:5]


# multilabel indicator not supported
# fpr, tpr, thresholds = roc_curve(y_test, scores01, pos_label=2)
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
roc_auc
fpr
tpr

import matplotlib.pyplot as plt
plt.figure()
lw = 2
# class 2
plt.plot(fpr[2], tpr[2], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


print(classification_report(y_test, pred01))

# multilabel indicator is not supported 
confusion_matrix(y_test, pred01)
                 
# TODO: other metrics
accuracy_score(y_test, pred01)
accuracy_score(y_test, pred02)
accuracy_score(y_test, pred03)

keys = [1,2,3]
dict(zip(keys, [None] * 3))
dict(zip(keys, [np.linspace(0, 1, 100)] * 3))

#-------------------------------------------------------------------------
# multi class ROC 
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py 
#-------------------------------------------------------------------------
# import numpy as np
# import matplotlib.pyplot as plt
# from itertools import cycle
# 
# from sklearn import svm, datasets
# 
# from scipy import interp
# 
# # Import some data to play with
# iris = datasets.load_iris()
# X = iris.data
# y = iris.target
# y[:5]
# 
# # Binarize the output
# y = label_binarize(y, classes=[0, 1, 2])
# n_classes = y.shape[1]
# 
# # Add noisy features to make the problem harder
# random_state = np.random.RandomState(0)
# n_samples, n_features = X.shape
# X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]
# 
# # shuffle and split training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5, random_state=0)
# 
# X_train[:5]
# y_train[:5]
# X_test.shape, y_test.shape
# 
# # Learn to predict each class against the other
# classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True, random_state=random_state))
# y_score = classifier.fit(X_train, y_train).decision_function(X_test)
# 
# # Compute ROC curve and ROC area for each class
# fpr = dict()
# tpr = dict()
# roc_auc = dict()
# for i in range(n_classes):
#     fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
#     roc_auc[i] = auc(fpr[i], tpr[i])
# 
# # Compute micro-average ROC curve and ROC area
# fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
# roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
# 
# roc_auc
# fpr
# tpr

#-------------------------------------------------------------------------
# multiclass
#-------------------------------------------------------------------------
# Compute macro-average ROC curve and ROC area

# First aggregate all false positive rates
# all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# 
# # Then interpolate all ROC curves at this points
# mean_tpr = np.zeros_like(all_fpr)
# for i in range(n_classes):
#     mean_tpr += interp(all_fpr, fpr[i], tpr[i])
# 
# # Finally average it and compute AUC
# mean_tpr /= n_classes
# 
# fpr["macro"] = all_fpr
# tpr["macro"] = mean_tpr
# roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
# 
# # Plot all ROC curves
# plt.figure()
# plt.plot(fpr["micro"], tpr["micro"],
#          label='micro-average ROC curve (area = {0:0.2f})'
#                ''.format(roc_auc["micro"]),
#          color='deeppink', linestyle=':', linewidth=4)
# 
# plt.plot(fpr["macro"], tpr["macro"],
#          label='macro-average ROC curve (area = {0:0.2f})'
#                ''.format(roc_auc["macro"]),
#          color='navy', linestyle=':', linewidth=4)
# 
# colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
# for i, color in zip(range(n_classes), colors):
#     plt.plot(fpr[i], tpr[i], color=color, lw=lw,
#              label='ROC curve of class {0} (area = {1:0.2f})'
#              ''.format(i, roc_auc[i]))
# 
# plt.plot([0, 1], [0, 1], 'k--', lw=lw)
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Some extension of Receiver operating characteristic to multi-class')
# plt.legend(loc="lower right")
# plt.show()



#-------------------------------------------------------------------------
# random forest classifier
# from sklearn.ensemble import RandomForestClassifier
# 
# rf = RandomForestClassifier()
# rf.fit(X_train, y_train)
# pred = rf.predict(X_test)
# 
# print(rf.feature_importances_)

#-------------------------------------------------------------------------
# multiclass evaluate
#-------------------------------------------------------------------------
from sklearn.metrics import confusion_matrix, auc, roc_curve, classification_report, accuracy_score

# multi-class confusion matrix
confusion_matrix(y_test, pred)

# random forest
probas = rf.fit(X_train, y_train).predict_proba(X_test)
probas[:10]

fpr, tpr, thresholds = roc_curve(y_test, probas[:, 0], pos_label=0)
auc(fpr, tpr) # 0.67053808182205599
fpr, tpr, thresholds = roc_curve(y_test, probas[:, 1], pos_label=1)
auc(fpr, tpr) # 0.87571037430784981
fpr, tpr, thresholds = roc_curve(y_test, probas[:, 2], pos_label=2)
auc(fpr, tpr) # 0.62286211479700193


# TODO: more evaluation
accuracy_score(y_test, pred)

# GBM multiclass
gbm = GradientBoostingClassifier()
probas = gbm.fit(X_train, y_train).predict_proba(X_test)

fpr, tpr, thresholds = roc_curve(y_test, probas[:, 0], pos_label=0)
auc(fpr, tpr) # 0.71398237369193618
fpr, tpr, thresholds = roc_curve(y_test, probas[:, 1], pos_label=1)
auc(fpr, tpr) # 0.90214402556618534
fpr, tpr, thresholds = roc_curve(y_test, probas[:, 2], pos_label=2)
auc(fpr, tpr) # 0.67478247763116417





#-------------------------------------------------------------------------
# encode target
#-------------------------------------------------------------------------
# label_encoder = tf.DataFrameLabelEncoder()
# newdf = label_encoder.transform(df)
# newdf.head()
# 
# target = 'sex'
# target_f = label_encoder.get_transformed_index(target, 'F')
# target_f
# 
# label_encoder.get_all_transform_map()

#-------------------------------------------------------------------------
# split data frame to feature and target
#-------------------------------------------------------------------------
# exclusions = []
# X, y = ut.extract_feature_target(newdf, target, exclusions)
# X.shape  # (4177, 8)
# y.shape  # (4177,)

#-------------------------------------------------------------------------
# train / test split
#-------------------------------------------------------------------------
# from mlearn import util as ut 
# X, y = ut.extract_feature_target(newdf, target, exclusions)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)




# Don't use this
# from sklearn.preprocessing import MultiLabelBinarizer
# mlb = MultiLabelBinarizer()
# y_indicator = mlb.fit_transform(y[:, None])
# y_indicator
# y_indicator.shape  # (4177, 3)
# mlb.classes_

#-------------------------------------------------------------------------
# NOT USE: multioutput classification
#-------------------------------------------------------------------------
from sklearn.datasets import make_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
import numpy as np

# create artificial data set
X, y1 = make_classification(n_samples=10, n_features=100, n_informative=30, n_classes=3, random_state=1)
y2 = shuffle(y1, random_state=1)
y3 = shuffle(y1, random_state=2)
Y = np.vstack((y1, y2, y3)).T
n_samples, n_features = X.shape # 10,100
n_outputs = Y.shape[1] # 3
n_classes = 3
X[:10]
Y[:10]
X.shape
Y.shape

forest = RandomForestClassifier(n_estimators=100, random_state=1)
multi_target_forest = MultiOutputClassifier(forest, n_jobs=-1)
multi_target_forest.fit(X, Y).predict(X)



#-------------------------------------------------------------------------
# indexer
#-------------------------------------------------------------------------
from sklearn.preprocessing import MultiLabelBinarizer
y = [[2, 3, 4], [2], [0, 1, 3], [0, 1, 2, 3, 4], [0, 1, 2]]
MultiLabelBinarizer().fit_transform(y)

#-------------------------------------------------------------------------
# One vs the rest
#-------------------------------------------------------------------------
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
iris = datasets.load_iris()
X, y = iris.data, iris.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
pred = OneVsRestClassifier(LinearSVC(random_state=0)).fit(X_train, y_train).predict(X_test)

import pandas as pd
pd.DataFrame({'true': y_test, 'pred': pred})

#-------------------------------------------------------------------------
# multiclass learning
#-------------------------------------------------------------------------
from sklearn.multiclass import  OutputCodeClassifier
clf = OutputCodeClassifier(LinearSVC(random_state=0),
                            code_size=2, random_state=0)
#clf.fit(X, y).predict(X)
pred = clf.fit(X_train, y_train).predict(X_test)
pd.DataFrame({'true': y_test, 'pred': pred})
