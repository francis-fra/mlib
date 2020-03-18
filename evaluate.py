"""
Author: Francis Chan

Classifier Evaluator for classification models

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from scipy import interp
from sklearn.base import clone
from sklearn.pipeline import _name_estimators
import ensemble as en
# from . import ensemble as en 
from sklearn.exceptions import NotFittedError

from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

class ClassificationEvaluator(object):
    def __init__(self, estimator, X_train, y_train,
                 X_test, y_test, id, features, cv, target_f, score_method="roc_auc",
                 nsizes=10, train_now=False, n_jobs=-1):
        self.estimator = estimator
        self.X = X_train
        self.y = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.id = id
        self.features = features
        self.cv = cv
        # numerical index of the target
        self.target_f = target_f
        self.nsizes = nsizes
        self.n_jobs = n_jobs
        self.score_method = score_method
        self.score_df = None
        self.prob_df = None

        self.assertData()

        if train_now == True:
            self.train()

    def set_param(self, name, val):
        setattr(self, name, val)

    def assertData(self):
        "assert the correct data type"
        assert(isinstance(self.X, np.ndarray))
        assert(isinstance(self.X_test, np.ndarray))
        assert(isinstance(self.y, np.ndarray))
        assert(isinstance(self.y_test, np.ndarray))

    def get_probability(self, X=None):
        "get the probability for each class"

        if X is None:
            X = self.X_test

        if self.is_fitted() == False:
            self.train()

        return self.estimator.predict_proba(X)
        
    def score(self, X=None):
        "score the fitted model"
        
        if X is None:
            X = self.X_test

        if self.is_fitted() == False:
            self.train()

        #y_pred = self.estimator.predict_proba(X)
        y_pred = self.get_probability(X)
        idx = self.target_f
        # get probability only for the specified target class
        score = y_pred[:,idx]
        
        # FIXME: add supplied feature column
        # column get id column in X
        ix = np.isin(self.features, self.id)
        try:
            colidx = np.where(ix)[0][0]
            self.prob_df = pd.DataFrame({self.id: X[:,colidx], 'score': score})
        except:
            print("No such column in feature data?")
            return self

        return self

    def get_score_table(self):
        "return the score table"

        return (self.prob_df)

    def train(self):
        "fit the estimator"

        self.estimator = self.estimator.fit(self.X, self.y)

        return self

    def is_fitted(self, estimator=None):
        "check if the estimator is fitted"

        if estimator is None:
            estimator = self.estimator

        try:
            estimator.predict(self.X[:1])
            fitted = True
        except NotFittedError as e:
            fitted = False
        except Exception as e:
            print (str(e))

        return fitted


    def get_lift_chart(self, data="test", cumulative=False):
        '''
            Get lift:
                INPUTS:
                    data : out of sample (test) set or training set
                OUTPUTS:
                    lift : lift data
        '''

        if data == "test":
            X = self.X_test
            y = self.y_test

        else:
            X = self.X
            y = self.y

        if self.is_fitted():
            y_pred = self.estimator.predict_proba(X)
        else:
            y_pred = clone(self.estimator).fit(self.X, self.y).predict_proba(X)

        lift_range = list(np.arange(0.1, 1.1, 0.1))

        # target index
        idx = self.target_f
        self.score_df = pd.DataFrame({'pred': y_pred[:,idx], 'true': y})
        if self.target_f == 1:
            self.score_df.sort_values(by=['pred'], ascending=False, inplace=True)
        else:
            self.score_df.sort_values(by=['pred'], ascending=True, inplace=True)
        self.score_df.reset_index(drop=True, inplace=True)

        # decile lift
        # get lower and upper range
        extended_range = [0] + lift_range
        lower_upper_list = list(zip(extended_range[:-1], extended_range[1:]))

        # cumulative lift
        if cumulative is False:
            result = [(upper_pct, self.get_lift(lower_pct, upper_pct))
                      for lower_pct, upper_pct in lower_upper_list]
        else:
            result = [(pct, self.get_cumulative_lift(pct)) for pct in lift_range]

        return (result)

    def get_cumulative_lift(self, percentile):
        "get lift for the given percentile"

        # random proportion rate
        base_rate = sum(self.score_df['true'] == self.target_f) * percentile
        # model rate
        numSamples = self.score_df.shape[0]
        cutoff = int(numSamples * percentile)
        # count targets
        model_rate = sum(self.score_df['true'][:cutoff] == self.target_f)

        lift = model_rate / base_rate
        prob_cutoff = self.score_df['pred'][cutoff-1]

        return ((lift, prob_cutoff))

    def get_lift(self, lower_pct, upper_pct):
        "get lift for the given percentile"

        base_rate = sum(self.score_df['true'] == self.target_f) * upper_pct - sum(self.score_df['true'] == self.target_f) * lower_pct
        # model rate
        numSamples = self.score_df.shape[0]
        upper_cutoff = int(numSamples * upper_pct)
        lower_cutoff = int(numSamples * lower_pct)
        # count targets
        model_rate = sum(self.score_df['true'][lower_cutoff:upper_cutoff] == self.target_f)

        lift = model_rate / base_rate
        prob_cutoff = self.score_df['pred'][upper_cutoff-1]

        return ((lift, prob_cutoff))

    def plot_lift_chart(self, data="test", cumulative=False):
        "lift chart or cumulative response"

        result = self.get_lift_chart(data, cumulative)
        pct_range = [int(x*100) for x, (y, z) in result]
        lift = [y for x, (y, z) in result]

        fig, ax = plt.subplots()
        if cumulative == False:
            plt.bar(pct_range, lift)
            # include zero
            x_range = [0] + pct_range
            N = len(pct_range) + 1

            plt.plot(x_range, [1.0]*N, 'r--')
            plt.xlim([0, 100])

            plt.xticks(pct_range)
            plt.grid()
            plt.xlabel('Percentile')
            plt.ylabel('Lift')
            plt.title('Lift Chart')
        else:
            # cumulative response
            pct = [x*y for (x,y) in zip(pct_range, lift)]
            # include zero
            x_range = [0] + pct_range

            pct = [0] + pct
            plt.plot(x_range, pct, 'r')
            plt.xlim([0, 100])
            plt.plot(x_range, x_range, 'k--')

            plt.xticks(pct_range)
            plt.grid()
            plt.xlabel('Percentile')
            plt.title('Cumulative Response Rate')

        plt.tight_layout()


    def plot_learning_curve(self):
        "plot accuracy vs num samples"
        line = np.linspace(0.1, 1.0, self.nsizes)
        train_sizes, train_scores, test_scores =\
                        learning_curve(estimator=self.estimator,
                                       X=self.X,
                                       y=self.y,
                                       train_sizes=line,
                                       cv=self.cv,
                                       n_jobs=self.n_jobs)

        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        plt.plot(train_sizes, train_mean,
                 color='blue', marker='o',
                 markersize=5, label='training accuracy')

        plt.fill_between(train_sizes,
                         train_mean + train_std,
                         train_mean - train_std,
                         alpha=0.15, color='blue')

        plt.plot(train_sizes, test_mean,
                 color='green', linestyle='--',
                 marker='s', markersize=5,
                 label='validation accuracy')

        plt.fill_between(train_sizes,
                         test_mean + test_std,
                         test_mean - test_std,
                         alpha=0.15, color='green')

        plt.grid()
        plt.xlabel('Number of training samples')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')
        plt.tight_layout()

    def kfold_validate(self, estimator=None, score_method=None):
        '''
            K fold cross validation

            Valid options are ['accuracy', 'adjusted_rand_score', 'average_precision',
                'f1', 'f1_macro', 'f1_micro', 'f1_samples', 'f1_weighted', 'neg_log_loss',
                'neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_median_absolute_error',
                'precision', 'precision_macro', 'precision_micro', 'precision_samples',
                'precision_weighted', 'r2', 'recall', 'recall_macro', 'recall_micro',
                'recall_samples', 'recall_weighted', 'roc_auc']

        '''

        if estimator is None:
            estimator = self.estimator
        if score_method is None:
            score_method = self.score_method

        estimator_name = _name_estimators([estimator])[0][0]

        scores = cross_val_score(estimator=estimator,
                                 X=self.X,
                                 y=self.y,
                                 cv=self.cv,
                                 scoring = score_method,
                                 n_jobs=self.n_jobs)

        self._print_title(estimator_name)
        print("{:d}-Fold {} score: {:.3f} +/- {:.3f}".format(self.cv, score_method,
                                                          np.mean(scores), np.std(scores)))
        return (scores)

    def _print_title(self, title, total_length=40):
        N = len(title)
        tail_len = total_length - N
        head_len = total_length >> 1
        print("{}  {}  {}".format('=' * head_len, title, '=' * tail_len))

    def assess_regulation(self, reg_name, reg_range):
        "Regularization curve"
        train_scores, test_scores = validation_curve(
                                            estimator=self.estimator,
                                            X=self.X,
                                            y=self.y,
                                            param_name=reg_name,
                                            param_range=reg_range,
                                            cv = self.cv)

        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        plt.plot(reg_range, train_mean, color='blue',
                 marker='o', markersize=5, label='training accuracy')

        plt.fill_between(reg_range, train_mean + train_std,
                         train_mean - train_std, alpha=0.15,
                         color='blue')

        plt.plot(reg_range, test_mean,
                 color='green', linestyle='--',
                 marker='s', markersize=5,
                 label='validation accuracy')

        plt.fill_between(reg_range,
                         test_mean + test_std,
                         test_mean - test_std,
                         alpha=0.15, color='green')

        plt.grid()
        plt.xscale('log')
        plt.legend(loc='lower right')
        plt.xlabel('Parameter C')
        plt.ylabel('Accuracy')
        plt.tight_layout()


    def grid_search(self, reg_grid, score_method=None):
        "return a fitted optimized classifier after hyperparameter search"

        if score_method is None:
            score_method = self.score_method

        gs = GridSearchCV(estimator=self.estimator,
                          param_grid=reg_grid,
                          scoring=score_method,
                          cv=self.cv,
                          n_jobs=self.cv)

        gs = gs.fit(self.X, self.y)
        print("Best score ({}): {:.3f}".format(score_method, gs.best_score_))
        print("Best params:")
        print(gs.best_params_)
        return (gs)

    def grid_search_validate(self, reg_grid, score_method=None):
        "grid search and validate"
        if score_method is None:
            score_method = self.score_method
        gs = self.grid_search(reg_grid, score_method)
        scores = self.kfold_validate(gs)
        return (scores)

    def get_confusion_matrix(self):
        "out of sample testing"
        self.estimator.fit(self.X, self.y)
        if self.is_fitted():
            y_pred = self.estimator.predict(self.X_test)
        else:
            y_pred = clone(self.estimator).fit(self.X, self.y).predict(self.X_test)
        confmat = confusion_matrix(y_true = self.y_test, y_pred=y_pred)
        return (confmat)

    def show_confusion_matrix(self):
        "plot confusion matrix"
        confmat = self.get_confusion_matrix()
        fig, ax = plt.subplots()
        ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
        for i in range(confmat.shape[0]):
            for j in range(confmat.shape[1]):
                ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

        plt.xlabel('predicted label')
        plt.ylabel('true label')
        plt.tight_layout()

    def get_target_levels(self):
        "return unique values of target"
        return (np.unique(self.y))

    def get_num_target_levels(self):
        "return unique values of target"
        return (len(np.unique(self.y)))


        "Out of sample Gini coefficient"

        if num_splits is None:
            num_splits = self.cv

        if auc is None:
            auc, mean_tpr, mean_fpr = self.get_auc(num_splits, splitter, estimator)
        return (2 * auc - 1)

    def get_auc(self, num_splits=None, splitter=None, estimator=None):
        "Out of sample AUC estimate"

        if estimator is None:
            estimator = self.estimator

        if num_splits is None:
            num_splits = self.cv

        # must be greater than 1
        num_splits = max(2, num_splits)

        # default is stratified KFold
        # (for more splitter see: sklearn.model_selection)
        if splitter is None:
            skf = StratifiedKFold(n_splits=num_splits, random_state=1)
        else:
            skf = splitter

        num_cv = skf.get_n_splits(self.X, self.y)

        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, 100)
        #all_tpr = []

        # split the training data set
        k = 0
        for train_index, test_index in skf.split(self.X, self.y):
            k += 1
            xtrain, xtest = self.X[train_index], self.X[test_index]
            ytrain, ytest = self.y[train_index], self.y[test_index]

            probas = clone(estimator).fit(xtrain, ytrain).predict_proba(xtest)

            fpr, tpr, thresholds = roc_curve(ytest, probas[:, self.target_f], pos_label=self.target_f)
            mean_tpr += interp(mean_fpr, fpr, tpr)
            mean_tpr[0] = 0.0
            #roc_auc = auc(fpr, tpr)

        mean_tpr /= num_cv
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)

        return (mean_auc, mean_tpr, mean_fpr)


    def get_performance_metrics(self, metrics=None, estimator=None):
        "print performance metrics"

        # for more see sklearn.metrics
        if metrics is None:
            metrics = [confusion_matrix, classification_report, accuracy_score]

        if estimator is None:
            estimator = self.estimator

        # estimator must be fitted first
        if self.is_fitted():
            y_pred = estimator.predict(self.X_test)
        else:
            y_pred = clone(estimator).fit(self.X, self.y).predict(self.X_test)
        # find if the target is multi_class or not
        num_levels = self.get_num_target_levels()
        if num_levels > 2:
            method = 'weighted'
        else:
            method = 'binary'

        for f in metrics:
            self._print_title(f.__name__, 40)
            print(f(y_true=self.y_test, y_pred=y_pred))

    def plot_roc_curve(self, splitter=None, num_splits=None):
        "ROC curve"

        mean_auc, mean_tpr, mean_fpr = self.get_auc(splitter, num_splits)

        # random line
        plt.figure()
        plt.plot([0, 1], [0, 1], linestyle='--',
                 color=(0.6, 0.6, 0.6), label='Random')

        # average line
        plt.plot(mean_fpr, mean_tpr, 'k--',
                 label='mean Gini = %0.2f' % (2*mean_auc-1), lw=2)

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('false positive rate')
        plt.ylabel('true positive rate')
        plt.title('Receiver Operator Characteristic')
        plt.legend(loc="lower right")
        plt.grid()
        plt.tight_layout()

    def get_gini(self, num_splits=None, auc=None, splitter=None, estimator=None):
        "Out of sample Gini coefficient"

        if num_splits is None:
            num_splits = self.cv

        if auc is None:
            auc, tpr, fpr = self.get_auc(num_splits, splitter, estimator)

        return (2 * auc - 1)

    def get_importance_features(self):
        "importance of features for RF and GBM only"

        # check if estimator is fitted
        if not self.is_fitted():
            self.train()

        # TODO: normalize importance
        try:
            importances = self.estimator.feature_importances_
        except:
            print("The estimator has no VIR!")
            return

        data = {'feature': self.features, 'VIR': importances}
        VIR_df = pd.DataFrame(data, columns=['feature', 'VIR'])
        VIR_df = VIR_df.sort_values(by='VIR', ascending=False)
        VIR_df.VIR =  VIR_df.VIR * 100
        return (VIR_df)

    def plot_importance_features(self, limit=20):
        "plot importance of feauturs for RF and GBM only"

        VIR_df = self.get_importance_features()
        if VIR_df is None:
            return

        VIR_df = VIR_df[:limit]

        left = 0.25
        bottom = 0.15
        width = 0.7
        height = 0.8
        size = (8, 6)
        fig = plt.figure(figsize=size)
        ax = fig.add_axes([left, bottom, width, height])

        y_pos = np.arange(len(VIR_df.feature))
        ax.barh(y_pos, VIR_df.VIR.values)
        ax.invert_yaxis()
        ax.set_yticks(y_pos)
        ax.set_yticklabels(VIR_df.feature)
        plt.title("Feature importances")
        plt.xlabel('Percentage')

        return self



class MultiClassEvaluator(ClassificationEvaluator):
    """
        Evaluator for multiclass learners

        For Over the Rest (OVR) learners, target must be encoded with LabelBinarizer
        For Mult-class learners, target must be encoded with DataFrameLabelEncoder

    """
    def __init__(self, estimator, X_train, y_train,
                 X_test, y_test, id, features, cv, target_f, target_column, label_encoder,
                 score_method="accuracy", nsizes=10, train_now=False, n_jobs=-1):

        super(MultiClassEvaluator, self).__init__(estimator, X_train, y_train,
                 X_test, y_test, id, features, cv, target_f, score_method, nsizes, train_now, n_jobs)

        # over the rest estimator
        self.target_column = target_column      # name of column in the data frame
        self.OVR = isinstance(estimator, OneVsRestClassifier)
        self.label_encoder = label_encoder
        self._set_target_transform_map()

    def get_target_levels(self):
        # the key must be ordered by the transformed index
        a = [(key, idx) for (key, idx) in self.get_target_transform_map().items()]
        # sorted by transform index
        a = sorted(a, key=lambda x: x[1])

        return ([key for key, idx in a])

    def get_num_target_levels(self):
            return (len(self.get_target_transform_map()))

    def get_auc(self, num_splits=None, splitter=None, estimator=None):
        "Out of sample AUC estimate"

        if self.OVR == True:
            return (self.get_ovr_auc(num_splits, splitter, estimator))
        else:
            return (self.get_multiclass_auc(num_splits, splitter, estimator))

    def _set_target_transform_map(self):
        "get dict of target transform map"

        if self.OVR == True:
            classes = self.label_encoder.classes_
            nclasses = len(classes)
            self.target_transform_map = dict(zip(classes, range(nclasses)))
        else:
            self.target_transform_map = self.label_encoder.get_transform_map(self.target_column)

    def get_target_transform_map(self):
        "return the transform map"
        return self.target_transform_map

    def get_inverse_transform_map(self):
        "return the inverse transform map"
        return (dict([(value, key) for key, value in self.get_target_transform_map().items()]))

    def get_multiclass_auc(self, num_splits=None, splitter=None, estimator=None):
        "AUC estimate for multiclass estimator"

        if estimator is None:
            estimator = self.estimator

        if num_splits is None:
            num_splits = self.cv

        # must be greater than 1
        num_splits = max(2, num_splits)

        # default is stratified KFold
        # (for more splitter see: sklearn.model_selection)
        if splitter is None:
            skf = StratifiedKFold(n_splits=num_splits, random_state=1)
        else:
            skf = splitter

        num_cv = skf.get_n_splits(self.X, self.y)
        n_classes = self.get_num_target_levels()

        mean_tpr = dict(zip(range(n_classes), [0.0] * n_classes))
        mean_fpr = dict(zip(range(n_classes), [np.linspace(0, 1, 100)] * n_classes))
        mean_auc = dict(zip(range(n_classes), [np.linspace(0, 1, 100)] * n_classes))

        # split the training data set
        k = 0
        for train_index, test_index in skf.split(self.X, self.y):
            k += 1
            xtrain, xtest = self.X[train_index], self.X[test_index]
            ytrain, ytest = self.y[train_index], self.y[test_index]

            probas = clone(estimator).fit(xtrain, ytrain).predict_proba(xtest)

            # temporary dictionary
            fpr = dict()
            tpr = dict()
            #roc_auc = dict()
            for idx in range(n_classes):
                fpr[idx], tpr[idx], thresholds = roc_curve(ytest, probas[:, idx], pos_label=idx)
                mean_tpr[idx] += interp(mean_fpr[idx], fpr[idx], tpr[idx])
                mean_tpr[idx][0] = 0.0

        # take the average over the splits for each class
        for idx in range(n_classes):
            mean_tpr[idx] /= num_cv
            mean_tpr[idx][-1] = 1.0
            mean_auc[idx] = auc(mean_fpr[idx], mean_tpr[idx])

        return (mean_auc, mean_tpr, mean_fpr)


    def get_ovr_auc(self, num_splits=None, splitter=None, estimator=None):
        "AUC estimate for OVR estimator"

        if estimator is None:
            estimator = self.estimator

        if num_splits is None:
            num_splits = self.cv

        # must be greater than 1
        num_splits = max(2, num_splits)

        # default is stratified KFold
        # (for more splitter see: sklearn.model_selection)
        if splitter is None:
            skf = StratifiedKFold(n_splits=num_splits, random_state=1)
        else:
            skf = splitter

        num_cv = skf.get_n_splits(self.X, self.y)
        n_classes = self.get_num_target_levels()

        # final outcome (empty dict)
        #classes = self.get_target_levels()
        mean_tpr = dict(zip(range(n_classes), [0.0] * n_classes))
        mean_fpr = dict(zip(range(n_classes), [np.linspace(0, 1, 100)] * n_classes))
        mean_auc = dict(zip(range(n_classes), [np.linspace(0, 1, 100)] * n_classes))

        # inverse transform for stratification
        y_raw = self.label_encoder.inverse_transform(self.y)

        # split the training data set
        k = 0
        for train_index, test_index in skf.split(self.X, y_raw):
            k += 1
            xtrain, xtest = self.X[train_index], self.X[test_index]
            ytrain, ytest = self.y[train_index,:], self.y[test_index,:]

            probas = clone(estimator).fit(xtrain, ytrain).predict_proba(xtest)
            # temporary dictionary
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            # get auc for all classes
            for idx in range(n_classes):
                #ytrue = self.label_encoder.inverse_transform(ytest)
                fpr[idx], tpr[idx], thresholds = roc_curve(ytest[:,idx], probas[:, idx])
                roc_auc[idx] = auc(fpr[idx], tpr[idx])

            # accumulate over splits
            for idx in range(n_classes):
                mean_tpr[idx] += interp(mean_fpr[idx], fpr[idx], tpr[idx])
                mean_tpr[idx][0] = 0.0

        # take the average over the splits for each class
        for idx in range(n_classes):
            mean_tpr[idx] /= num_cv
            mean_tpr[idx][-1] = 1.0
            mean_auc[idx] = auc(mean_fpr[idx], mean_tpr[idx])

        return (mean_auc, mean_tpr, mean_fpr)

    def plot_roc_curve(self, splitter=None, num_splits=None):
        "ROV curve for OVR learners"

        mean_auc_dict, mean_tpr_dict, mean_fpr_dict = self.get_auc(splitter, num_splits)

        # random line
        plt.figure()
        plt.plot([0, 1], [0, 1], linestyle='--',
                 color=(0.6, 0.6, 0.6), label='Random')

        # ROC for each class
        classes = self.get_target_levels()
        num_classes = self.get_num_target_levels()
        for k in range(num_classes):
            plt.plot(mean_fpr_dict[k], mean_tpr_dict[k],
                     label='%s: Gini = %0.2f' % (classes[k], 2*mean_auc_dict[k]-1), lw=2)

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('false positive rate')
        plt.ylabel('true positive rate')
        plt.title('Receiver Operator Characteristic')
        plt.legend(loc="lower right")
        plt.grid()
        plt.tight_layout()

    def get_gini(self, target=None, num_splits=None, auc=None, splitter=None, estimator=None):
        "Out of sample Gini coefficient"

        # target can be text or numeric
        if isinstance(target, str):
            # transform back to index
            target_idx = self.get_target_transform_map()[target]

        if isinstance(target, int):
            target_idx = target

        if num_splits is None:
            num_splits = self.cv

        if auc is None:
            auc_dict, tpr_dict, fpr_dict = self.get_auc(num_splits, splitter, estimator)

        if target is None:
            # get all gini
            transform_map = self.get_target_transform_map()
            return [(self.get_inverse_transform_map()[key], 2*auc-1) for key, auc in auc_dict.items()]
        else:
            auc = auc_dict[target_idx]
            return (2 * auc - 1)

    def get_lift(self, lower_pct, upper_pct, target=None):
        "get lift for the given percentile"

        if target is None:
            target_idx = self.target_f

        # target can be text or numeric
        if isinstance(target, str):
            # transform back to index
            target_idx = self.get_target_transform_map()[target]

        if isinstance(target, int):
            target_idx = target

        base_rate = sum(self.score_df_dict[target_idx]['true'] == target_idx) * upper_pct - sum(self.score_df_dict[target_idx]['true'] == target_idx) * lower_pct
        # model rate
        numSamples = self.score_df_dict[target_idx].shape[0]
        upper_cutoff = int(numSamples * upper_pct)
        lower_cutoff = int(numSamples * lower_pct)
        # count targets
        model_rate = sum(self.score_df_dict[target_idx]['true'][lower_cutoff:upper_cutoff] == target_idx)

        lift = model_rate / base_rate
        prob_cutoff = self.score_df_dict[target_idx]['pred'][upper_cutoff-1]

        return ((lift, prob_cutoff))

    def get_cumulative_lift(self, percentile, target=None):
        "get lift for the given percentile"

        if target is None:
            target_idx = self.target_f

        # target can be text or numeric
        if isinstance(target, str):
            # transform back to index
            target_idx = self.get_target_transform_map()[target]

        if isinstance(target, int):
            target_idx = target

        # random proportion rate
        base_rate = sum(self.score_df_dict[target_idx]['true'] == target_idx) * percentile
        # model rate
        numSamples = self.score_df_dict[target_idx].shape[0]
        cutoff = int(numSamples * percentile)
        # count targets
        model_rate = sum(self.score_df_dict[target_idx]['true'][:cutoff] == target_idx)

        lift = model_rate / base_rate
        prob_cutoff = self.score_df_dict[target_idx]['pred'][cutoff-1]

        return ((lift, prob_cutoff))

    def get_lift_chart(self, data="test", cumulative=False):
        '''
            Get lift:
                INPUTS:
                    data : out of sample (test) set or training set
                OUTPUTS:
                    lift : lift data for all target labels
        '''

        if data == "test":
            X = self.X_test
            y = self.y_test
        else:
            X = self.X
            y = self.y

        if self.is_fitted():
            y_pred = self.estimator.predict_proba(X)
        else:
            y_pred = clone(self.estimator).fit(self.X, self.y).predict_proba(X)

        lift_range = list(np.arange(0.1, 1.1, 0.1))

        transform_map = self.get_target_transform_map()
        num_classes = self.get_num_target_levels()

        # decile lift
        # get lower and upper range
        extended_range = [0] + lift_range
        lower_upper_list = list(zip(extended_range[:-1], extended_range[1:]))

        # score dictionary for each class
        self.score_df_dict = dict()

        # if OVR, need to convert indicator (y) back to numerical index
        if self.OVR == True:
            y_raw = self.label_encoder.inverse_transform(y)
            y = list(map(lambda x: transform_map[x], y_raw))

        # loop over all classes
        for k in range(num_classes):
            self.score_df_dict[k] = pd.DataFrame({'pred': y_pred[:,k], 'true': y})
            # sort by predicted probability
            self.score_df_dict[k].sort_values(by=['pred'], ascending=False, inplace=True)
            self.score_df_dict[k].reset_index(drop=True, inplace=True)

        lift_dict = dict()
        inverse_map = self.get_inverse_transform_map()
        for k in range(num_classes):
            # translate index back to label
            label = inverse_map[k]
            # cumulative lift
            if cumulative == False:
                lift_dict[label] = [(upper_pct, self.get_lift(lower_pct, upper_pct, k))
                          for lower_pct, upper_pct in lower_upper_list]
            else:
                lift_dict[label] = [(pct, self.get_cumulative_lift(pct, k)) for pct in lift_range]

        return (lift_dict)

    def plot_lift_chart(self, data="test", cumulative=False):
        "lift chart or cumulative response"

        # dictionary of lift
        result_dict = self.get_lift_chart(data, cumulative)
        classes = self.get_target_levels()
        num_classes = self.get_num_target_levels()
        # loop over each target label
        k = 0
        plt.figure()
        for target in classes:
            k += 1
            result = result_dict[target]
            pct_range = [int(x*100) for x, (y, z) in result]
            lift = [y for x, (y, z) in result]

            plt.subplot(num_classes, 1, k)
            if cumulative == False:
                plt.bar(pct_range, lift)
                # include zero
                x_range = [0] + pct_range
                N = len(pct_range) + 1

                plt.plot(x_range, [1.0]*N, 'r--')
                plt.xlim([0, 100])

                plt.xticks(pct_range)
                plt.grid()
                plt.xlabel('Percentile')
                plt.ylabel('Lift')
                plt.title('Lift Chart: %s' % (target))
            else:
                # cumulative response
                pct = [x*y for (x,y) in zip(pct_range, lift)]
                # include zero
                x_range = [0] + pct_range

                pct = [0] + pct
                plt.plot(x_range, pct, 'r')
                plt.xlim([0, 100])
                plt.plot(x_range, x_range, 'k--')

                plt.xticks(pct_range)
                plt.grid()
                plt.xlabel('Percentile')
                plt.title('Cumulative Response Rate: %s' % (target))

            plt.tight_layout()

    def get_confusion_matrix(self):
        "out of sample testing"
        self.estimator.fit(self.X, self.y)
        if self.is_fitted():
            y_pred = self.estimator.predict(self.X_test)
        else:
            y_pred = clone(self.estimator).fit(self.X, self.y).predict(self.X_test)

        if self.OVR == True:
            transform_map = self.get_target_transform_map()
            y_raw = self.label_encoder.inverse_transform(y_pred)
            y_pred = list(map(lambda x: transform_map[x], y_raw))

            y_test = self.label_encoder.inverse_transform(self.y_test)
            y_test = list(map(lambda x: transform_map[x], y_test))
        else:
            y_test = self.y_test

        confmat = confusion_matrix(y_true = y_test, y_pred=y_pred)
        return (confmat)

    def show_confusion_matrix(self):
        "plot confusion matrix"

        confmat = self.get_confusion_matrix()
        fig, ax = plt.subplots()
        ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
        for i in range(confmat.shape[0]):
            for j in range(confmat.shape[1]):
                ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

        # convert to text label
        transform_map = self.get_inverse_transform_map()
        labels = range(self.get_num_target_levels())
        ticks = list(map(lambda x: transform_map[x], labels))
        plt.xticks(labels, ticks)
        plt.yticks(labels, ticks)

        plt.xlabel('predicted label')
        plt.ylabel('true label')
        plt.tight_layout()

    def get_performance_metrics(self, metrics=None, estimator=None):
        "print performance metrics"
        # TODO: some metric is not supported for multi-class
        pass


class EnsembleEvaluator(ClassificationEvaluator):
        "Evaluator for ensemble model"
        def __init__(self, estimator, X_train, y_train,
                 X_test, y_test, cv, target_f, score_method="roc_auc",
                 nsizes=10, train_now=False, n_jobs=-1):

            super(EnsembleEvaluator, self).__init__(estimator, X_train, y_train,
                 X_test, y_test, cv, target_f, score_method, nsizes, train_now, n_jobs)

        def is_fitted(self, estimator=None):
            "check if the estimator is fitted"

            if estimator is None:
                estimator = self.estimator

            try:
                estimator.predict(self.X[:1])
                fitted = True
            except NotFittedError as e:
                fitted = False
            except Exception as e:
                fitted = False

            return fitted


        def get_component_classifiers(self):
            "return classifier and their names"

            all_clf = self.estimator.classifiers
            clf_names = list(self.estimator.named_classifiers.keys())

            return (all_clf, clf_names)

        def get_all_performance_metrics(self, metrics=None):
            "get metrics for all classifiers"

            (clf_list, clf_names) = self.get_component_classifiers()
            ensemble_classifier = _name_estimators([self.estimator])[0][0]

            for clf, label in zip(clf_list, clf_names):
                print("\n{}\n{}\n{}\n".format("*"*60, label, "*"*60))
                self.get_performance_metrics(metrics, clf)

            print("\n{}\n{}\n{}\n".format("*"*60, ensemble_classifier, "*"*60))
            self.get_performance_metrics(metrics, self.estimator)

        def get_all_gini(self, splitter=None):
            "get a list of gini of all classifiers"
            (clf_list, clf_names) = self.get_component_classifiers()
            ensemble_classifier = _name_estimators([self.estimator])[0][0]

            gini_list = []
            for clf, label in zip(clf_list, clf_names):
                gini = self.get_gini(splitter=splitter, estimator=clf)
                print("{}: {:.3f}".format(label, gini))
                gini_list.append((label, gini))

            gini = self.get_gini(splitter=splitter)
            print("{}: {:.3f}".format(ensemble_classifier, gini))
            gini_list.append((ensemble_classifier, gini))

            return (gini_list)

        # overloaded function
        def grid_search_validate(self, grids, score_method=None):
            "grid search and validate"
            if score_method is None:
                score_method = self.score_method

            # list of fitted classifiers
            fitted_classifiers = self.grid_search(grids, score_method)

            scores_list = []
            for label, clf in fitted_classifiers:
                scores = self.kfold_validate(estimator=clf)
                scores_list.append((label, scores))

            # combine classifiers after grid search
            clf_list = [clf for label, clf in fitted_classifiers]
            mvclf = en.MajorityVoteClassifier(classifiers=clf_list)
            ensemble_classifier = _name_estimators([mvclf])[0][0]

            scores = self.kfold_validate(estimator=mvclf)
            scores_list.append((ensemble_classifier, scores))

            return (scores_list)

        def get_optimized_classifier(self, grids, score_method=None):
            "combined classifiers after grid search"
            if score_method is None:
                score_method = self.score_method

            # list of fitted classifiers
            fitted_classifiers = self.grid_search(grids, score_method)
            clf_list = [clf for label, clf in fitted_classifiers]
            # TODO: other ensembler??
            mvclf = en.MajorityVoteClassifier(classifiers=clf_list)

            return (mvclf)

        # overloaded function
        def grid_search(self, grids, score_method=None):
            '''
                hyperparameter search for each classifier
                INPUTS:
                    grids : dictionary of params
                OUTPUTS:
                    list of fitted classifiers,
                    if no grid supplied, original classifier is given
            '''
            if score_method is None:
                score_method = self.score_method

            (clf_list, clf_names) = self.get_component_classifiers()
            ensemble_classifier = _name_estimators([self.estimator])[0][0]

            fitted_classifiers = []
            for clf, label in zip(clf_list, clf_names):
                # get grid
                try:
                    reg_grid = grids[label]
                    gs = GridSearchCV(estimator=clf,
                                      param_grid=reg_grid,
                                      scoring=score_method,
                                      cv=self.cv,
                                      n_jobs=self.cv)
                    gs = gs.fit(self.X, self.y)
                    print("Classifier: {}".format(label))
                    print("Best score ({}): {:.3f}".format(score_method, gs.best_score_))
                    print("Best params:")
                    print(gs.best_params_)
                except:
                    gs = clone(clf).fit(self.X, self.y)

                fitted_classifiers.append((label, gs))

            return (fitted_classifiers)

        def plot_all_roc_curve(self, splitter=None, num_splits=None):
            "plot ROC curves for all estimators"

            (clf_list, clf_names) = self.get_component_classifiers()
            ensemble_classifier = _name_estimators([self.estimator])[0][0]

            colors = ['black', 'orange', 'blue', 'green', 'yellow', 'magenta'] * 2
            linestyles = [':', '--', '-.', '-'] * 3
            plt.figure()
            for clf, label, clr, lsytle in zip(clf_list, clf_names, colors, linestyles):
                mean_auc, mean_tpr, mean_fpr = self.get_auc(splitter, num_splits, clf)

                legendString = "{}: Gini = {:.2f}".format(label, 2*mean_auc-1)
                plt.plot(mean_fpr, mean_tpr, color=clr, linestyle=lsytle,
                             label=legendString, lw=2)

            # plot the roc curve for ensemble
            mean_auc, mean_tpr, mean_fpr = self.get_auc(splitter, num_splits, self.estimator)
            ensemble_classifier = _name_estimators([self.estimator])[0][0]

            legendString = "{}: Gini = {:.2f}".format(ensemble_classifier, 2*mean_auc-1)
            plt.plot(mean_fpr, mean_tpr, color='red', linestyle='-',
                             label=legendString, lw=2)

            # random line
            plt.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=2)

            plt.xlim([-0.05, 1.05])
            plt.ylim([-0.05, 1.05])
            plt.xlabel('false positive rate')
            plt.ylabel('true positive rate')
            plt.title('Receiver Operator Characteristic')
            plt.legend(loc="lower right")
            plt.grid()
            plt.tight_layout()
