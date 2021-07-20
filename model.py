import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix
from sklearn.metrics import auc
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import cross_val_score, GridSearchCV
from abc import ABC, abstractmethod 

import utility as util
import plot

class ClassificationEvaluator(ABC, BaseEstimator):
    """Base Classification Evaluator 
    """

    @abstractmethod
    def performance_summary(self, X, y):
        pass

    @abstractmethod
    def get_lift(self):
        pass

    @abstractmethod
    def get_roc_curve(self):
        pass

    # @abstractmethod
    # def plot_roc_curve(self):
    #     pass

class BinaryClassificationEvaluator(ClassificationEvaluator):

    def __init__(self, classifier):
        self.classifier = classifier

    def _get_single_class_lift(self, y_true, y_pred, pos_label=1, bypct=True, cumulative=False):
        "called by get lift"
        scoredf = pd.DataFrame({'pred': y_pred, 'true': y_true == pos_label})
        scoredf = util.sort_and_rank(scoredf, col='pred', bypct=bypct)
        stat = scoredf.groupby('rank')['true'].agg(['sum', 'count'])

        base_rate = sum(y_true == pos_label) / len(y_true)
        if cumulative == True:
            stat = stat.cumsum()

        lift = stat['sum'] /  (stat['count'] * base_rate)
        return lift
        
    def get_lift(self, y_true, y_proba, bypct=True, cumulative=False, pos_label=1):
        """out of sample lift

            Parameters
            ----------
            y_true : true target (numeric)
            y_proba : predicted target (numeric)
            bypct : boolean (show percentile if true, show decile otherwise)
            cumulative : boolean (show cumulative lift if true)
            pos_label : encoded number for target
        
        """
        # pos_label=1

        return self.__get_single_class_lift(y_true, y_proba[:,pos_label], pos_label, bypct, cumulative)

    def get_roc_curve(self, X_test, y_test, X_train=None, y_train=None):
        """fitted and test ROC curve

            Parameters
            ----------
            X_test :  testing feature data frame 
            y_test :  testing label column
            X_train : training feature data frame
            y_train : training label column
        
        """

        probas = self.classifier.predict_proba(X_test)
        fpr, tpr, thresholds = roc_curve(y_test, probas[:, 1], pos_label=1)

        if (X_train is not None and y_train is not None):
            fitted_probas = self.classifier.predict_proba(X_train)
            fitted_fpr, fitted_tpr, fitted_thresholds = roc_curve(y_train, fitted_probas[:, 1], pos_label=1)
            return (fpr, tpr, thresholds, fitted_fpr, fitted_tpr, fitted_thresholds)
        else:
            return (fpr, tpr, thresholds, None, None, None)

    def plot_confusion_matrix(self, X, y, mapping):
        """plot confusion matrix
        
            Parameters
            ----------
            X : feature data set
            y :  target label column
            mapping : target encode map
        """
        predictions = self.classifier.predict(X)
        # change the matrix in reverse order for plotting
        confmat = confusion_matrix(y_true=y, y_pred=predictions)
        # the labels are in reverse order too
        mapping = util.sort_dict(mapping, sort_key = lambda x: -x[1])
        plot.plot_confusion_matrix(np.flip(confmat), list(mapping.keys()))

    def plot_roc_curve(self, X_test, y_test, label_test='test', X_train=None, y_train=None, label_train='train'):
        """plot roc curves

            Parameters:
            -----------

            X_test : feature data set
            y_test : true target label
            X_train: secondary data set (optinal)
            y_train: secondary target label (optinal)
        
        """
        (fpr, tpr, thresholds, fitted_fpr, fitted_tpr, fitted_thresholds) = self.get_roc_curve(X_test, y_test, X_train, y_train)
        plotdata = {
            'fpr': fpr,
            'tpr': tpr,
            'primary': label_test,
            'fitted_fpr': fitted_fpr,
            'fitted_tpr': fitted_tpr,
            'secondary': label_train
        }
        plot.plot_roc_curve(plotdata)

    def plot_top_drivers(self, var_names ,limit=20):
        """plot top drivers
        
            Parameters:
            -----------
            var_names : list of feature names
            limit : max number of features to plot
        
        """
        key_drivers = self.classifier.key_drivers(var_names)
        plot.plot_drivers(key_drivers.iloc[:limit,:])

    # TODO:
    # def plot_learning_curve(self):
    #     pass

    # TODO:
    # def plot_lift_chart(self):
    #     pass

    def performance_summary(self, X, y):
        "out of sample performance summary"

        predictions = self.classifier.predict(X)
        pred_prob = self.classifier.predict_proba(X)

        summary = {}
        summary['Accuracy'] = accuracy_score(y, predictions)
        summary['Consfusion Matrix'] = confusion_matrix(y_true=y, y_pred=predictions)
        summary['AUC'] = roc_auc_score(y, pred_prob[:,1])
        summary['Gini'] = 2 * summary['AUC'] - 1
        return summary

class MultiClassificationEvaluator(BinaryClassificationEvaluator):

    def __init__(self, estimator):
        super().__init__(estimator)

    def get_lift(self, y_true, y_proba, bypct=True, cumulative=False):
        """out of sample lift
        
            Parameters
            ----------
            y_true : true target (numeric)
            y_proba : predicted target (numeric)
            bypct : boolean (show percentile if true, show decile otherwise)
            cumulative : boolean (show cumulative lift if true)
        """
        # lift for each class
        n_classes = len(np.unique(y_true))
        result = []
        for idx in range(n_classes):
            df = super()._get_single_class_lift(y_true, y_proba[:,idx], idx , bypct, cumulative)
            result.append(df)

        return pd.concat(result, axis=1)

    def plot_roc_curve(self, X, y, mapping):
        "plot ROC curve"
        (fpr, tpr) = self.get_roc_curve(X, y)
        plotData = {
            'fpr': fpr,
            'tpr': tpr,
            'label_map': util.reverse_dict(mapping),
        }
        plot.plot_multiclass_roc_curve(plotData)

    def get_gini(self, X, y, mapping=None):
        """get gini for each class

            multiclass method
        """
        result = self.get_auc(X, y)
        for k, v in result.items():
            result[k] = 2*v - 1
        return result

    def get_auc(self, X, y, mapping=None):
        """get auc for each class
        
            multiclass method 
        """
        (fpr, tpr) = self.get_roc_curve(X, y)

        result = dict()
        if mapping is not None:
            for label, idx in mapping.items():
                result[label] = auc(fpr[idx], tpr[idx])
        else:
            n_classes = len(np.unique(y))
            for idx in range(n_classes):
                result[idx] = auc(fpr[idx], tpr[idx])

        return result

    def get_roc_curve(self, X, y):
        """plot ROC curve for each class

        """
        fpr = dict()
        tpr = dict()
        n_classes = len(np.unique(y))
        proba = self.classifier.predict_proba(X)

        for idx in range(n_classes):
            fpr[idx], tpr[idx], _ = roc_curve(y, proba[:, idx], pos_label=idx)

        return (fpr, tpr)

    def performance_summary(self, X, y, mapping=None):
        """out of sample performance summary

            Parameters
            ----------
            X : feature data set
            y : target label (numeric)
            mapping : lable encode map (optional)
        
        """

        predictions = self.classifier.predict(X)
        pred_prob = self.classifier.predict_proba(X)

        summary = {}
        summary['Accuracy'] = accuracy_score(y, predictions)
        summary['Consfusion Matrix'] = confusion_matrix(y_true=y, y_pred=predictions)
        summary['AUC'] = self.get_auc(X, y, mapping)
        summary['GINI'] = self.get_gini(X, y, mapping)
        # summary['AUC'] = roc_auc_score(y, pred_prob, multi_class=roc_multi_class)
        # summary['Gini'] = 2 * summary['AUC'] - 1

        return summary

#---------------------------------------------------------------------
# standard classfier
#---------------------------------------------------------------------
class Classifier(ABC, BaseEstimator):

    @abstractmethod
    def set_params(self, **params):
        "set parameters"
        pass

    @abstractmethod
    def fit(self, X, y):
        "train"
        pass

    @abstractmethod
    def predict_proba(self, X):
        "soft decision predict"
        pass

    @abstractmethod
    def predict(self, X):
        "hard decision predict"
        pass

    @abstractmethod
    def key_drivers(self):
        "get key drivers"
        pass

    # similar to predict_proba but with the customer id 
    # @abstractmethod
    # def score(self, X):
    #     "score from fitted estimator"
    #     pass

    @abstractmethod
    def tune(self, X, y, param_grid, cv=3, metric="roc_auc"):
        "do grid search"
        pass

    @abstractmethod
    def cross_validate(self, X, y, cv=3, metric="roc_auc"):
        "cross validation"
        pass


#---------------------------------------------------------------------
# binary classifier
#---------------------------------------------------------------------
class BinaryClassifier(Classifier):
    """Standardized Binary Classifier
        potentially provide the same interface for all classifier
        including deep learning
    """
    def __init__(self, estimator):
        self._estimator = estimator
        self._gs = None

    @property
    def estimator(self):
        return self._estimator

    @estimator.setter
    def estimator(self, estimator):
        self._estimator = estimator

    def set_params(self, **params):
        "set parameters"
        self.estimator.set_params(params)
        return self

    def fit(self, X, y):
        self.estimator.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.estimator.predict_proba(X) 
        # return self

    def predict(self, X):
        return self.estimator.predict(X) 
        # return self

    def key_drivers(self, var_names):
        "get key drivers"
        if hasattr(self.estimator, 'feature_importances_'):
            drivers = self.estimator.feature_importances_
        elif hasattr(self.estimator, 'coef_'):
            # LGR / SVM
            drivers = self.estimator.coef_
        else:
            drivers = [0] * len(var_names)
        data = {'name': var_names, 'value': drivers}
        df = pd.DataFrame(data)
        return df.sort_values(by='value', ignore_index=True, ascending=False)

    def tune(self, X, y, param_grid, **kwargs):
        "do grid search and assign best estimator"
        gs = GridSearchCV(estimator=self.estimator, param_grid=param_grid, **kwargs)

        gs = gs.fit(X, y)
        print("Best params:")
        print(gs.best_params_)
        self.gs = gs
        # by assigning, the estimator is not fitted
        self.estimator = gs.best_estimator_
        return self

    def cross_validate(self, X, y, **kwargs):
        "cross validation"
        scores = cross_val_score(estimator=self.estimator, X=X, y=y, **kwargs)

        # self._print_title(estimator_name)
        print("{:d}-Fold score: {:.3f} +/- {:.3f}".format(len(scores), np.mean(scores), np.std(scores)))
        return scores

#---------------------------------------------------------------------
# multiclass classifier
#---------------------------------------------------------------------
class MultiClassifier(BinaryClassifier):
    """Standardized Multiclass Classifier
        potentially provide the same interface for all classifier
        including deep learning
    """
    def __init__(self, estimator):
        super().__init__(estimator)



#---------------------------------------------------------------------
# Custom classifiers
#---------------------------------------------------------------------
class RandomClassifier(BaseEstimator):
    """
        Parameters
        ----------
        prop : base rate
    """
    def __init__(self, prop):
        self.prop = prop
    def fit(self, X, y=None):
        pass
    def predict(self, X):
        nums = np.zeros((len(X), 1), dtype=bool)
        N = np.floor(len(X) * self.prop)
        nums[:N] = True
        np.random.shuffle(nums)
        return nums

class NeverClassifier(BaseEstimator):
    def fit(self, X, y=None):
        pass
    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)