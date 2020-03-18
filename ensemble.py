"""
Author: Francis Chan

Ensemble Classifier for binary classification models

"""
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.pipeline import _name_estimators
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone
from sklearn.externals import six
import numpy as np
import evaluate as ev
# from . import evaluate as ev
from sklearn.exceptions import NotFittedError


# majority vote classifier
class MajorityVoteClassifier(BaseEstimator, ClassifierMixin):
    """ A majority vote ensemble classifier

    Parameters
    ----------
    classifiers : array-like, shape = [n_classifiers]
      Different classifiers for the ensemble

    vote : str, {'classlabel', 'probability'} (default='label')
      If 'classlabel' the prediction is based on the argmax of
        class labels. Else if 'probability', the argmax of
        the sum of probabilities is used to predict the class label
        (recommended for calibrated classifiers).

    weights : array-like, shape = [n_classifiers], optional (default=None)
      If a list of `int` or `float` values are provided, the classifiers
      are weighted by importance; Uses uniform weights if `weights=None`.

    """
    def __init__(self, classifiers, vote='classlabel', weights=None):

        # list of classifiers
        self.classifiers = classifiers
        # dictionary of classifiers
        self.named_classifiers = {key: value for key, value
                                  in _name_estimators(classifiers)}
        self.vote = vote
        # TODO: optimize the weights??
        self.weights = weights
        self.fitted_classifiers = None
        # error checking
        self.assertion()

    def assertion(self):
        "check parameters"

        if self.vote not in ('probability', 'classlabel'):
            raise ValueError("vote must be 'probability' or 'classlabel'"
                             "; got (vote=%r)"
                             % self.vote)

        if self.weights and len(self.weights) != len(self.classifiers):
            raise ValueError('Number of classifiers and weights must be equal'
                             '; got %d weights, %d classifiers'
                             % (len(self.weights), len(self.classifiers)))
            
    def fit(self, X, y):
        """ Fit classifiers.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Matrix of training samples.

        y : array-like, shape = [n_samples]
            Vector of target class labels.

        Returns
        -------
        self : object

        """
        # Use LabelEncoder to ensure class labels start with 0, which
        # is important for np.argmax call in self.predict
        self.lablenc_ = LabelEncoder()
        # convert class label (it will revert the encoding when making prediction)
        self.lablenc_.fit(y)
        
        # list of all converted class labels
        self.classes_ = self.lablenc_.classes_
        self.fitted_classifiers = []
        
        # loop for each classifier and fit
        for clf in self.classifiers:
#             fitted_clf = clone(clf).fit(X, self.lablenc_.transform(y))
            fitted_clf = clf.fit(X, self.lablenc_.transform(y))
            self.fitted_classifiers.append(fitted_clf)
        return self

    def predict(self, X):
        """ Predict class labels for X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Matrix of training samples.

        Returns
        ----------
        maj_vote : array-like, shape = [n_samples]
            Predicted class labels.
            
        """
        # error checking
        if self.fitted_classifiers is None:
            raise NotFittedError("Classifiers must be fitted first")
        
        # voting methods 
        if self.vote == 'probability':
            # 'probability' vote (soft decision)
            # return the class label with the highest average (weighted) probability
            maj_vote = np.argmax(self.predict_proba(X), axis=1) 
        else:
            # 'classlabel' vote (hard decision)
            #  Collect results from clf.predict calls
            predictions = np.asarray([clf.predict(X) for clf in self.fitted_classifiers]).T

            # return the argmax of the prediction array
            maj_vote = np.apply_along_axis(
                                      lambda x:
                                      np.argmax(np.bincount(x,
                                                weights=self.weights)),
                                      axis=1,
                                      arr=predictions)
            
        # return voting outcomes (raw label)   
        maj_vote = self.lablenc_.inverse_transform(maj_vote)
        return maj_vote

    def predict_proba(self, X):
        """ Predict class probabilities for X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        ----------
        avg_proba : array-like, shape = [n_samples, n_classes]
            Weighted average probability for each class per sample.

        """
        # evaluate probabilities for each classifier (FIXME: must be fitted first!!)
        probas = np.asarray([clf.predict_proba(X) for clf in self.fitted_classifiers])
        # averaging (weighted) the probabilities
        avg_proba = np.average(probas, axis=0, weights=self.weights)
        return avg_proba

    def get_params(self, deep=True):
        """ Get classifier parameter names for GridSearch"""
        if not deep:
            return super(MajorityVoteClassifier, self).get_params(deep=False)
        else:
            out = self.named_classifiers.copy()
            # get parameters of all classifiers
            for name, step in six.iteritems(self.named_classifiers):
                for key, value in six.iteritems(step.get_params(deep=True)):
                    out['%s__%s' % (name, key)] = value
            return out
        
        
