from sklearn import datasets
iris = datasets.load_iris()

# binary: iris-virginica
y = (iris["target"] == 2).astype(np.int) 
X = iris["data"]


#-------------------------------------------------- 
# modelling
# logistic regression
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()


#-------------------------------------------------- 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y)
log_reg.fit(X_train, y_train)

y_pred = log_reg.predict(X_test)

# evaluate
from sklearn.metrics import roc_auc_score

# gini
2 * roc_auc_score(y_test, y_pred) - 1
# training
2 * roc_auc_score(y_train, log_reg.predict(X_train)) - 1


#-------------------------------------------------- 
# SGD
from sklearn.linear_model import SGDClassifier
clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=5)
clf.fit(X_train, y_train)

# gini
2 * roc_auc_score(y_test, clf.predict(X_test)) - 1

#-------------------------------------------------- 
# ensemble
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.ensemble import VotingClassifier

rf = RandomForestClassifier()
gbm = GradientBoostingClassifier()
rf.fit(X_train, y_train)
gbm.fit(X_train, y_train)

2 * roc_auc_score(y_test, rf.predict(X_test)) - 1
2 * roc_auc_score(y_test, gbm.predict(X_test)) - 1

estimators = [
    ('rf', RandomForestClassifier()),
    ('log_reg', LogisticRegression()),
    ('gbm', GradientBoostingClassifier()),
    ('sgd', SGDClassifier(loss="hinge", penalty="l2", max_iter=5))
]

sk = StackingClassifier(estimators=estimators)
sk.fit(X_train, y_train)
2 * roc_auc_score(y_test, sk.predict(X_test)) - 1

vc = VotingClassifier(estimators=estimators)
vc.fit(X_train, y_train)
2 * roc_auc_score(y_test, vc.predict(X_test)) - 1
