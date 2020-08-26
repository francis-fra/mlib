import sys
srcDir = '/home/fra/FraDir/learn/Learnpy/Mypy/classifications'
sys.path.append(srcDir)

from decision_boundary import *


from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.datasets import make_moons
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from imp import reload
import decision_boundary
reload(decision_boundary)
from decision_boundary import *

#-----------------------------------------------------------------
# moon data
#-----------------------------------------------------------------
X, y = make_moons(n_samples=100, noise=0.15)

#-----------------------------------------------------------------
# iris data
#-----------------------------------------------------------------
iris = load_iris()
X = iris.data[:, 2:] # petal length and width
y = iris.target

#-----------------------------------------------------------------
# classifiers
#-----------------------------------------------------------------
tree_clf = DecisionTreeClassifier(max_depth=2, random_state=42)
tree_clf.fit(X, y)

deep_tree_clf2 = DecisionTreeClassifier(min_samples_leaf=4, random_state=42)
deep_tree_clf2.fit(X, y)


polynomial_svm_clf = Pipeline([
        ("poly_features", PolynomialFeatures(degree=3)),
        ("scaler", StandardScaler()),
        ("svm_clf", LinearSVC(C=10, loss="hinge", random_state=42))
    ])
polynomial_svm_clf.fit(X, y)


rf = RandomForestClassifier()
rf.fit(X, y)
bag = BaggingClassifier()
bag.fit(X, y)
knn3 = KNeighborsClassifier(3)
knn3.fit(X, y)    


#-----------------------------------------------------------------
# plot boundary
#-----------------------------------------------------------------
# plt.figure(figsize=(8, 4))
plot_decision_boundary(tree_clf, X, y)
plot_decision_boundary(tree_clf, X, y, newPlot=True)
plot_decision_boundary(deep_tree_clf2, X, y)
plot_decision_boundary(deep_tree_clf2, X, y, newPlot=False)
plot_decision_boundary(polynomial_svm_clf, X, y)
plt.figure()
plot_decision_boundary(bag, X, y)


X = StandardScaler().fit_transform(X)
plot_decision_contour(tree_clf, X, y)
plot_decision_contour(polynomial_svm_clf, X, y)
plot_decision_contour(rf, X, y)
plot_decision_contour(bag, X, y)
plot_decision_contour(knn3, X, y)
plot_decision_contour(knn3, X, y, newPlot=True)

plot_decision_contour(deep_tree_clf2, X, y)
plot_decision_contour(polynomial_svm_clf, X, y, False)
plot_decision_contour(polynomial_svm_clf, X, y)
plot_decision_contour(tree_clf, X, y)

# plot_decision_boundary(polynomial_svm_clf, X, y)
# plot_decision_boundary(polynomial_svm_clf, X, y, bg=False)


plot_decision_contour(polynomial_svm_clf, X, y)
# hasattr(polynomial_svm_clf, "decision_function")

#-------------------------------------------------------------------------
# regression
#-------------------------------------------------------------------------
from sklearn.tree import DecisionTreeRegressor        
np.random.seed(42)
m = 200
X = np.random.rand(m, 1)
y = 4 * (X - 0.5) ** 2
y = y + np.random.randn(m, 1) / 10

tree_reg = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg.fit(X, y)
fig = plt.figure()
plot_regression_predictions(tree_reg, X, y)

from sklearn.linear_model import LinearRegression   
lr = LinearRegression()
lr.fit(X, y)
plot_regression_predictions(lr, X, y)


from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)
lr.fit(X_poly, y)
plot_regression_predictions(lr, X, y, transformer=poly_features.transform)

# lr.predict(X_poly).shape
# lr.predict(X_poly)
# X.shape, y.shape

#-------------------------------------------------------------------------
# animation
#-------------------------------------------------------------------------
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier


from imp import reload
import decision_boundary
reload(decision_boundary)
from decision_boundary import *

# fig = plt.figure()
X, y = make_moons(n_samples=100, noise=0.15)
clf = DecisionTreeClassifier(splitter="random", max_leaf_nodes=16)
animate_boundary(X, y, clf)
animate_boundary(X, y)





