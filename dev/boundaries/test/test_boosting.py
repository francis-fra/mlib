np.random.seed(42)
X = np.random.rand(100, 1) - 0.5
y = 3*X[:, 0]**2 + 0.05 * np.random.randn(100)

from sklearn.tree import DecisionTreeRegressor

# first tree
tree_reg1 = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg1.fit(X, y)

# residual (error)
y2 = y - tree_reg1.predict(X)
# fit residual error
tree_reg2 = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg2.fit(X, y2)

# residual error of second tree
y3 = y2 - tree_reg2.predict(X)
# fit
tree_reg3 = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg3.fit(X, y3)

#-----------------------------------------------------------------
# show figure
#-----------------------------------------------------------------
def plot_predictions(regressors, X, y, axes, label=None, style="r-", data_style="b.", data_label=None):
    x1 = np.linspace(axes[0], axes[1], 500)
    # sum of predicted values
    y_pred = sum(regressor.predict(x1.reshape(-1, 1)) for regressor in regressors)
    plt.plot(X[:, 0], y, data_style, label=data_label)
    plt.plot(x1, y_pred, style, linewidth=2, label=label)
    if label or data_label:
        plt.legend(loc="upper center", fontsize=16)
    plt.axis(axes)



plt.figure(figsize=(11,11))

plt.subplot(321)
plot_predictions([tree_reg1], X, y, axes=[-0.5, 0.5, -0.1, 0.8], label="$h_1(x_1)$", style="g-", data_label="Training set")
plt.ylabel("$y$", fontsize=16, rotation=0)
plt.title("Residuals and tree predictions", fontsize=16)

plt.subplot(322)
plot_predictions([tree_reg1], X, y, axes=[-0.5, 0.5, -0.1, 0.8], label="$h(x_1) = h_1(x_1)$", data_label="Training set")
plt.ylabel("$y$", fontsize=16, rotation=0)
plt.title("Ensemble predictions", fontsize=16)

plt.subplot(323)
plot_predictions([tree_reg2], X, y2, axes=[-0.5, 0.5, -0.5, 0.5], label="$h_2(x_1)$", style="g-", data_style="k+", data_label="Residuals")
plt.ylabel("$y - h_1(x_1)$", fontsize=16)

plt.subplot(324)
plot_predictions([tree_reg1, tree_reg2], X, y, axes=[-0.5, 0.5, -0.1, 0.8], label="$h(x_1) = h_1(x_1) + h_2(x_1)$")
plt.ylabel("$y$", fontsize=16, rotation=0)

plt.subplot(325)
plot_predictions([tree_reg3], X, y3, axes=[-0.5, 0.5, -0.5, 0.5], label="$h_3(x_1)$", style="g-", data_style="k+")
plt.ylabel("$y - h_1(x_1) - h_2(x_1)$", fontsize=16)
plt.xlabel("$x_1$", fontsize=16)

plt.subplot(326)
plot_predictions([tree_reg1, tree_reg2, tree_reg3], X, y, axes=[-0.5, 0.5, -0.1, 0.8], label="$h(x_1) = h_1(x_1) + h_2(x_1) + h_3(x_1)$")
plt.xlabel("$x_1$", fontsize=16)
plt.ylabel("$y$", fontsize=16, rotation=0)



#-----------------------------------------------------------------
# trial
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=100, noise=0.15)

from sklearn.tree import DecisionTreeClassifier
tree_clf1 = DecisionTreeClassifier(max_depth=3)

tree_clf1.fit(X, y)

# residual (error)
error = 1 * (y == tree_clf1.predict(X))
error
count_errors(tree_clf1.predict(X), y)
# 0.079

# second classifier
tree_clf2 = DecisionTreeRegressor(max_depth=3)
tree_clf2.fit(X, error)

# residual (error)
error2 = 1 * (error == tree_clf2.predict(X))
error2


# third classifier
tree_clf3 = DecisionTreeRegressor(max_depth=3)
tree_clf3.fit(X, error2)

# residual (error)
error3 = 1 * (error2 == tree_clf3.predict(X))
error3

classifiers = [tree_clf1, tree_clf2, tree_clf3]
results = sum(clf.predict(X) for clf in classifiers)
results

final_results = 1 * (results > 2)
1 * (y == final_results)

count_errors(final_results, y)
# 0.10999999999999999


tree_clf1.predict(X)
tree_clf2.predict(X)
tree_clf3.predict(X)


def count_errors(pred, y):
    result = 1 * (y == pred)
    N = len(y)
    return (1 - sum(result)/N)