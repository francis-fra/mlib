import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt

# data length
m = 100
# param: [0.5, 1, 2]
X = 6 * np.random.rand(m, 1) - 3
noise = np.random.randn(m, 1)
# polynomial
y = 0.5 * X**2 + X + 2 + noise 

# true param to be estimated
coeff = [0.5, 1, 2]
# tmp = X*X
# tmp[0]
# X[0] * X[0]
XX = np.c_[np.ones(m), X, X*X]
# (100, 3)
# XX.shape
yy = np.dot(XX, coeff)

#------------------------------------------------------------   
# linear model
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()

# fit the training data
lin_reg.fit(X, yy)
lin_reg.coef_, lin_reg.intercept_, 

# test with the test data
from sklearn.metrics import mean_absolute_error
# make a test data set
X_test = 6 * np.random.rand(m, 1) - 3
y_test = 0.5 * X_test**2 + X_test + 2 + np.random.randn(m, 1)

y_pred = lin_reg.predict(X_test)
lin_mae = mean_absolute_error(y_test, y_pred)
# 3.20
lin_mae

#------------------------------------------------------------   
X_new=np.linspace(-3, 3, 100).reshape(100, 1)
# predict Y using linear regression
y_new = lin_reg.predict(X_new)
plt.plot(X_test, y_test, "b.")
plt.plot(X_new, y_new, "r-", linewidth=2, label="Predictions")
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.legend(loc="upper left", fontsize=14)

plt.show()

#------------------------------------------------------------   
from sklearn.preprocessing import PolynomialFeatures

# degree is a param
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)

# fit after poly transformation
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)
lin_reg.coef_, lin_reg.intercept_


X_poly_test = poly_features.fit_transform(X_test)
y_pred = lin_reg.predict(X_poly_test)
lin_mae = mean_absolute_error(y_test, y_pred)
# 0.809
lin_mae

#------------------------------------------------------------   
# testing data set
#------------------------------------------------------------   
lower = -3
upper = 6
X_new=np.linspace(lower, upper, 100).reshape(100, 1)
# predict Y using linear regression
X_poly_new = poly_features.fit_transform(X_new)
y_new = lin_reg.predict(X_poly_new)
plt.plot(X_test, y_test, "b.")
plt.plot(X_new, y_new, "r-", linewidth=2, label="Predictions")
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.legend(loc="upper left", fontsize=14)

plt.show()

#------------------------------------------------------------   
# degress as a param
# overfitted is severe outside the range
lower = -3
upper = 6
X_new=np.linspace(lower, upper, 100).reshape(100, 1)
degrees = range(1, 8)
for deg in degrees:
    poly_features = PolynomialFeatures(degree=deg, include_bias=False)
    X_poly = poly_features.fit_transform(X_new)
    # fit after poly transformation
    lin_reg = LinearRegression()
    lin_reg.fit(X_poly, y)
    X_poly_test = poly_features.fit_transform(X_test)
    y_pred = lin_reg.predict(X_poly_test)
    lin_mae = mean_absolute_error(y_test, y_pred)
    print(lin_mae)

#------------------------------------------------------------   
# plot degree = 8
def plot_poly(deg=2):
    poly_features = PolynomialFeatures(degree=deg, include_bias=False)
    # train within [-3 3]
    X=np.linspace(-3, 8, 100).reshape(100, 1)
    y= 0.5 * X**2 + X+ 2 + np.random.randn(m, 1)
    # split into train and test data
    idx = np.where(X == 3)
    X[idx[0][0]:]
    X_train=X[:idx[0][0]]
    X_unseen=X[idx[0][0]:]
    y_train = y[:idx[0][0]]
    y_unseen= y[idx[0][0]:]
    # training
    X_poly = poly_features.fit_transform(X_train)
    # fit after poly transformation
    lin_reg = LinearRegression()
    lin_reg.fit(X_poly, y_train)
    # test within [3, 9]
    X_poly_test = poly_features.fit_transform(X_unseen)
    # predict Y using linear regression
    y_pred = lin_reg.predict(X_poly_test)
    # training data sets
    plt.plot(X, y, "b.")
    # predict unseen data
    plt.plot(X_unseen, y_pred, "r--", linewidth=2, label="Predictions")
    # fitted line
    X_poly= poly_features.fit_transform(X)
    y_full = lin_reg.predict(X_poly)
    plt.plot(X, y_full, "g-", linewidth=2, label="Fitted Line")
    plt.xlabel("$x_1$", fontsize=18)
    plt.ylabel("$y$", rotation=0, fontsize=18)
    plt.legend(loc="upper left", fontsize=14)
    plt.show()

plot_poly(2)
plot_poly(3)
plot_poly(4)
plot_poly(5)