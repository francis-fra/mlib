import numpy as np
import matplotlib.pyplot as plt
# true param to be estimated
coeff = [3, 4]

# uniform distribution from [0, 2]
X = 2 * np.random.rand(100, 1)
# vector form
# [1x2]*[2*100]
XX = np.c_[np.ones(100), X]
# (100, 2)
XX.shape
# get y data
yy = np.dot(XX, coeff) 
noise = np.random.randn(100, 1)
yy = yy + noise

# y = 4 + 3 * X + np.random.randn(100, 1)
yy.shape
yy = yy.reshape(-1, 1)
yy.shape
yy[:10]
y.shape
y[:10]

#------------------------------------------------------------   
# histogram
X.shape
cnts, bins = np.histogram(X)
bins, cnts
plt.hist(X)
plt.show()

plt.hist(yy)
plt.hist(y)
plt.show()

#------------------------------------------------------------   
plt.plot(X, y, "b.")
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.axis([0, 2, 0, 15])

plt.show()

#------------------------------------------------------------   
# linear model
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
X = 2 * np.random.rand(100, 1)

noise = np.random.rand(100, 1)
y = 4 + 3*X + noise

# scalar form
lin_reg.fit(X, y)
lin_reg.coef_, lin_reg.intercept_, 

# vector form
XX = np.c_[np.ones(100), X]
yy = np.dot(XX, coeff) + noise
lin_reg.fit(X, y)
lin_reg.coef_, lin_reg.intercept_, 

# TODO: error estimate
from sklearn.metrics import mean_absolute_error

y_pred = lin_reg.predict(X)
lin_mae = mean_absolute_error(y, y_pred)
lin_mae

#------------------------------------------------------------   
# SGD
from sklearn.linear_model import SGDRegressor
learning_rate = 0.1
max_iter = 50
sgd_reg = SGDRegressor(max_iter=max_iter, penalty=None, eta0=learning_rate)
sgd_reg.fit(X, y.ravel())
sgd_reg.coef_, sgd_reg.intercept_, 

y_pred = sgd_reg.predict(X)
lin_mae = mean_absolute_error(y, y_pred)
lin_mae