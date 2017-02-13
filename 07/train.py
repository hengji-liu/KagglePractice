from sklearn.datasets import load_boston
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

boston = load_boston()
# print(boston.DESCR)

X = boston.data
y = boston.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)

print('The max target value is', np.max(boston.target))
print('The min target value is', np.min(boston.target))
print('The average target value is', np.mean(boston.target))
print()

ss_X = StandardScaler()
ss_y = StandardScaler()

# DeprecationWarning: Passing 1d arrays as data is deprecated in 0.17 and will raise ValueError in 0.19. Reshape your data either using X.reshape(-1, 1) if your data has a single feature or X.reshape(1, -1) if it contains a single sample.
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

X_train = ss_X.fit_transform(X_train)
X_test = ss_X.transform(X_test)
y_train = ss_y.fit_transform(y_train)
y_test = ss_y.transform(y_test)

lr = LinearRegression()
lr.fit(X_train, y_train)
lr_y_predict = lr.predict(X_test)

# DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
y_train = np.ravel(y_train)

sgdr = SGDRegressor()
sgdr.fit(X_train, y_train)
sgdr_y_predict = sgdr.predict(X_test)

# score() returns R^2 score, it reflects the percentage that regression fluctuation tells the true value
print('The value of default measurement of LinearRegression is', lr.score(X_test, y_test))
print('The value of R-squared of LinearRegression is', r2_score(y_test, lr_y_predict))
print('The mean squared error of LinearRegression is',
      mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(lr_y_predict)))
print('The mean absolute error of LinearRegression is',
      mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(lr_y_predict)))

print()

print('The value of default measurement of SGDRegressor is', sgdr.score(X_test, y_test))
print('The value of R-squared of SGDRegressor is', r2_score(y_test, sgdr_y_predict))
print('The mean squared error of SGDRegressor is',
      mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(sgdr_y_predict)))
print('The mean absolute error of SGDRegressor is',
      mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(sgdr_y_predict)))
