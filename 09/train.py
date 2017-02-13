from sklearn.datasets import load_boston
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

boston = load_boston()
# print(boston.DESCR)

X = boston.data
y = boston.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)

# print('The max target value is', np.max(boston.target))
# print('The min target value is', np.min(boston.target))
# print('The average target value is', np.mean(boston.target))
# print()

ss_X = StandardScaler()
ss_y = StandardScaler()

# DeprecationWarning: Passing 1d arrays as data is deprecated in 0.17 and will raise ValueError in 0.19. Reshape your data either using X.reshape(-01, 01) if your data has a single feature or X.reshape(01, -01) if it contains a single sample.
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

X_train = ss_X.fit_transform(X_train)
X_test = ss_X.transform(X_test)
y_train = ss_y.fit_transform(y_train)
y_test = ss_y.transform(y_test)

# DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
y_train = np.ravel(y_train)

# uniform weights
uni_knr = KNeighborsRegressor(weights='uniform')
uni_knr.fit(X_train, y_train)
uni_knr_y_predict = uni_knr.predict(X_test)

# distance weights
dis_knr = KNeighborsRegressor(weights='distance')
dis_knr.fit(X_train, y_train)
dis_knr_y_predict = dis_knr.predict(X_test)

print('R-squared value of uniform-weighted KNeighborsRegression is', uni_knr.score(X_test, y_test))
print('The mean squared error of uniform-weighted KNeighborsRegression is',
      mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(uni_knr_y_predict)))
print('The mean absolute error of uniform-weighted KNeighborsRegression is',
      mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(uni_knr_y_predict)))

print()

print('R-squared value of distance-weighted KNeighborsRegression is', dis_knr.score(X_test, y_test))
print('The mean squared error of distance-weighted KNeighborsRegression is',
      mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(dis_knr_y_predict)))
print('The mean absolute error of distance-weighted KNeighborsRegression is',
      mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(dis_knr_y_predict)))
