from sklearn.datasets import load_boston
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
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

# DeprecationWarning: Passing 1d arrays as data is deprecated in 0.17 and will raise ValueError in 0.19. Reshape your data either using X.reshape(-1, 1) if your data has a single feature or X.reshape(1, -1) if it contains a single sample.
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

X_train = ss_X.fit_transform(X_train)
X_test = ss_X.transform(X_test)
y_train = ss_y.fit_transform(y_train)
y_test = ss_y.transform(y_test)

# DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
y_train = np.ravel(y_train)

# random forest regression
rfr = RandomForestRegressor()
rfr.fit(X_train, y_train)
rfr_y_predict = rfr.predict(X_test)

# extra tree regression
etr = ExtraTreesRegressor()
etr.fit(X_train, y_train)
etr_y_predict = etr.predict(X_test)

# gradient boosting regression
gbr = GradientBoostingRegressor()
gbr.fit(X_train, y_train)
gbr_y_predict = gbr.predict(X_test)

print('R-squared value of uniform-weighted RandomForestRegressor is', rfr.score(X_test, y_test))
print('The mean squared error of uniform-weighted RandomForestRegressor is',
      mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rfr_y_predict)))
print('The mean absolute error of uniform-weighted RandomForestRegressor is',
      mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rfr_y_predict)))

print()

print('R-squared value of uniform-weighted ExtraTreesRegressor is', etr.score(X_test, y_test))
print('The mean squared error of uniform-weighted ExtraTreesRegressor is',
      mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(etr_y_predict)))
print('The mean absolute error of uniform-weighted ExtraTreesRegressor is',
      mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(etr_y_predict)))
# feature importance
print(np.sort(list(zip(etr.feature_importances_, boston.feature_names)), axis=0))

print()

print('R-squared value of uniform-weighted GradientBoostingRegressor is', gbr.score(X_test, y_test))
print('The mean squared error of uniform-weighted GradientBoostingRegressor is',
      mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(gbr_y_predict)))
print('The mean absolute error of uniform-weighted GradientBoostingRegressor is',
      mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(gbr_y_predict)))
