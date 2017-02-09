import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

X_train = [[6], [8], [10], [14], [18]]
y_train = [[7], [9], [13], [17.5], [18]]

X_test = [[6], [8], [11], [16]]
y_test = [[8], [12], [15], [18]]

poly4 = PolynomialFeatures(degree=4)
X_train_poly4 = poly4.fit_transform(X_train)
X_test_poly4 = poly4.transform(X_test)

# normal Linear Regression
regressor_poly4 = LinearRegression()
regressor_poly4.fit(X_train_poly4, y_train)

# Lasso, L1 penalty, making coefficients towards 0
lasso_poly4 = Lasso()
lasso_poly4.fit(X_train_poly4, y_train)

print(regressor_poly4.score(X_test_poly4, y_test))
print(lasso_poly4.score(X_test_poly4, y_test))
print()
print(regressor_poly4.coef_)
print(lasso_poly4.coef_)

print()
print('---')
print()

# Ridge, L2 penalty, making coefficients closer to each other
ridge_poly4 = Ridge()
ridge_poly4.fit(X_train_poly4, y_train)

print(regressor_poly4.coef_)
print(np.sum([regressor_poly4.coef_ ** 2]))

# print(ridge_poly4.score(X_test_poly4, y_test))
print(ridge_poly4.coef_)
print(np.sum([ridge_poly4.coef_ ** 2]))
