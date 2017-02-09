import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

X_train = [[6], [8], [10], [14], [18]]
y_train = [[7], [9], [13], [17.5], [18]]

# linear, uniform sampling 100 points between [0, 25]
regressor = LinearRegression()
regressor.fit(X_train, y_train)
xx = np.linspace(0, 26, 100)
xx = xx.reshape(xx.shape[0], 1)
yy = regressor.predict(xx)

# poly 2
poly2 = PolynomialFeatures(degree=2)
X_train_poly2 = poly2.fit_transform(X_train)
regressor_poly2 = LinearRegression()
regressor_poly2.fit(X_train_poly2, y_train)
xx_poly2 = poly2.transform(xx)
yy_poly2 = regressor_poly2.predict(xx_poly2)

# poly 4
poly4 = PolynomialFeatures(degree=4)
X_train_poly4 = poly4.fit_transform(X_train)
regressor_poly4 = LinearRegression()
regressor_poly4.fit(X_train_poly4, y_train)
xx_poly4 = poly4.transform(xx)
yy_poly4 = regressor_poly4.predict(xx_poly4)

# plot
plt.scatter(X_train, y_train)

plt1, = plt.plot(xx, yy, label="Degree=1")
plt2, = plt.plot(xx, yy_poly2, label="Degree=2")
plt4, = plt.plot(xx, yy_poly4, label="Degree=4")

plt.axis([0, 25, 0, 25])
plt.xlabel('Diameter of Pizza')
plt.ylabel('Price of Pizza')
plt.legend(handles=[plt1, plt2, plt4])
plt.show()

print('The R-squared value of Linear Regressor performing on the training data is', regressor.score(X_train, y_train))

print('The R-squared value of Polynominal Regressor (Degree=2) performing on the training data is',
      regressor_poly2.score(X_train_poly2, y_train))

print('The R-squared value of Polynominal Regressor (Degree=4) performing on the training data is',
      regressor_poly4.score(X_train_poly4, y_train))

print()

# test
X_test = [[6], [8], [11], [16]]
y_test = [[8], [12], [15], [18]]

X_test_poly2 = poly2.transform(X_test)
X_test_poly4 = poly4.transform(X_test)

print(regressor.score(X_test, y_test))
print(regressor_poly2.score(X_test_poly2, y_test))
print(regressor_poly4.score(X_test_poly4, y_test))
