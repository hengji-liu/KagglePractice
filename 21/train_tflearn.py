# from sklearn import datasets, metrics, preprocessing, model_selection
# import tensorflow.contrib.learn.python.learn as learn
#
# boston = datasets.load_boston()
#
# X, y = boston.data, boston.target
#
# X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25, random_state=33)
#
# scaler = preprocessing.StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.fit(X_test)
#
# feature_columns = learn.infer_real_valued_columns_from_input(X_train)
# tf_lr = learn.LinearRegressor(feature_columns=feature_columns)
# tf_lr.fit(X_train, y_train, steps=10000, batch_size=50)
#
# tf_lr_y_predict = tf_lr.predict(X_test)

# print(metrics.mean_absolute_error(tf_lr_y_predict, y_test))

import tensorflow.contrib.learn.python.learn as learn
from sklearn import datasets, metrics, preprocessing

boston = datasets.load_boston()
x = preprocessing.StandardScaler().fit_transform(boston.data)
feature_columns = learn.infer_real_valued_columns_from_input(x)
regressor = learn.LinearRegressor(feature_columns=feature_columns)
regressor.fit(x, boston.target, steps=200, batch_size=32)
boston_predictions = list(regressor.predict(x, as_iterable=True))
score = metrics.mean_squared_error(boston_predictions, boston.target)
print("MSE: %f" % score)
