import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn import feature_selection
from sklearn.model_selection import cross_val_score
import pylab as pl

titanic = pd.read_csv('./data/titanic.txt')

y = titanic['survived']
X = titanic.drop(['row.names', 'name', 'survived'], axis=1)

# fill in missing data
X['age'].fillna(X['age'].mean(), inplace=True)
X.fillna('UNKONW', inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)

vec = DictVectorizer()
X_train = vec.fit_transform(X_train.to_dict(orient='record'))
X_test = vec.transform(X_test.to_dict(orient='record'))

print(len(vec.feature_names_))

# vanilla DT
dt = DecisionTreeClassifier(criterion='entropy')
dt.fit(X_train, y_train)
print(dt.score(X_test, y_test))

# select top 20% features
fs = feature_selection.SelectPercentile(feature_selection.chi2, percentile=20)

X_train_fs = fs.fit_transform(X_train, y_train)
X_test_fs = fs.transform(X_test)

dt.fit(X_train_fs, y_train)
print(dt.score(X_test_fs, y_test))

print()

# cross validation
percentiles = range(1, 100, 2)
results = []

for i in percentiles:
    fs = feature_selection.SelectPercentile(feature_selection.chi2, percentile=i)
    X_train_fs = fs.fit_transform(X_train, y_train)
    scores = cross_val_score(dt, X_train_fs, y_train, cv=5)
    results = np.append(results, scores.mean())

print(results)

opt = np.where(results == results.max())[0]
print('Optimal number of features %d' % percentiles[opt[0]])

# graph
pl.plot(percentiles, results)
pl.xlabel('percentiles of features')
pl.ylabel('accuracy')
pl.show()

# final model
fs = feature_selection.SelectPercentile(feature_selection.chi2, percentile=opt[0])
X_train_fs = fs.fit_transform(X_train, y_train)
dt.fit(X_train_fs, y_train)
X_test_fs = fs.transform(X_test)
print(dt.score(X_test_fs, y_test))
