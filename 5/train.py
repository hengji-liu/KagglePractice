import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

titanic = pd.read_csv('./data/titanic.txt')

# print(titanic.head())
# titanic.info()

# X = titanic[['pclass', 'age', 'sex']]
# http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy

X = titanic.loc[:, ('pclass', 'age', 'sex')]
y = titanic['survived']
# X.info()
# only 633 out of 1313 entries have attri age
X['age'].fillna(X['age'].mean(), inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)
vec = DictVectorizer(sparse=False)
X_train = vec.fit_transform(X_train.to_dict(orient='record'))
# one-hot encoding
# print(vec.feature_names_)
# ['age', 'pclass=1st', 'pclass=2nd', 'pclass=3rd', 'sex=female', 'sex=male']
X_test = vec.transform(X_test.to_dict(orient='record'))

dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
y_predict = dtc.predict(X_test)
print(dtc.score(X_test, y_test))
print(classification_report(y_test, y_predict, target_names=['died', 'survived']))
