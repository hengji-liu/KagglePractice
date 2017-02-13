import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

titanic = pd.read_csv('./data/titanic.txt')
X = titanic[['pclass', 'age', 'sex']]
y = titanic['survived']
X['age'].fillna(X['age'].mean(), inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)
vec = DictVectorizer(sparse=False)
X_train = vec.fit_transform(X_train.to_dict(orient='record'))
X_test = vec.transform(X_test.to_dict(orient='record'))

rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)

xgbc = XGBClassifier()
xgbc.fit(X_train, y_train)

print('The accuracy of Random Forest Classifier on testing set: ', rfc.score(X_test, y_test))
print('The accuracy of eXtreme Gradient Boosting Classifier on testing set: ', xgbc.score(X_test, y_test))
