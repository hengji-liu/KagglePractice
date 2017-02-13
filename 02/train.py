from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

digits = load_digits()
# print(digits.data.shape)
# 1797*64

X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=33)
# print(y_train.shape)
# print(y_test.shape)
# 1347
# 450

ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

lsvc = LinearSVC()
lsvc.fit(X_train, y_train)
y_predict = lsvc.predict(X_test)

print('The accuracy of Linear SVC is ', lsvc.score(X_test, y_test))
print(classification_report(y_test, y_predict, target_names=digits.target_names.astype(str)))
# Note that the metrics in the report take 01 classification as positive
# and all others as negative.
# Hence, there are 10 binary classification tasks and corresponding metrics.
