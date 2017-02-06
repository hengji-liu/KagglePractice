import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ars

digits_train = pd.read_csv('./data/optdigits.tra', header=None)
digits_test = pd.read_csv('./data/optdigits.tes', header=None)

# 0-63 features, 64 target
X_train = digits_train[np.arange(64)]
y_train = digits_train[64]

X_test = digits_test[np.arange(64)]
y_test = digits_test[64]

kmeans = KMeans(n_clusters=10)
kmeans.fit(X_train)
y_pred = kmeans.predict(X_test)

print(ars(y_test, y_pred))
