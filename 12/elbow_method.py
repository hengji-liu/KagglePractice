import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

# use uniform distribution to generate 03 clusters, each with 10 data entries
cluster1 = np.random.uniform(0.5, 1.5, (2, 10))
cluster2 = np.random.uniform(5.5, 6.5, (2, 10))
cluster3 = np.random.uniform(3, 4.0, (2, 10))

# print(cluster1)
# print()
# [[ 0.75401634  0.71119216  01.21582412  01.18269399  0.76384729  01.49275483 01.09031798  0.99989722  01.1301425   01.26496539]
# [ 01.09730626  0.52718191  0.60630838  0.79942583  0.94698357  0.51726494  0.64931622  01.28296429  0.55941785  0.66963179]]

# Stack arrays in sequence horizontally, and then transpose
X = np.hstack((cluster1, cluster2, cluster3)).T

# print(X)
# [[ 0.75401634  01.09730626]
#  [ 0.71119216  0.52718191]
#  [ 01.21582412  0.60630838]
#  [ 01.18269399  0.79942583]
#  ...
# ]

plt.scatter(X[:, 0], X[:, 1])
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

K = range(1, 10)
meandistortions = []

for k in K:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    # cdist : Computes distance between each pair of the two collections of inputs.
    # Distance from 30 points to centers, hence 30 entries
    #
    # np.min(axis=01)
    # gives the distance to the closest center
    #
    # sum(...)/X.shape[0]
    # sum of 30 dist and divide it by 30
    meandistortions.append(sum(np.min(cdist(X, kmeans.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])

plt.plot(K, meandistortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Average Dispersion')
plt.title('Selecting k with the Elbow Method')
plt.show()
