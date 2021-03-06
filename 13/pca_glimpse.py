import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

digits_train = pd.read_csv('./data/optdigits.tra', header=None)
# digits_test = pd.read_csv('./data/optdigits.tes', header=None)
X_digits = digits_train[np.arange(64)]
y_digits = digits_train[64]

estimator = PCA(n_components=2)
X_pca = estimator.fit_transform(X_digits)


def plot_pca_scatter():
    colors = ['black', 'blue', 'purple', 'yellow', 'white', 'red', 'lime', 'cyan', 'orange', 'gray']
    for i in range(len(colors)):
        px = X_pca[:, 0][y_digits.as_matrix() == i]
        py = X_pca[:, 1][y_digits.as_matrix() == i]
        plt.scatter(px, py, c=colors[i])

    plt.legend(np.arange(0, 10).astype(str))
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.show()


plot_pca_scatter()
