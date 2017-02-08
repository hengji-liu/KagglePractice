import numpy as np

M = np.array([[1, 2], [2, 4]])
r = np.linalg.matrix_rank(M, tol=None)
print(r)