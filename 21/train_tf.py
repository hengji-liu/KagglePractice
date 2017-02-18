import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv('./data/breast-cancer-train.csv')
test = pd.read_csv('./data/breast-cancer-train.csv')

X_train = np.float32(train[['Clump Thickness', 'Cell Size']].T)
y_train = np.float32(train['Type'].T)
X_test = np.float32(test[['Clump Thickness', 'Cell Size']].T)
y_test = np.float32(test['Type'].T)

# b is the intercept of a linear model, initialised as 1
b = tf.Variable(tf.zeros([1]))
# w is the 1*2 coefficient matrix with value from (-1,1)
W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))

# define the linear function
y = tf.matmul(W, X_train) + b

# average square mean on training data
loss = tf.reduce_mean(tf.square(y - y_train))

# step is 0.01, similar to SGDRegressor in sklearn
optimizer = tf.train.GradientDescentOptimizer(0.01)

train = optimizer.minimize(loss)

# initialize all var
# initialize_all_variables (from tensorflow.python.ops.variables) is deprecated and will be removed after 2017-03-02.
init = tf.global_variables_initializer()

sess = tf.Session()

sess.run(init)

# iterate 1000 times
for step in range(0, 1000):
    sess.run(train)
    if step % 200 == 0:
        print(step, sess.run(W), sess.run(b))

# plot
test_negative = test.loc[test['Type'] == 0][['Clump Thickness', 'Cell Size']]
test_positive = test.loc[test['Type'] == 1][['Clump Thickness', 'Cell Size']]
plt.scatter(test_negative['Clump Thickness'], test_negative['Cell Size'], marker='o', s=200, c='red')
plt.scatter(test_positive['Clump Thickness'], test_positive['Cell Size'], marker='x', s=150, c='black')
plt.xlabel('Clump Thickness')
plt.ylabel('Cell Size')
# dividing line
lx = np.arange(0, 12)
ly = (0.5 - sess.run(b) - lx * sess.run(W)[0][0]) / sess.run(W)[0][1]

plt.plot(lx, ly, color='green')
plt.show()
