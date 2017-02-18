import tensorflow as tf

# 1*2 row vec
matrix1 = tf.constant([[3., 3.]])

# 2*1 col vec
matrix2 = tf.constant([[2.], [2.]])

product = tf.matmul(matrix1, matrix2)

linear = tf.add(product, tf.constant(2.0))

with tf.Session() as sess:
    result = sess.run(linear)
    print(result)