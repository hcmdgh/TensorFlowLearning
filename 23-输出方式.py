from imports import *

# 1. sigmoid
# 公式见：assets/sigmoid.png
# 将数据缩放到(0, 1)范围内
a = tf.linspace(-6., 6, 10)
print(a)
''' tf.Tensor(
[-6.        -4.6666665 -3.3333333 -2.        -0.6666665  0.666667
  2.         3.333334   4.666667   6.       ], shape=(10,), dtype=float32) '''

b = tf.sigmoid(a)
print(b)
''' tf.Tensor(
[0.00247262 0.00931596 0.0344452  0.11920292 0.33924368 0.6607564
 0.880797   0.96555483 0.99068403 0.9975274 ], shape=(10,), dtype=float32) '''

# 2. softmax
# 公式见：assets/softmax.png
# 将数据缩放到[0, 1]范围内，且和为1
a = tf.constant([2, 5, 3], dtype=tf.float32)
b = tf.nn.softmax(a)
print(b)
''' tf.Tensor([0.04201007 0.8437947  0.1141952 ], shape=(3,), dtype=float32) '''

print(tf.reduce_sum(b))
''' tf.Tensor(1.0, shape=(), dtype=float32) '''

# 3. tanh
# 公式见：assets/tanh.png
# 将数据缩放到(-1, 1)范围内
a = tf.random.uniform(minval=-10, maxval=10, shape=[8])
print(a)
''' tf.Tensor(
[-9.661379  -6.960578   2.2584896  9.342945   2.4876194  6.610443
  8.635563   5.382082 ], shape=(8,), dtype=float32) '''

b = tf.tanh(a)
print(b)
''' tf.Tensor(
[-1.         -0.9999982   0.97839206  1.          0.98628104  0.99999636
  0.99999994  0.99995774], shape=(8,), dtype=float32) '''
