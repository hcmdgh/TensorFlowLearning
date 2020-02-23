from imports import *

# 1. shape、ndim
a = tf.random.normal([4, 28, 28, 3])
print(a.shape, a.ndim)
''' (4, 28, 28, 3) 4 '''

# 2. reshape
b = tf.reshape(a, [4, -1, 3])
print(b.shape)
''' (4, 784, 3) '''

c = tf.reshape(a, [4, -1])
print(c.shape)
''' (4, 2352) '''

# 3. transpose
# reshape改变的是view，而transpose改变content
a = tf.random.normal([4, 3, 2, 1])
b = tf.transpose(a)
print(b.shape)
''' (1, 2, 3, 4) '''

# 将末2个维度转置
c = tf.transpose(a, perm=[0, 1, 3, 2])
print(c.shape)
''' (4, 3, 1, 2) '''

# 4. expand_dims
# data: [classes, students, subjects]
data = tf.ones([4, 35, 8])

# 4.1 增加“学校”维度
a = tf.expand_dims(data, axis=0)
print(a.shape)
''' (1, 4, 35, 8) '''

# 4.2 axis为正数，在轴前面增加维度
b = tf.expand_dims(data, axis=1)
print(b.shape)
''' (4, 1, 35, 8) '''

# 4.3 axis为负数，在轴后面增加维度
c = tf.expand_dims(data, axis=-1)
print(c.shape)
''' (4, 35, 8, 1) '''

# 5. squeeze
# Only squeeze for shape=1 dim
a = tf.zeros([1, 2, 1, 1, 3])
b = tf.squeeze(a)
print(b.shape)
''' (2, 3) '''

c = tf.squeeze(a, axis=0)
print(c.shape)
''' (2, 1, 1, 3) '''