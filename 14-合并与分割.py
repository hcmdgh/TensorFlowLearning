from imports import *

# 1. concat（不创建新维度）
a = tf.constant([[1, 2], [3, 4]])
print(a)
''' [[1 2]
 [3 4]], shape=(2, 2), dtype=int32) '''

b = tf.constant([[5, 6]])
c = tf.concat([a, b], axis=0)
print(c)
''' tf.Tensor(
[[1 2]
 [3 4]
 [5 6]], shape=(3, 2), dtype=int32) '''

d = tf.constant([[7], [8]])
print(d)
''' tf.Tensor(
[[7]
 [8]], shape=(2, 1), dtype=int32) '''
e = tf.concat([a, d], axis=1)
print(e)
''' tf.Tensor(
[[1 2 7]
 [3 4 8]], shape=(2, 3), dtype=int32) '''

a = tf.zeros([2, 3, 4, 5])
b = tf.zeros([2, 9, 4, 5])
c = tf.concat([a, b], axis=1)
print(c.shape)
''' (2, 12, 4, 5) '''

# 2. stack（创建新维度）
# stack接收的tensor必须有相同的shape
a = tf.zeros([4, 35, 8])
b = tf.zeros([4, 35, 8])
c = tf.stack([a, b])
print(c.shape)
''' (2, 4, 35, 8) '''

c = tf.stack([a, b], axis=1)
print(c.shape)
''' (4, 2, 35, 8) '''

c = tf.stack([a, b], axis=-1)
print(c.shape)
''' (4, 35, 8, 2) '''

# 3. unstack
a = tf.zeros([2, 4, 35, 8])
b, c = tf.unstack(a)
print(b.shape, c.shape)
''' (4, 35, 8) (4, 35, 8) '''

b, c, d, e = tf.unstack(a, axis=1)
print(b.shape, c.shape, d.shape, e.shape)
''' (2, 35, 8) (2, 35, 8) (2, 35, 8) (2, 35, 8) '''

# 4. split
print(a.shape)
''' (2, 4, 35, 8) '''

b, c, d, e, f = tf.split(a, axis=2, num_or_size_splits=5)
print(b.shape, c.shape, d.shape)
''' (2, 4, 7, 8) (2, 4, 7, 8) (2, 4, 7, 8) '''

b, c, d = tf.split(a, axis=3, num_or_size_splits=[1, 3, 4])
print(b.shape, c.shape, d.shape)
''' (2, 4, 35, 1) (2, 4, 35, 3) (2, 4, 35, 4) '''
