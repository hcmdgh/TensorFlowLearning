from imports import *

# 1. pad
# 往一个矩阵的四周（周围）填充
a = tf.reshape(tf.range(9), [3, 3])
print(a)
''' tf.Tensor(
[[0 1 2]
 [3 4 5]
 [6 7 8]], shape=(3, 3), dtype=int32) '''

# 1.1 在上方填充1行，下方填充2行
b = tf.pad(a, [[1, 2], [0, 0]])
print(b)
''' tf.Tensor(
[[0 0 0]
 [0 1 2]
 [3 4 5]
 [6 7 8]
 [0 0 0]
 [0 0 0]], shape=(6, 3), dtype=int32) '''

# 1.2 在左边填充1列，右边填充2列
c = tf.pad(a, [[0, 0], [1, 2]])
print(c)
''' tf.Tensor(
[[0 0 1 2 0 0]
 [0 3 4 5 0 0]
 [0 6 7 8 0 0]], shape=(3, 6), dtype=int32) '''

# 1.3 在四周填充一圈
d = tf.pad(a, [[1, 1], [1, 1]])
print(d)
''' tf.Tensor(
[[0 0 0 0 0]
 [0 0 1 2 0]
 [0 3 4 5 0]
 [0 6 7 8 0]
 [0 0 0 0 0]], shape=(5, 5), dtype=int32) '''

# 1.4 填充一维向量
a = tf.constant([2, 3, 5, 7])
b = tf.pad(a, [[1, 2]])
print(b)
''' tf.Tensor([0 2 3 5 7 0 0], shape=(7,), dtype=int32) '''

# 1.5 高维pad
a = tf.random.normal([4, 28, 28, 3])
b = tf.pad(a, [[0, 0], [1, 2], [3, 4], [0, 0]])
print(b.shape)
''' (4, 31, 35, 3) '''

# 2. tile
# 数据的复制，与broadcast类似，但是broadcast是隐式复制，性能更好
a = tf.constant([
    [1, 2, 3],
    [4, 5, 6],
])
b = tf.tile(a, [1, 2])
print(b)
''' tf.Tensor(
[[1 2 3 1 2 3]
 [4 5 6 4 5 6]], shape=(2, 6), dtype=int32) '''

c = tf.tile(a, [2, 1])
print(c)
''' tf.Tensor(
[[1 2 3]
 [4 5 6]
 [1 2 3]
 [4 5 6]], shape=(4, 3), dtype=int32) '''
