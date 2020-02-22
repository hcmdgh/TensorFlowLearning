from imports import *

# 创建tensor
# （tf.convert_to_tensor与tf.constant几乎相同）

# 1. 从numpy创建tensor
a = np.ones((2, 3))
b = tf.convert_to_tensor(a)
print(b)
''' tf.Tensor(
[[1. 1. 1.]
 [1. 1. 1.]], shape=(2, 3), dtype=float64) '''

# 2. 从list创建tensor
c = [[2, 3], [4.0, 5]]
d = tf.convert_to_tensor(c)
print(d)
''' tf.Tensor(
[[2. 3.]
 [4. 5.]], shape=(2, 2), dtype=float32) '''

# 3. zeros_like，根据指定tensor生成值全为0的tensor
e = tf.convert_to_tensor([(2, 3, 5), (1, 2, 4)])
f = tf.zeros_like(e)
print(f)
''' tf.Tensor(
[[0 0 0]
 [0 0 0]], shape=(2, 3), dtype=int32) '''

# 4. fill，创建tensor并用指定值填充
g = tf.fill([2, 2], 6)
print(g)
''' tf.Tensor(
[[6 6]
 [6 6]], shape=(2, 2), dtype=int32) '''

h = tf.fill([3], 6)
print(h)
''' tf.Tensor([6 6 6], shape=(3,), dtype=int32) '''

# 5. 正态分布
a = tf.random.normal([2, 3], mean=1, stddev=1)
print(a)
''' tf.Tensor(
[[ 1.240536    1.4384677   1.7262609 ]
 [ 0.51700103 -0.12437785  1.8561022 ]], shape=(2, 3), dtype=float32) '''

# 5.1 截断正态分布
# 产生截断正态分布随机数，取值范围为 [ mean - 2 * stddev, mean + 2 * stddev ]
b = tf.random.truncated_normal([2, 3], mean=0, stddev=1)
print(b)
''' tf.Tensor(
[[-0.92869145 -0.6822932   1.0377316 ]
 [ 0.57966846 -0.37941933 -0.19191416]], shape=(2, 3), dtype=float32) '''

# 6. 均匀分布
# 范围：[minval, maxval]
c = tf.random.uniform([2, 3], minval=0, maxval=1)
print(c)
''' tf.Tensor(
[[0.88482714 0.01443505 0.35363746]
 [0.5170467  0.5404012  0.21803832]], shape=(2, 3), dtype=float32) '''
d = tf.random.uniform([10], minval=0, maxval=5, dtype=tf.int32)
print(d)
''' tf.Tensor([0 2 3 1 0 4 0 1 2 2], shape=(10,), dtype=int32) '''
