from imports import *

# Broadcasting
# 解释：对某个维度重复多次，但是不复制数据（内存优化）
# 准则：例如，a的shape是[4, 16, 16, 32]，b的shape是[32]
# 首先将两个shape右对齐，然后延伸b的shape，对于a有b无的维度，b插入1的维度
# 此时b的shape为[1, 1, 1, 32]，然后按照a的shape，对b的某些维度重复多次
# 直到b的shape变得与a相同
# 图示：Broadcasting.png

# 1. 隐式扩张
a = tf.constant([[1, 2, 4], [6, 7, 8]])
print(a)
''' tf.Tensor(
[[1 2 4]
 [6 7 8]], shape=(2, 3), dtype=int32) '''

b = tf.constant(3)
aa = a + b
print(aa)
''' tf.Tensor(
[[ 4  5  7]
 [ 9 10 11]], shape=(2, 3), dtype=int32) '''

c = tf.constant([[1], [2]])
aa = a + c
print(aa)
''' tf.Tensor(
[[ 2  3  5]
 [ 8  9 10]], shape=(2, 3), dtype=int32) '''

# 2. 显示扩张
a = tf.constant([2, 3])
b = tf.broadcast_to(a, [3, 2])
print(b)
''' tf.Tensor(
[[2 3]
 [2 3]
 [2 3]], shape=(3, 2), dtype=int32) '''