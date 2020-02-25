from imports import *

# 1. sort

# 1.1 sort不改变原张量
a = tf.constant([2, 5, 1, 2])
b = tf.sort(a)
print(a, b)
''' tf.Tensor([2 5 1 2], shape=(4,), dtype=int32) tf.Tensor([1 2 2 5], shape=(4,), dtype=int32) '''

# 1.2 降序排序
b = tf.sort(a, direction="DESCENDING")
print(b)
''' tf.Tensor([5 2 2 1], shape=(4,), dtype=int32) '''

# 1.3 指定轴axis排序
a = tf.random.uniform(maxval=10, shape=[3, 4], dtype=tf.int32)
print(a)
''' tf.Tensor(
[[8 5 3 4]
 [0 6 7 7]
 [8 4 3 8]], shape=(3, 4), dtype=int32) '''

b = tf.sort(a, axis=0)
print(b)
''' tf.Tensor(
[[0 4 3 4]
 [8 5 3 7]
 [8 6 7 8]], shape=(3, 4), dtype=int32) '''

b = tf.sort(a, axis=1)
print(b)
''' tf.Tensor(
[[3 4 5 8]
 [0 6 7 7]
 [3 4 8 8]], shape=(3, 4), dtype=int32) '''

# 2. argsort
a = tf.constant([4, 1, 2, 5, 3])
b = tf.argsort(a)  # 此时b中元素为12345在a中的下标
print(b)
''' tf.Tensor([1 2 4 0 3], shape=(5,), dtype=int32) '''

c = tf.gather(a, b)
print(c)
''' tf.Tensor([1 2 3 4 5], shape=(5,), dtype=int32) '''

# 3. top_k
# 返回最大的k个元素以及它们的下标
print(a)
''' tf.Tensor([4 1 2 5 3], shape=(5,), dtype=int32) '''

b = tf.math.top_k(a, 2)
print(b)
''' TopKV2(values=<tf.Tensor: shape=(2,), dtype=int32, numpy=array([5, 4])>, indices=<tf.Tensor: shape=(2,), dtype=int32, numpy=array([3, 0])>) '''

print(b.values)
''' tf.Tensor([5 4], shape=(2,), dtype=int32) '''

print(b.indices)
''' tf.Tensor([3 0], shape=(2,), dtype=int32) '''
