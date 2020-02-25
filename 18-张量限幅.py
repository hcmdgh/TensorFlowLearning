from imports import *

# 1. maximum/minimum
a = tf.range(10)
print(a)
''' tf.Tensor([0 1 2 3 4 5 6 7 8 9], shape=(10,), dtype=int32) '''

b = tf.maximum(a, 4)
print(b)
''' tf.Tensor([4 4 4 4 4 5 6 7 8 9], shape=(10,), dtype=int32) '''

c = tf.minimum(a, 6)
print(c)
''' tf.Tensor([0 1 2 3 4 5 6 6 6 6], shape=(10,), dtype=int32) '''

# 2. clip_by_value
# maximum + minimum
d = tf.clip_by_value(a, 3, 7)
print(d)
''' tf.Tensor([3 3 3 3 4 5 6 7 7 7], shape=(10,), dtype=int32) '''

# 3. relu
a = tf.constant([1, 3, -5, -2, 4, 0])
b = tf.nn.relu(a)
print(b)
''' tf.Tensor([1 3 0 0 4 0], shape=(6,), dtype=int32) '''

# relu等效于maximum 0
b = tf.maximum(a, 0)
print(b)
''' tf.Tensor([1 3 0 0 4 0], shape=(6,), dtype=int32) '''

# 4. clip_by_norm
# 按2-范数进行裁剪，将大于指定值的进行裁剪，小于的则不变
a = tf.constant([
    [2, -3, -5],  # 无需裁剪
    [1, 4, -7],  # 无需裁剪
    [9, -3, 8],  # 需要裁剪(>10)
    [10, 7, 5],  # 需要裁剪(>10)
], dtype=tf.float32)
print(tf.norm(a, axis=1))
''' tf.Tensor([ 6.164414  8.124039 12.409674 13.190906], shape=(4,), dtype=float32) '''

b = tf.clip_by_norm(a, axes=1, clip_norm=10)
print(tf.norm(b, axis=1))
''' tf.Tensor([ 6.164414  8.124039 10.       10.      ], shape=(4,), dtype=float32) '''

print(b)
''' tf.Tensor(
[[ 2.        -3.        -5.       ]
 [ 1.         4.        -7.       ]
 [ 7.2524066 -2.4174688  6.4465837]
 [ 7.580981   5.3066864  3.7904904]], shape=(4, 3), dtype=float32) '''

# 5. clip_by_global_norm
# 有待进一步挖掘
