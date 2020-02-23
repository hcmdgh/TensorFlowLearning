from imports import *

# 1. 加减乘除、取余
a = tf.fill([2, 2], 3.0)
b = tf.fill([2, 2], 2.0)
print(a + b)
''' tf.Tensor(
[[5. 5.]
 [5. 5.]], shape=(2, 2), dtype=float32) '''

print(a - b)
''' tf.Tensor(
[[1. 1.]
 [1. 1.]], shape=(2, 2), dtype=float32) '''

print(a * b)
''' tf.Tensor(
[[6. 6.]
 [6. 6.]], shape=(2, 2), dtype=float32) '''

print(a / b)
''' tf.Tensor(
[[1.5 1.5]
 [1.5 1.5]], shape=(2, 2), dtype=float32) '''

print(a // b)
''' tf.Tensor(
[[1. 1.]
 [1. 1.]], shape=(2, 2), dtype=float32) '''

print(a % b)
''' tf.Tensor(
[[1. 1.]
 [1. 1.]], shape=(2, 2), dtype=float32) '''

# 2. 指数和对数
a = tf.range(1.0, 5.0)

print(tf.math.log(a))
''' tf.Tensor([0.        0.6931472 1.0986123 1.3862944], shape=(4,), dtype=float32) '''

print(tf.math.exp(a))
''' tf.Tensor([ 2.7182817  7.389056  20.085537  54.59815  ], shape=(4,), dtype=float32) '''

print(tf.pow(a, 2))
''' tf.Tensor([ 1.  4.  9. 16.], shape=(4,), dtype=float32) '''

print(a ** 2)
''' tf.Tensor([ 1.  4.  9. 16.], shape=(4,), dtype=float32) '''

print(tf.sqrt(a))
''' tf.Tensor([1.        1.4142135 1.7320508 2.       ], shape=(4,), dtype=float32) '''

# 3. 矩阵乘法
# 对于3维矩阵，对末2个维度作矩阵乘法
a = tf.ones([4, 2, 3])
b = tf.fill([4, 3, 5], 2.)
c = a @ b
print(c.shape)
''' (4, 2, 5)'''
print(c)
''' 全是6 '''
