from imports import *

# 1. 向量范数
# 见norm.png

# 1.1 默认为2-范数
a = tf.ones([2, 2])
b = tf.norm(a)
print(b)
''' tf.Tensor(2.0, shape=(), dtype=float32) '''

# 实际上，norm就等效于下面这种写法
c = tf.sqrt(tf.reduce_sum(tf.square(a)))
print(c)
''' tf.Tensor(2.0, shape=(), dtype=float32) '''

a = tf.ones([4, 28, 27, 3])
b = tf.norm(a)
print(b)
''' tf.Tensor(95.24705, shape=(), dtype=float32) '''

c = tf.sqrt(tf.reduce_sum(tf.square(a)))
print(c)
''' tf.Tensor(95.24705, shape=(), dtype=float32) '''

# 1.2 指定范数ord和轴axis
a = tf.constant([[1, -2, 3], [3, -4, -5]], dtype=tf.float32)
print(a)
''' [[ 1. -2.  3.]
 [ 3. -4. -5.]], shape=(2, 3), dtype=float32) '''

b = tf.norm(a, ord=2)  # b = sqrt(1^2 + 2^2 + 3^2 + 3^2 + 4^2 + 5^2)
print(b)
''' tf.Tensor(8.0, shape=(), dtype=float32) '''

c = tf.norm(a, ord=1)  # c = 1 + 2 + 3 + 3 + 4 + 5
print(c)
''' tf.Tensor(18.0, shape=(), dtype=float32) '''

d = tf.norm(a, ord=1, axis=0)
print(d)
''' tf.Tensor([4. 6. 8.], shape=(3,), dtype=float32) '''

e = tf.norm(a, ord=1, axis=1)
print(e)
''' tf.Tensor([ 6. 12.], shape=(2,), dtype=float32) '''

# 2. reduce_min/max/sum/mean
# reduce表示这是一个降维的过程

# 2.1 将tensor中所有数据当成一个整体
print(a)
''' [[ 1. -2.  3.]
 [ 3. -4. -5.]], shape=(2, 3), dtype=float32) '''

b = tf.reduce_min(a)
print(b)
''' tf.Tensor(-5.0, shape=(), dtype=float32) '''

b = tf.reduce_max(a)
print(b)
''' tf.Tensor(3.0, shape=(), dtype=float32) '''

b = tf.reduce_sum(a)
print(b)
''' tf.Tensor(-4.0, shape=(), dtype=float32) '''

b = tf.reduce_mean(a)
print(b)
''' tf.Tensor(-0.6666667, shape=(), dtype=float32) '''

# 2.2 对指定轴axis进行计算
print(a)
''' [[ 1. -2.  3.]
 [ 3. -4. -5.]], shape=(2, 3), dtype=float32) '''

b = tf.reduce_sum(a, axis=0)
print(b)
''' tf.Tensor([ 4. -6. -2.], shape=(3,), dtype=float32) '''

b = tf.reduce_sum(a, axis=1)
print(b)
''' tf.Tensor([ 2. -6.], shape=(2,), dtype=float32) '''

# 3. argmax/argmin
print(a)
''' [[ 1. -2.  3.]
 [ 3. -4. -5.]], shape=(2, 3), dtype=float32) '''

# axis默认为0
b = tf.argmax(a)  # 每一列最大元素的下标：每一列最大元素为3、-2、3，下标依次为1、0、0
print(b)
''' tf.Tensor([1 0 0], shape=(3,), dtype=int64) '''

b = tf.argmax(a, axis=1)
print(b)
''' tf.Tensor([2 0], shape=(2,), dtype=int64) '''

c = tf.constant([1, 2, 0, 3, 0, 4, 0, 5])
print(tf.argmin(c))
''' tf.Tensor(2, shape=(), dtype=int64) '''

# 4. equal
a = tf.constant([1, 4, 3, 2, 5])
b = tf.constant([1, 2, 3, 4, 5])
c = tf.equal(a, b)
print(c)
''' tf.Tensor([ True False  True False  True], shape=(5,), dtype=bool) '''

# 统计匹配的元素个数（True的个数）
d = tf.reduce_sum(tf.cast(c, tf.int32))
print(d)
''' tf.Tensor(3, shape=(), dtype=int32) '''

# 5. unique
# 去除重复元素，同时提供老tensor中元素在新tensor中的位置
a = tf.constant([2, 5, 3, 7, 2, 3, 6, 7, 8, 5])
b = tf.unique(a)
print(b)
''' Unique(y=<tf.Tensor: shape=(6,), dtype=int32, numpy=array([2, 5, 3, 7, 6, 8])>, idx=<tf.Tensor: shape=(10,), dtype=int32, numpy=array([0, 1, 2, 3, 0, 2, 4, 3, 5, 1])>) '''

c = b.y
print(c)
''' tf.Tensor([2 5 3 7 6 8], shape=(6,), dtype=int32) '''

d = b.idx
print(d)
''' tf.Tensor([0 1 2 3 0 2 4 3 5 1], shape=(10,), dtype=int32) '''
