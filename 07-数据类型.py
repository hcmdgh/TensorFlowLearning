import tensorflow as tf
import numpy as np

# 张量tensor
# 整数默认类型为int32
a = tf.constant(1)
print(a)  # tf.Tensor(1, shape=(), dtype=int32)

# 实数默认类型为float32
b = tf.constant(1.0)
print(b)  # tf.Tensor(1.0, shape=(), dtype=float32)

# c = tf.constant(2.9, dtype=tf.int32)
# print(c)
# TypeError

d = tf.constant(2., dtype=tf.double)
print(d)  # tf.Tensor(2.0, shape=(), dtype=float64)

e = tf.constant([True, False])
print(e)  # tf.Tensor([ True False], shape=(2,), dtype=bool)

f = tf.constant("hello, 你好")
print(f)  # tf.Tensor(b'hello, \xe4\xbd\xa0\xe5\xa5\xbd', shape=(), dtype=string)

# tensor转numpy
g = e.numpy()
print(type(g))  # <class 'numpy.ndarray'>
print(g)  # [ True False]

# 查看tensor的维度
h = tf.constant([1, 2, 4, 8])
print(h)  # tf.Tensor([1 2 4 8], shape=(4,), dtype=int32)
print(h.shape)  # (4,)

# 维数
print(d.ndim)  # 0
print(h.ndim)  # 1

# 维数（返回值为tensor形式）
print(tf.rank(d))  # tf.Tensor(0, shape=(), dtype=int32)
print(tf.rank(h))  # tf.Tensor(1, shape=(), dtype=int32)

# 判断一个变量是不是tensor
print(tf.is_tensor(a))  # True

# 类型转换
a = np.arange(5)
print(a.dtype)  # int32
aa = tf.convert_to_tensor(a)
print(aa)  # tf.Tensor([0 1 2 3 4], shape=(5,), dtype=int32)
a2 = tf.convert_to_tensor(a, dtype=tf.int64)
print(a2)  # tf.Tensor([0 1 2 3 4], shape=(5,), dtype=int64)

# 强制类型转换
b = tf.cast(aa, dtype=tf.float32)
print(b)  # tf.Tensor([0. 1. 2. 3. 4.], shape=(5,), dtype=float32)

# tf.range
c = tf.range(5)
print(c)  # tf.Tensor([0 1 2 3 4], shape=(5,), dtype=int32)

# tf.Variable
d = tf.Variable(c)
print(d)  # <tf.Variable 'Variable:0' shape=(5,) dtype=int32, numpy=array([0, 1, 2, 3, 4])>
print(d.name)  # Variable:0
d = tf.Variable(c, name="input_data")
print(d)  # <tf.Variable 'input_data:0' shape=(5,) dtype=int32, numpy=array([0, 1, 2, 3, 4])>
print(d.name)  # input_data:0

# isinstance
print(isinstance(d, tf.Tensor))  # False
print(isinstance(d, tf.Variable))  # True
print(tf.is_tensor(d))  # True

# tensor取回数据（tensor转换为numpy）
e = d.numpy()
print(e)  # [0 1 2 3 4]
f = tf.ones([])
f1 = f.numpy()
print(f1)  # 1.0
f2 = int(f)
print(f2)  # 1
