from imports import *

# 1. 按下标索引
# （与numpy基本相同）

# 2. 切片
# （与numpy基本相同）
a = tf.range(10)
print(a)
''' tf.Tensor([0 1 2 3 4 5 6 7 8 9], shape=(10,), dtype=int32) '''

b = a[-2:]
print(b)
''' tf.Tensor([8 9], shape=(2,), dtype=int32) '''

c = a[:-2]
print(c)
''' tf.Tensor([0 1 2 3 4 5 6 7], shape=(8,), dtype=int32) '''

d = tf.ones([2, 3, 4])
dd = d[0, :, :]
print(dd.shape)
''' (3, 4) '''

e = tf.range(10)
ee = e[::2]
print(ee)
''' tf.Tensor([0 2 4 6 8], shape=(5,), dtype=int32) '''

# 2.1 逆序
ee = e[::-1]
print(ee)
''' tf.Tensor([9 8 7 6 5 4 3 2 1 0], shape=(10,), dtype=int32) '''

# 2.2 省略号...
f = tf.zeros([2, 3, 4, 5, 8])
f1 = f[..., 0]
print(f1.shape)
''' (2, 3, 4, 5) '''

f2 = f[0, ..., 2]
print(f2.shape)
''' (3, 4, 5) '''

f3 = f[1, ...]
print(f3.shape)
''' (3, 4, 5, 8) '''

# 3. 更高级的索引

# data: [classes, students, subjects]
data = tf.zeros([4, 35, 8])

# 3.1 提取2、3、1号班级的信息
a = tf.gather(data, axis=0, indices=[2, 3, 1])
print(a.shape)
''' (3, 35, 8) '''

# 3.2 提取一部分学生的信息
b = tf.gather(data, axis=1, indices=[2, 30, 19, 7, 34])
print(b.shape)
''' (4, 5, 8) '''

# 3.3 提取2、7、3号课程的信息
c = tf.gather(data, axis=2, indices=[2, 7, 3])
print(c.shape)
''' (4, 35, 3) '''

# 3.4 gather_nd
# 诀窍：把最内层[]当做要取的索引，然后将每一个索引对应的结果堆叠
a = tf.gather_nd(data, [0])
print(a.shape)
''' (35, 8) '''

b = tf.gather_nd(data, [[1, 2]])
print(b.shape)
''' (1, 8) '''

c = tf.gather_nd(data, [0, 2, 1])
print(c.shape)
''' () '''

d = tf.gather_nd(data, [[0, 2, 1], [2, 3, 4]])
print(d.shape)
''' (2,) '''

# 3.5 boolean_mask

# 3.5.1 提取1号和3号班级的信息
a = tf.boolean_mask(data, axis=0, mask=[False, True, False, True])
print(a.shape)
''' (2, 35, 8) '''

# 3.5.2 提取1号和3号课程的信息
b = tf.boolean_mask(data, axis=2, mask=[False, True, False, True, False, False, False, False])
print(b.shape)
''' (4, 35, 2) '''

# 3.5.3 boolean_mask也可以多维组合，类似gather_nd
a = tf.ones([2, 3, 4])
b = tf.boolean_mask(a, mask=[[True, False, False], [False, True, True]])
print(b.shape)
''' (3, 4) '''
