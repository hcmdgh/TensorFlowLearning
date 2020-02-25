from imports import *

# 1. where

# 1.1 查询tensor中True值的所有下标
tf.random.set_seed(1234)
a = tf.random.normal([3, 4])
print(a)
''' tf.Tensor(
[[ 0.8369314  -0.73429775  1.0402943   0.04035992]
 [-0.72186583  1.0794858   0.9032698  -0.73601735]
 [-0.36105633 -0.6078763   0.07614239 -0.7211218 ]], shape=(3, 4), dtype=float32) '''

mask = a > 0
b = tf.boolean_mask(a, mask)
print(b)
''' tf.Tensor([0.8369314  1.0402943  0.04035992 1.0794858  0.9032698  0.07614239], shape=(6,), dtype=float32) '''

c = tf.where(mask)
print(c)
''' tf.Tensor(
[[0 0]
 [0 2]
 [0 3]
 [1 1]
 [1 2]
 [2 2]], shape=(6, 2), dtype=int64) '''

d = tf.gather_nd(a, c)
print(d)
''' 与b相同 '''

# 1.2 从A和B中筛选元素
a = tf.constant([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
])
b = tf.constant([
    [10, 11, 12],
    [13, 14, 15],
    [16, 17, 18],
])
condition = tf.constant([
    [True, False, True],
    [False, False, True],
    [True, True, False],
])

# condition中True值对应的位置填a，False值对应的位置填b
c = tf.where(condition, a, b)
print(c)
''' tf.Tensor(
[[ 1 11  3]
 [13 14  6]
 [ 7  8 18]], shape=(3, 3), dtype=int32) '''

# 2. scatter_nd
# 按照shape创建一个全0的tensor（底板），然后按照indices和updates更新
a = tf.scatter_nd(
    indices=tf.constant([[4], [3], [1], [7]]),
    updates=tf.constant([9, 10, 11, 12]),
    shape=[8],
)
print(a)
''' tf.Tensor([ 0 11  0 10  9  0  0 12], shape=(8,), dtype=int32) '''

# 3. meshgrid
# 给出x和y的范围，生成x坐标y坐标构成的点集
# 比如x=-2,-1,0,1,2 y=-2,-1,0,1,2 那么生成25个点的坐标
points = []
for y in np.linspace(-2, 2, 5):
    for x in np.linspace(-2, 2, 5):
        points.append([x, y])
points = np.array(points)
print(points.shape)
''' (25, 2) '''

x = tf.linspace(-2., 2, 5)  # 第一个参数必须为浮点型，否则报错
y = tf.linspace(-2., 2, 5)
print(x)
''' tf.Tensor([-2. -1.  0.  1.  2.], shape=(5,), dtype=float32) '''

points_x, points_y = tf.meshgrid(x, y)
print(points_x)
''' tf.Tensor(
[[-2. -1.  0.  1.  2.]
 [-2. -1.  0.  1.  2.]
 [-2. -1.  0.  1.  2.]
 [-2. -1.  0.  1.  2.]
 [-2. -1.  0.  1.  2.]], shape=(5, 5), dtype=float32) '''

print(points_y)
''' tf.Tensor(
[[-2. -2. -2. -2. -2.]
 [-1. -1. -1. -1. -1.]
 [ 0.  0.  0.  0.  0.]
 [ 1.  1.  1.  1.  1.]
 [ 2.  2.  2.  2.  2.]], shape=(5, 5), dtype=float32) '''

# points_x和points_y结合
points = tf.stack([points_x, points_y], axis=-1)
print(points.shape)
''' (5, 5, 2) '''

points = tf.reshape(points, [25, 2])
print(points)
''' tf.Tensor(
[[-2. -2.]
 [-1. -2.]
 [ 0. -2.]
 [ 1. -2.]
 [ 2. -2.]
 [-2. -1.]
 [-1. -1.]
 [ 0. -1.]
 [ 1. -1.]
 [ 2. -1.]
 [-2.  0.]
 [-1.  0.]
 [ 0.  0.]
 [ 1.  0.]
 [ 2.  0.]
 [-2.  1.]
 [-1.  1.]
 [ 0.  1.]
 [ 1.  1.]
 [ 2.  1.]
 [-2.  2.]
 [-1.  2.]
 [ 0.  2.]
 [ 1.  2.]
 [ 2.  2.]], shape=(25, 2), dtype=float32) '''
