from imports import *

# 1. MSE
tf.random.set_seed(1234)
y = tf.constant([1, 2, 3, 0, 2])
y = tf.one_hot(y, depth=4)
y = tf.cast(y, dtype=tf.float32)
print(y.shape)
''' (5, 4) '''

out = tf.random.normal([5, 4])

loss1 = tf.losses.MSE(y, out)
print(loss1)
''' tf.Tensor([1.1980209 0.5593645 0.8669833 0.8581686 0.3134297], shape=(5,), dtype=float32) '''

# 以下代码揭示了MSE的过程
y_numpy = y.numpy()
out_numpy = out.numpy()
loss2 = np.zeros([5])
for i in range(5):
    for j in range(4):
        loss2[i] += (y_numpy[i][j] - out_numpy[i][j]) ** 2
    loss2[i] /= 4
print(loss2)
''' [1.198021   0.55936452 0.86698329 0.85816863 0.31342972] '''


# 2. Entropy（熵）
# 公式见：assets/entropy.png
# lower entropy => more certainty
def entropy(tensor):
    res = -tf.reduce_sum(tensor * tf.math.log(tensor) / tf.math.log(2.))
    return float(res)

a = tf.constant([0., 0., 1., 0.])
print(entropy(a))
''' nan '''  # 此处理解为0更为合理

b = tf.constant([0.25, 0.25, 0.25, 0.25])
print(entropy(b))
''' 2.0 '''

c = tf.constant([0.01, 0.01, 0.01, 0.97])
print(entropy(c))
''' 0.2419406771659851 '''

d = tf.constant([0.1, 0.1, 0.1, 0.7])
print(entropy(d))
''' 1.3567795753479004 '''

# 3. Cross Entropy（交叉熵）
# 公式：assets/cross_entropy.png
# 计算实例：assets/cross_entropy_example.png

# 3.1 多分类问题计算交叉熵
# 注意！请记得加上from_logits=True，不然有可能出现数值不稳定的情况
a = tf.losses.categorical_crossentropy(
    [0, 1, 0, 0],
    [0.25, 0.25, 0.25, 0.25],
    from_logits=True,
)  # = -1 * log(0.25)
print(a)
''' tf.Tensor(1.3862944, shape=(), dtype=float32) '''

b = tf.losses.categorical_crossentropy(
    [0, 1, 0, 0],
    [0.05, 0.1, 0.8, 0.05],
)  # = -1 * log(0.1)
print(b)
''' tf.Tensor(2.3025851, shape=(), dtype=float32) '''

c = tf.losses.categorical_crossentropy(
    [0, 1, 0, 0],
    [0.1, 0.7, 0.1, 0.1],
)  # = -1 * log(0.7)
print(c)
''' tf.Tensor(0.35667497, shape=(), dtype=float32) '''

d = tf.losses.categorical_crossentropy(
    [0, 1, 0, 0],
    [0.01, 0.97, 0.01, 0.01],
)  # = -1 * log(0.97)
print(d)
''' tf.Tensor(0.030459179, shape=(), dtype=float32) '''

# 3.2 二分类
a = tf.losses.binary_crossentropy(
    [1],
    [0.2],
)
print(a)
''' tf.Tensor(1.6094373, shape=(), dtype=float32) '''

b = tf.losses.categorical_crossentropy(
    [1, 0],
    [0.2, 0.8],
)
print(b)
''' tf.Tensor(1.609438, shape=(), dtype=float32) '''
