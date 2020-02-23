from imports import *
from tensorflow import keras
from tensorflow.keras import datasets

# 前向传播（张量）实战
# 推导过程见：前向传播.png

# 1. 加载数据集
(x, y), _ = datasets.mnist.load_data()
print(x.shape, y.shape)
''' (60000, 28, 28) (60000,) '''

# 2. 数据转换为tensor
x = tf.convert_to_tensor(x, dtype=tf.float32)
y = tf.convert_to_tensor(y, dtype=tf.int32)

# 查看x的数据范围
print(tf.reduce_min(x), tf.reduce_max(x))
''' tf.Tensor(0.0, shape=(), dtype=float32) tf.Tensor(255.0, shape=(), dtype=float32) '''
print(tf.reduce_min(y), tf.reduce_max(y))
''' tf.Tensor(0, shape=(), dtype=int32) tf.Tensor(9, shape=(), dtype=int32) '''

# 3. 将x的范围由 0~255 => 0~1
x /= 255.

# 4. 创建数据集
train_db = tf.data.Dataset.from_tensor_slices((x, y)).batch(128)

# 每次取128个样本
train_iter = iter(train_db)
sample = next(train_iter)
print("batch_shape:", sample[0].shape, sample[1].shape)
''' batch_shape: (128, 28, 28) (128,) '''

# 5. 创建权值
# [b, 784] => [b, 256] => [b, 128] => [b, 10]
# w_shape: [dim_in, dim_out]
# b_shape: [dim_out]
# 注：一定要用tf.Variable封装
# 注2：方差要调小一些，防止出现梯度爆炸
w1 = tf.Variable(tf.random.truncated_normal([784, 256], stddev=0.1))
b1 = tf.Variable(tf.zeros([256]))
w2 = tf.Variable(tf.random.truncated_normal([256, 128], stddev=0.1))
b2 = tf.Variable(tf.zeros([128]))
w3 = tf.Variable(tf.random.truncated_normal([128, 10], stddev=0.1))
b3 = tf.Variable(tf.zeros([10]))

# 学习率
learning_rate = 1e-3

# 6. 前向运算
for epoch in range(10):  # 对整个数据集迭代
    for step, (x, y) in enumerate(train_db):  # 对每个batch迭代
        # x_shape: [128, 28, 28]
        # y_shape: [128]

        # 将x的shape变换为[b, 28*28]
        x = tf.reshape(x, [-1, 28*28])

        with tf.GradientTape() as tape:  # 注：只会跟踪tf.Variable类型
            # [b, 784] @ [784, 256] + [256] = [b, 256]
            h1 = x @ w1 + b1

            # 加入非线性转换，下同
            h1 = tf.nn.relu(h1)

            # [b, 256] @ [256, 128] + [128] = [b, 128]
            h2 = h1 @ w2 + b2
            h2 = tf.nn.relu(h2)

            out = h2 @ w3 + b3

            # 计算loss

            # out_shape: [b, 10]
            # y_shape: [b] => [b, 10]
            y_one_hot = tf.one_hot(y, depth=10)

            # 均方差
            loss = tf.reduce_mean(tf.square(y_one_hot - out))

        # 计算梯度
        grads = tape.gradient(loss, [w1, b1, w2, b2, w3, b3])

        # 更新w1, b1, w2, b2, w3, b3
        # 原地更新，使用assign_sub()
        w1.assign_sub(learning_rate * grads[0])
        b1.assign_sub(learning_rate * grads[1])
        w2.assign_sub(learning_rate * grads[2])
        b2.assign_sub(learning_rate * grads[3])
        w3.assign_sub(learning_rate * grads[4])
        b3.assign_sub(learning_rate * grads[5])

        if step % 100 == 0:
            print(epoch, step, "loss:", float(loss))
