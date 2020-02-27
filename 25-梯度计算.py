from imports import *

# 常见函数的梯度，参见：
# assets/常见函数的梯度/

# 1. 自动计算梯度

# 1.1 tape.watch()

w = tf.constant(2.)
x = tf.constant(3.)

with tf.GradientTape() as tape:
    tape.watch([x])
    y = (w**2) * (x**2)

# y = (w^2)(x^2)
# y对x求导(dy/dx)
# y' = 2(w^2)x = 24
grad = tape.gradient(y, [x])
print(grad)
''' [<tf.Tensor: shape=(), dtype=float32, numpy=24.0>] '''

# grad2 = tape.gradient(y, [x])
# RuntimeError: GradientTape.gradient can only be called once on non-persistent tapes.
# 如果要调用多次，上方修改为"with tf.GradientTape(persistent=True) as tape:"

# 1.2 tf.Variable

x = tf.Variable(2.)
w = tf.Variable(3.)

with tf.GradientTape() as tape:
    y = (w**2) * (x**3)

# y = (w^2)(x^3)
# y' = 3(w^2)(x^2) = 108
grad = tape.gradient(y, [x])
print(grad)
''' [<tf.Tensor: shape=(), dtype=float32, numpy=108.0>] '''

# 2. MSE梯度计算

x = tf.random.normal([2, 4])
w = tf.random.normal([4, 3])
b = tf.zeros([3])
y = tf.constant([2, 0])

with tf.GradientTape() as tape:
    tape.watch([w, b])

    # prob_shape: [2, 3]
    # y_onehot_shape: [2, 3]
    prob = tf.nn.softmax(x @ w + b, axis=1)
    loss = tf.reduce_mean(tf.losses.MSE(tf.one_hot(y, depth=3), prob))

grads = tape.gradient(loss, [w, b])
print(grads[0])
''' tf.Tensor(
[[-0.04857158  0.0378082   0.0107634 ]
 [-0.03037842  0.00383345  0.02654496]
 [ 0.03670761 -0.03769311  0.00098549]
 [ 0.00831253 -0.0252247   0.01691214]], shape=(4, 3), dtype=float32) '''

print(grads[1])
''' tf.Tensor([-0.09032173  0.05539995  0.0349218 ], shape=(3,), dtype=float32) '''
