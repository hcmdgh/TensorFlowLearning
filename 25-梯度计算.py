from imports import *

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
