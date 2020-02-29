from imports import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


# himmelblau函数的定义
# f(x, y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2
# 一共有4个精确解，使得f(x, y) = 0
# 见assets/himmelblau精确解.png
def himmelblau(x):
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2


# 作图
def draw():
    x = np.arange(-6, 6, 0.1)
    y = np.arange(-6, 6, 0.1)
    X, Y = np.meshgrid(x, y)
    Z = himmelblau([X, Y])

    fig = plt.figure("himmelblau")
    ax = fig.gca(projection="3d")
    ax.plot_surface(X, Y, Z)
    ax.view_init(60, -30)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.show()


# 梯度下降
# 参数为x初始点的坐标
# 使用不同的初始点，可以得到4个不同的解
def gradient_descent(x):
    for step in range(200):
        with tf.GradientTape() as tape:
            tape.watch([x])
            y = himmelblau(x)
        grads = tape.gradient(y, [x])[0]
        x -= 0.01 * grads
        if (step + 1) % 20 == 0:
            print("step:", step, ", x:", x, ", f(x)=", y.numpy())


if __name__ == '__main__':
    x = tf.constant([-4., 0.])
    gradient_descent(x)
