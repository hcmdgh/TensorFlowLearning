from imports import *

# 1. 全连接层
# 图示见assets/全连接层xx
# 层包括：输入层、隐藏层、输出层

x = tf.random.normal([4, 784])
net = tf.keras.layers.Dense(512)

# 1.1 调用net，自动创建net.kernel和net.bias
out = net(x)

# [4, 784] @ [784, 512] = [4, 512]
print(out.shape)
''' (4, 512) '''

# w和k的shape
print(net.kernel.shape, net.bias.shape)
''' (784, 512) (512,) '''

# 1.2 net.build
net = tf.keras.layers.Dense(10)
net.build(input_shape=(None, 4))
print(net.kernel.shape, net.bias.shape)
''' (4, 10) (10,) '''

net = tf.keras.layers.Dense(10)
net.build(input_shape=(None, 20))
print(net.kernel.shape, net.bias.shape)
''' (20, 10) (10,) '''

net = tf.keras.layers.Dense(10)
net.build(input_shape=(3, 4))
print(net.kernel.shape, net.bias.shape)
''' (4, 10) (10,) '''

# 1.3 keras.Sequential（Dense层的嵌套）
x = tf.random.normal([2, 3])

model = tf.keras.Sequential([
    tf.keras.layers.Dense(2, activation="relu"),
    tf.keras.layers.Dense(2, activation="relu"),
    tf.keras.layers.Dense(2),
])
model.build(input_shape=[None, 3])

# 查看网络结构
model.summary()
'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_4 (Dense)              multiple                  8         
_________________________________________________________________
dense_5 (Dense)              multiple                  6         
_________________________________________________________________
dense_6 (Dense)              multiple                  6         
=================================================================
Total params: 20
Trainable params: 20
Non-trainable params: 0
_________________________________________________________________
'''

# [w1, b1, w2, b2, w3, b3]
for p in model.trainable_variables:
    print(type(p), p.name, p.shape)
'''
<class 'tensorflow.python.ops.resource_variable_ops.ResourceVariable'> dense_4/kernel:0 (3, 2)
<class 'tensorflow.python.ops.resource_variable_ops.ResourceVariable'> dense_4/bias:0 (2,)
<class 'tensorflow.python.ops.resource_variable_ops.ResourceVariable'> dense_5/kernel:0 (2, 2)
<class 'tensorflow.python.ops.resource_variable_ops.ResourceVariable'> dense_5/bias:0 (2,)
<class 'tensorflow.python.ops.resource_variable_ops.ResourceVariable'> dense_6/kernel:0 (2, 2)
<class 'tensorflow.python.ops.resource_variable_ops.ResourceVariable'> dense_6/bias:0 (2,)
'''