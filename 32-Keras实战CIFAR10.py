from imports import *
from tensorflow.keras import datasets, layers, Model, optimizers


def preprocess(x, y):
    # 将x的范围缩放到：-1 ~ 1
    x = (tf.cast(x, dtype=tf.float32) / 255. - 0.5) * 2
    y = tf.cast(y, dtype=tf.int32)
    return x, y


BATCH_SIZE = 128
(x, y), (x_test, y_test) = datasets.cifar10.load_data()
print(x.shape, y.shape)
''' (50000, 32, 32, 3) (50000, 1) '''
print(x.min(), x.max(), y.min(), y.max())
''' 0 255 0 9 '''

print(x_test.shape, y_test.shape)
''' (10000, 32, 32, 3) (10000, 1) '''

y = tf.squeeze(y)
y_test = tf.squeeze(y_test)
y = tf.one_hot(y, depth=10)
y_test = tf.one_hot(y_test, depth=10)
print(y.shape, y_test.shape)
''' (50000, 10) (10000, 10) '''

train_db = tf.data.Dataset.from_tensor_slices((x, y))
train_db = train_db.map(preprocess).shuffle(50000).batch(BATCH_SIZE)
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_db = test_db.map(preprocess).batch(BATCH_SIZE)

sample = next(iter(train_db))
print(sample[0].shape, sample[1].shape)
''' (128, 32, 32, 3) (128, 10) '''


class MyDense(layers.Layer):

    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.kernel = self.add_weight("w", [input_dim, output_dim])
        # self.bias = self.add_weight("b", [output_dim])

    def call(self, inputs, training=None):
        x = inputs @ self.kernel
        return x


class MyNetwork(Model):

    def __init__(self):
        super().__init__()

        self.fc1 = MyDense(32 * 32 * 3, 256)
        self.fc2 = MyDense(256, 256)
        self.fc3 = MyDense(256, 256)
        self.fc4 = MyDense(256, 256)
        self.fc5 = MyDense(256, 10)

    def call(self, inputs, training=None):
        # inputs_shape: [b, 32, 32, 3]

        x = tf.reshape(inputs, [-1, 32 * 32 * 3])

        # [b, 32*32*3] => [b, 256]
        x = self.fc1(x)
        x = tf.nn.relu(x)

        # [b, 256] => [b, 128]
        x = self.fc2(x)
        x = tf.nn.relu(x)

        # [b, 128] => [b, 64]
        x = self.fc3(x)
        x = tf.nn.relu(x)

        # [b, 64] => [b, 32]
        x = self.fc4(x)
        x = tf.nn.relu(x)

        # [b, 32] => [b, 10]
        x = self.fc5(x)

        return x


network = MyNetwork()
network.compile(
    optimizer=optimizers.Adam(lr=1e-3),
    loss=tf.losses.CategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)
network.fit(train_db, epochs=15, validation_data=test_db, validation_freq=1)

network.evaluate(test_db)
network.save_weights("./check_points/weights.ckpt")
del network
print("weights已保存！")

network = MyNetwork()
network.compile(
    optimizer=optimizers.Adam(lr=1e-3),
    loss=tf.losses.CategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)
network.load_weights("./check_points/weights.ckpt")
print("weights已加载！")

network.evaluate(test_db)
