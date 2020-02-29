from imports import *
from tensorflow.keras import datasets, layers, optimizers
from tensorflow import keras


def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255.
    x = tf.reshape(x, [784])
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)
    return x, y


BATCH_SIZE = 128
(x, y), (x_test, y_test) = datasets.mnist.load_data()
db = tf.data.Dataset.from_tensor_slices((x, y))
db = db.map(preprocess).shuffle(60000).batch(BATCH_SIZE)
ds_val = tf.data.Dataset.from_tensor_slices((x_test, y_test))
ds_val = ds_val.map(preprocess).batch(BATCH_SIZE)


class MyDense(layers.Layer):

    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.kernel = self.add_weight("w", [input_dim, output_dim])
        self.bias = self.add_weight("b", [output_dim])

    def call(self, inputs, training=None):
        out = inputs @ self.kernel + self.bias
        return out


class MyModel(keras.Model):

    def __init__(self):
        super().__init__()

        self.fc1 = MyDense(784, 256)
        self.fc2 = MyDense(256, 128)
        self.fc3 = MyDense(128, 64)
        self.fc4 = MyDense(64, 32)
        self.fc5 = MyDense(32, 10)

    def call(self, inputs, training=None):
        x = self.fc1(inputs)
        x = tf.nn.relu(x)
        x = self.fc2(x)
        x = tf.nn.relu(x)
        x = self.fc3(x)
        x = tf.nn.relu(x)
        x = self.fc4(x)
        x = tf.nn.relu(x)
        x = self.fc5(x)
        return x


network = MyModel()
network.compile(
    optimizer=optimizers.Adam(lr=0.01),
    loss=tf.losses.CategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
network.fit(db, epochs=5, validation_data=ds_val, validation_freq=1)
network.evaluate(ds_val)

sample = next(iter(ds_val))
x, y = sample
pred = network.predict(x)
y = tf.argmax(y, axis=1)
pred = tf.argmax(pred, axis=1)

print(pred)
print(y)
'''
tf.Tensor(
[7 2 1 0 4 1 4 9 5 9 0 6 9 0 1 5 9 7 3 4 9 6 6 5 4 0 7 4 0 1 3 1 3 4 7 2 7
 1 2 1 1 7 4 2 3 5 1 2 4 4 6 3 5 5 6 0 4 1 9 5 7 8 9 3 7 4 6 4 3 0 7 0 2 9
 1 7 3 2 9 7 7 6 2 7 8 4 7 3 6 1 3 6 9 3 1 4 1 7 6 9 6 0 5 4 9 9 2 1 9 4 8
 7 3 9 7 9 4 4 9 2 5 4 7 6 7 9 0 5], shape=(128,), dtype=int64)
tf.Tensor(
[7 2 1 0 4 1 4 9 5 9 0 6 9 0 1 5 9 7 3 4 9 6 6 5 4 0 7 4 0 1 3 1 3 4 7 2 7
 1 2 1 1 7 4 2 3 5 1 2 4 4 6 3 5 5 6 0 4 1 9 5 7 8 9 3 7 4 6 4 3 0 7 0 2 9
 1 7 3 2 9 7 7 6 2 7 8 4 7 3 6 1 3 6 9 3 1 4 1 7 6 9 6 0 5 4 9 9 2 1 9 4 8
 7 3 9 7 4 4 4 9 2 5 4 7 6 7 9 0 5], shape=(128,), dtype=int64)
'''
