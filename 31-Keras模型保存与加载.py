from imports import *
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics


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


# 1. 保存权值weights
def save_weights():
    network = Sequential([
        layers.Dense(256, activation="relu"),
        layers.Dense(128, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(10),
    ])
    network.build(input_shape=[None, 784])
    network.summary()
    network.compile(
        optimizer=optimizers.Adam(lr=0.01),
        loss=tf.losses.CategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    network.fit(db, epochs=3, validation_data=ds_val, validation_freq=1)
    network.evaluate(ds_val)

    # 保存权值
    network.save_weights("./check_points/weights.ckpt")
    print("weights已保存！")

    del network
    network = Sequential([
        layers.Dense(256, activation="relu"),
        layers.Dense(128, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(10),
    ])
    # network.build(input_shape=[None, 784])
    network.compile(
        optimizer=optimizers.Adam(lr=0.01),
        loss=tf.losses.CategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    # 加载权值
    network.load_weights("./check_points/weights.ckpt")
    print("weights已加载！")

    network.evaluate(ds_val)


# 2. 保存模型model
def save_model():
    network = Sequential([
        layers.Dense(256, activation="relu"),
        layers.Dense(128, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(10),
    ])
    network.build(input_shape=[None, 784])
    network.summary()
    network.compile(
        optimizer=optimizers.Adam(lr=0.01),
        loss=tf.losses.CategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    network.fit(db, epochs=3, validation_data=ds_val, validation_freq=1)
    network.evaluate(ds_val)

    # 保存model
    network.save("./check_points/model.h5")
    print("模型已保存！")
    del network

    # 加载model
    network = tf.keras.models.load_model("./check_points/model.h5")
    print("模型已加载！")

    network.summary()
    network.evaluate(ds_val)


if __name__ == '__main__':
    save_model()
