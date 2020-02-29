# 1. Metrics
# 1.1 metrics.Accuracy()
# 1.2 metrics.Mean()

import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics


def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.cast(y, dtype=tf.int32)

    return x, y


batchsz = 128
(x, y), (x_val, y_val) = datasets.mnist.load_data()

db = tf.data.Dataset.from_tensor_slices((x, y))
db = db.map(preprocess).shuffle(60000).batch(batchsz).repeat(10)

ds_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
ds_val = ds_val.map(preprocess).batch(batchsz)

network = Sequential([layers.Dense(256, activation='relu'),
                      layers.Dense(128, activation='relu'),
                      layers.Dense(64, activation='relu'),
                      layers.Dense(32, activation='relu'),
                      layers.Dense(10)])
network.build(input_shape=(None, 28 * 28))

optimizer = optimizers.Adam(lr=0.01)

# 创建meter
acc_meter = metrics.Accuracy()
loss_meter = metrics.Mean()

for step, (x, y) in enumerate(db):

    with tf.GradientTape() as tape:
        x = tf.reshape(x, (-1, 28 * 28))
        out = network(x)
        y_onehot = tf.one_hot(y, depth=10)

        # 更新loss_meter
        loss = tf.reduce_mean(tf.losses.categorical_crossentropy(y_onehot, out, from_logits=True))
        loss_meter.update_state(loss)

    grads = tape.gradient(loss, network.trainable_variables)
    optimizer.apply_gradients(zip(grads, network.trainable_variables))

    # 获取Metrics维护的loss结果
    # 然后清除数据
    if step % 100 == 0:
        print(step, 'loss:', loss_meter.result().numpy())
        loss_meter.reset_states()

    # evaluate
    # 使用两种方法：Metrics和非Metrics来计算准确度accuracy
    if step % 500 == 0:
        total, total_correct = 0., 0
        acc_meter.reset_states()

        for step, (x, y) in enumerate(ds_val):
            x = tf.reshape(x, (-1, 28 * 28))
            out = network(x)

            pred = tf.argmax(out, axis=1)
            pred = tf.cast(pred, dtype=tf.int32)
            correct = tf.equal(pred, y)
            total_correct += tf.reduce_sum(tf.cast(correct, dtype=tf.int32)).numpy()
            total += x.shape[0]

            # 计算pred相较于y的准确度，然后记录
            # 等效于上面几行代码的效果
            acc_meter.update_state(y, pred)

        # Metrics和手动计算的结果相同
        print(step, 'Evaluate Acc:', total_correct / total, acc_meter.result().numpy())
