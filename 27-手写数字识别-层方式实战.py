from imports import *
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics

(x, y), (x_test, y_test) = datasets.fashion_mnist.load_data()
print(x.shape, y.shape)
''' (60000, 28, 28) (60000,) '''

print(x_test.shape, y_test.shape)
''' (10000, 28, 28) (10000,) '''


def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.cast(y, dtype=tf.int32)
    return x, y


BATCH_SIZE = 128

db = tf.data.Dataset.from_tensor_slices((x, y))
db = db.map(preprocess).shuffle(60000).batch(BATCH_SIZE)

db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db_test = db_test.map(preprocess).batch(BATCH_SIZE)

db_iter = iter(db)
sample = next(db_iter)
print("batch_shape:", sample[0].shape, sample[1].shape)
''' batch_shape: (128, 28, 28) (128,) '''

# 定义网络层
model = Sequential([
    # [b, 784] => [b, 256]
    layers.Dense(256, activation=tf.nn.relu),

    # [b, 256] => [b, 128]
    layers.Dense(128, activation=tf.nn.relu),

    # [b, 128] => [b, 64]
    layers.Dense(64, activation=tf.nn.relu),

    # [b, 64] => [b, 32]
    # 2080 = 64*32 + 32
    layers.Dense(32, activation=tf.nn.relu),

    # [b, 32] => [b, 10]
    # 330 = 32*10 + 10
    layers.Dense(10),
])
model.build(input_shape=[None, 28 * 28])

# 输出网络结构
model.summary()
'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense (Dense)                multiple                  200960    
_________________________________________________________________
dense_1 (Dense)              multiple                  32896     
_________________________________________________________________
dense_2 (Dense)              multiple                  8256      
_________________________________________________________________
dense_3 (Dense)              multiple                  2080      
_________________________________________________________________
dense_4 (Dense)              multiple                  330       
=================================================================
Total params: 244,522
Trainable params: 244,522
Non-trainable params: 0
_________________________________________________________________
'''

optimizer = optimizers.Adam(lr=1e-3)


def main():
    for epoch in range(30):
        for step, (x, y) in enumerate(db):
            # x_shape: [b, 28, 28] => [b, 784]
            # y_shape: [b]
            x = tf.reshape(x, [-1, 28 * 28])

            with tf.GradientTape() as tape:
                # [b, 784] => [b, 10]
                logits = model(x)
                y_onehot = tf.one_hot(y, depth=10)

                loss_mse = tf.reduce_mean(tf.losses.MSE(y_onehot, logits))
                loss_ce = tf.reduce_mean(tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True))

            grads = tape.gradient(loss_ce, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if (step + 1) % 100 == 0:
                print(f"epoch: {epoch}, step: {step}, loss_ce: {float(loss_ce)}, loss_mse: {float(loss_mse)}")

        # test

        total_correct = 0
        total_num = 0
        for x, y in db_test:
            x = tf.reshape(x, [-1, 28 * 28])
            logits = model(x)

            # logits => prob
            prob = tf.nn.softmax(logits, axis=1)

            pred = tf.argmax(prob, axis=1)
            pred = tf.cast(pred, dtype=tf.int32)
            correct = tf.equal(pred, y)
            correct = tf.reduce_sum(tf.cast(correct, dtype=tf.int32))

            total_correct += int(correct)
            total_num += x.shape[0]

        acc = total_correct / total_num
        print(f"epoch: {epoch}, test acc: {acc}")


if __name__ == '__main__':
    main()
