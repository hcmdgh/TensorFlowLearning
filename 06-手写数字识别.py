import os

# 清除不必要的输出
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets

(x, y), (x_val, y_val) = datasets.mnist.load_data()
x = tf.convert_to_tensor(x, dtype=tf.float32) / 255.
y = tf.convert_to_tensor(y, dtype=tf.int32)

# 将张量转换为one-hot编码
y = tf.one_hot(y, depth=10)
print(x.shape)  # (60000, 28, 28)
print(y.shape)  # (60000, 10)

train_dataset = tf.data.Dataset.from_tensor_slices((x, y))
train_dataset = train_dataset.batch(200)

model = keras.Sequential(
    [
        layers.Dense(512, activation="relu"),
        layers.Dense(256, activation="relu"),
        layers.Dense(10),
    ]
)

optimizer = optimizers.SGD(learning_rate=0.001)


def train_epoch(epoch):
    for step, (x, y) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            # [b, 28, 28] => [b, 784]
            x = tf.reshape(x, (-1, 784))

            # 计算output
            # [b, 784] => [b, 10]
            out = model(x)

            # 计算loss
            loss = tf.reduce_sum(tf.square(out - y)) / x.shape[0]

        # 优化并更新w1, w2, w3, b1, b2, b3
        grads = tape.gradient(loss, model.trainable_variables)
        # w' = w - lr * grad
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if step % 99 == 0:
            print(epoch, step, "loss:", loss.numpy())


if __name__ == '__main__':
    for epoch in range(30):
        train_epoch(epoch)

'''
运行结果：
(60000, 28, 28)
(60000, 10)
0 0 loss: 1.9427952
0 99 loss: 0.9489237
0 198 loss: 0.8204551
0 297 loss: 0.5992148
1 0 loss: 0.6933165
1 99 loss: 0.66119885
1 198 loss: 0.66197187
1 297 loss: 0.4650647
2 0 loss: 0.56663835
2 99 loss: 0.5717135
2 198 loss: 0.59216046
2 297 loss: 0.40554875
3 0 loss: 0.5054202
3 99 loss: 0.5223989
3 198 loss: 0.549177
3 297 loss: 0.36987087
4 0 loss: 0.46695146
4 99 loss: 0.48866454
4 198 loss: 0.51815295
4 297 loss: 0.34500998
5 0 loss: 0.43938935
5 99 loss: 0.4630725
5 198 loss: 0.49426347
5 297 loss: 0.32629138
6 0 loss: 0.41785565
6 99 loss: 0.44239652
6 198 loss: 0.47452083
6 297 loss: 0.311254
7 0 loss: 0.4002349
7 99 loss: 0.42521423
7 198 loss: 0.45781717
7 297 loss: 0.298826
8 0 loss: 0.38550422
8 99 loss: 0.4104765
8 198 loss: 0.44319534
8 297 loss: 0.2882716
9 0 loss: 0.37280056
9 99 loss: 0.39756644
9 198 loss: 0.43012097
9 297 loss: 0.27922007
10 0 loss: 0.36155906
10 99 loss: 0.3861824
10 198 loss: 0.4182988
10 297 loss: 0.27110744
11 0 loss: 0.35150287
11 99 loss: 0.3757983
11 198 loss: 0.40758383
11 297 loss: 0.2638176
12 0 loss: 0.34256953
12 99 loss: 0.36633292
12 198 loss: 0.39774266
12 297 loss: 0.25728923
13 0 loss: 0.3344528
13 99 loss: 0.35764092
13 198 loss: 0.38865036
13 297 loss: 0.25140312
14 0 loss: 0.32704926
14 99 loss: 0.34959236
14 198 loss: 0.38027692
14 297 loss: 0.2459471
15 0 loss: 0.32022342
15 99 loss: 0.34217513
15 198 loss: 0.37242195
15 297 loss: 0.24088418
16 0 loss: 0.31396425
16 99 loss: 0.33531928
16 198 loss: 0.365142
16 297 loss: 0.23622131
17 0 loss: 0.30813968
17 99 loss: 0.32895118
17 198 loss: 0.35831946
17 297 loss: 0.2318518
18 0 loss: 0.30263457
18 99 loss: 0.322976
18 198 loss: 0.3519615
18 297 loss: 0.22775929
19 0 loss: 0.29741836
19 99 loss: 0.31735855
19 198 loss: 0.34598082
19 297 loss: 0.22381787
20 0 loss: 0.29253083
20 99 loss: 0.31198192
20 198 loss: 0.3403627
20 297 loss: 0.22005539
21 0 loss: 0.2879795
21 99 loss: 0.30689004
21 198 loss: 0.3350758
21 297 loss: 0.21651638
22 0 loss: 0.2836554
22 99 loss: 0.30208993
22 198 loss: 0.33004677
22 297 loss: 0.21318223
23 0 loss: 0.27956322
23 99 loss: 0.29759112
23 198 loss: 0.32527757
23 297 loss: 0.20995507
24 0 loss: 0.27567226
24 99 loss: 0.29332554
24 198 loss: 0.32075456
24 297 loss: 0.20689078
25 0 loss: 0.27201143
25 99 loss: 0.28931895
25 198 loss: 0.31645742
25 297 loss: 0.20394886
26 0 loss: 0.2684835
26 99 loss: 0.28549367
26 198 loss: 0.31235754
26 297 loss: 0.20114964
27 0 loss: 0.26510993
27 99 loss: 0.28184238
27 198 loss: 0.3084033
27 297 loss: 0.19848865
28 0 loss: 0.2619337
28 99 loss: 0.27837008
28 198 loss: 0.30461788
28 297 loss: 0.1959042
29 0 loss: 0.2589062
29 99 loss: 0.27503926
29 198 loss: 0.3010289
29 297 loss: 0.19343865
'''