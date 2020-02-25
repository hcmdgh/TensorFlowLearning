from imports import *


# Top-k Accuracy
def accuracy(output, target, topk=(1,)):
    # output_shape: [b, N]
    # target_shape: [b]

    maxk = max(topk)
    N = target.shape[0]

    pred = tf.math.top_k(output, maxk).indices
    pred = tf.transpose(pred)
    target_ = tf.broadcast_to(target, pred.shape)
    correct = tf.equal(pred, target_)

    res = []
    for k in topk:
        correct_k = tf.cast(correct[:k], dtype=tf.float32)
        correct_k = tf.reduce_sum(correct_k)
        acc = float(correct_k / N)
        res.append(acc)

    return res


if __name__ == '__main__':
    # 实战
    # output_shape: [3, 4]
    output = tf.constant([
        [0.1, 0.5, 0.3, 0.1],
        [0.2, 0.2, 0.4, 0.2],
        [0.5, 0.1, 0.1, 0.3],
    ])

    # target_shape: [3]
    target = tf.constant([2, 3, 0])

    res = accuracy(output, target, (1, 2, 3, 4))
    print(res)
    ''' [0.3333333432674408, 0.6666666865348816, 0.6666666865348816, 1.0] '''

    # 以k=2为例说明：
    # 取出output每个batch的top2的下标得到：
    # 并与target进行比对
    # [1 2] VS 2 √
    # [2 0] VS 3 ×
    # [0 3] VS 0 √
    # 所以accuracy = 2 / 3
