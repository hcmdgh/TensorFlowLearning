from imports import *
from tensorflow import keras

# 1. boston housing（波斯顿的房价）


# 2. mnist（手写数字）
def mnist():
    (x, y), (x_test, y_test) = keras.datasets.mnist.load_data()
    print(type(x))
    ''' <class 'numpy.ndarray'> '''

    print(x.shape, y.shape)
    ''' (60000, 28, 28) (60000,) '''

    print(x_test.shape, y_test.shape)
    ''' (10000, 28, 28) (10000,) '''

    print(x.min(), x.max(), x.mean())
    ''' 0 255 33.318421449829934 '''

    print(y.min(), y.max(), y.mean())
    ''' 0 9 4.4539333333333335 '''

    # one-hot编码
    print(y[:4])
    ''' [5 0 4 1] '''

    y_onehot = tf.one_hot(y, depth=10)
    print(y_onehot[:4])
    ''' tf.Tensor(
    [[0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
     [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
     [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]], shape=(4, 10), dtype=float32) '''


# 3. cifar10/100（小型图片分类数据集）
# 使用说明："用户目录\.keras\datasets\cifar-10-batches-py\readme.html"
def cifar():
    (x, y), (x_test, y_test) = keras.datasets.cifar10.load_data()
    print(type(x))
    ''' <class 'numpy.ndarray'> '''

    print(x.shape, y.shape)
    ''' (50000, 32, 32, 3) (50000, 1) '''

    print(x_test.shape, y_test.shape)
    ''' (10000, 32, 32, 3) (10000, 1) '''

    print(x.min(), x.max(), x.mean())
    ''' 0 255 120.70756512369792 '''

    print(y.min(), y.max(), y.mean())
    ''' 0 9 4.5 '''

# 4. imdb（情感分类数据集）


# 5. tf.data.Dataset
def dataset():
    (x, y), (x_test, y_test) = keras.datasets.mnist.load_data()
    db = tf.data.Dataset.from_tensor_slices(x_test)
    print(db)
    ''' <TensorSliceDataset shapes: (28, 28), types: tf.uint8> '''

    # 5.1 使用迭代器取出每一个样本
    it = iter(db)
    sample = next(it)
    print(sample.shape)
    ''' (28, 28) '''

    samples = list(db)
    print(len(samples))
    ''' 10000 '''

    print(samples[100].shape)
    ''' (28, 28) '''

    # 5.2 将x和y合并为一个Dataset
    # Tips: 不能使用[x_test, y_test]
    db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    print(db)
    ''' <TensorSliceDataset shapes: ((28, 28), ()), types: (tf.uint8, tf.uint8)> '''

    samples = list(db)
    print(len(samples))
    ''' 10000 '''

    x, y = samples[100]
    print(x.shape, y.shape)
    ''' (28, 28) () '''

    # 5.3 shuffle
    # Tips1: buffer_size应当大于等于数据集的大小
    # Tips2: 每一次对Dataset迭代，都会执行shuffle操作
    x = np.array([
        [23, 34, 45],
        [12, 13, 14],
        [55, 56, 57],
        [97, 98, 99],
    ])
    y = np.array([2, 1, 3, 4])
    db = tf.data.Dataset.from_tensor_slices((x, y))
    print(db)
    ''' <TensorSliceDataset shapes: ((3,), ()), types: (tf.int32, tf.int32)> '''

    tf.random.set_seed(1234)
    db2 = db.shuffle(buffer_size=100)  # buffer_size应当大于等于数据集的大小
    for item in db2:
        print(item)
    '''
    (<tf.Tensor: shape=(3,), dtype=int32, numpy=array([23, 34, 45])>, <tf.Tensor: shape=(), dtype=int32, numpy=2>)
    (<tf.Tensor: shape=(3,), dtype=int32, numpy=array([97, 98, 99])>, <tf.Tensor: shape=(), dtype=int32, numpy=4>)
    (<tf.Tensor: shape=(3,), dtype=int32, numpy=array([55, 56, 57])>, <tf.Tensor: shape=(), dtype=int32, numpy=3>)
    (<tf.Tensor: shape=(3,), dtype=int32, numpy=array([12, 13, 14])>, <tf.Tensor: shape=(), dtype=int32, numpy=1>)
    '''

    # 5.4 map
    def preprocess(x, y):
        x = tf.cast(x, dtype=tf.float32) / 10.
        y = tf.one_hot(y, depth=5)
        return x, y

    db3 = db.map(preprocess)
    for item in db3:
        print(item)
    '''
    (<tf.Tensor: shape=(3,), dtype=float32, numpy=array([2.3, 3.4, 4.5], dtype=float32)>, <tf.Tensor: shape=(5,), dtype=float32, numpy=array([0., 0., 1., 0., 0.], dtype=float32)>)
    (<tf.Tensor: shape=(3,), dtype=float32, numpy=array([1.2, 1.3, 1.4], dtype=float32)>, <tf.Tensor: shape=(5,), dtype=float32, numpy=array([0., 1., 0., 0., 0.], dtype=float32)>)
    (<tf.Tensor: shape=(3,), dtype=float32, numpy=array([5.5, 5.6, 5.7], dtype=float32)>, <tf.Tensor: shape=(5,), dtype=float32, numpy=array([0., 0., 0., 1., 0.], dtype=float32)>)
    (<tf.Tensor: shape=(3,), dtype=float32, numpy=array([9.7, 9.8, 9.9], dtype=float32)>, <tf.Tensor: shape=(5,), dtype=float32, numpy=array([0., 0., 0., 0., 1.], dtype=float32)>)
    '''

    # 5.5 batch
    db4 = db3.batch(batch_size=2)
    for item in db4:
        print(item)
    '''
    (<tf.Tensor: shape=(2, 3), dtype=float32, numpy=
    array([[2.3, 3.4, 4.5],
           [1.2, 1.3, 1.4]], dtype=float32)>, <tf.Tensor: shape=(2, 5), dtype=float32, numpy=
    array([[0., 0., 1., 0., 0.],
           [0., 1., 0., 0., 0.]], dtype=float32)>)
           
    (<tf.Tensor: shape=(2, 3), dtype=float32, numpy=
    array([[5.5, 5.6, 5.7],
           [9.7, 9.8, 9.9]], dtype=float32)>, <tf.Tensor: shape=(2, 5), dtype=float32, numpy=
    array([[0., 0., 0., 1., 0.],
           [0., 0., 0., 0., 1.]], dtype=float32)>)
    '''

    # 5.6 repeat
    db = tf.data.Dataset.from_tensor_slices([
        [1, 2, 3],
        [4, 5, 6]
    ])
    for item in db:
        print(item)
    '''
    tf.Tensor([1 2 3], shape=(3,), dtype=int32)
    tf.Tensor([4 5 6], shape=(3,), dtype=int32)
    '''

    db2 = db.repeat(2)
    for item in db2:
        print(item)
    '''
    tf.Tensor([1 2 3], shape=(3,), dtype=int32)
    tf.Tensor([4 5 6], shape=(3,), dtype=int32)
    tf.Tensor([1 2 3], shape=(3,), dtype=int32)
    tf.Tensor([4 5 6], shape=(3,), dtype=int32)
    '''

    # 重复无限次
    db3 = db.repeat()
    for item in db3:
        print(item)  # 永不停止


if __name__ == '__main__':
    dataset()
