import tensorflow as tf
import numpy as np

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer("batch_size", 100, "每批次读取的数据大小")
tf.app.flags.DEFINE_integer("image_label", 4, "每个样本的标签数量")
tf.app.flags.DEFINE_integer("label_size", 26, "每个标签的可能性")

def read_captcha_decode():

    file_queue = tf.train.string_input_producer(["./tfrecords/captcha.tfrecords"])

    reader = tf.TFRecordReader()

    key, value = reader.read(file_queue)

    # 解析eample协议
    features = tf.parse_single_example(value, features={
        "image": tf.FixedLenFeature([], tf.string),
        "label": tf.FixedLenFeature([], tf.string),
    })

    # 对bytes类型进行解码
    # 进行图片特征值解码
    image = tf.decode_raw(features["image"], tf.uint8)

    # 进行图片的目标值解码
    label = tf.decode_raw(features["label"], tf.uint8)

    # print(image, label)

    image_reshape = tf.reshape(image, [20, 80, 3])

    label_reshape = tf.reshape(label, [4])

    # print(image_reshape, label_reshape)

    # 批处理数据
    image_batch, label_batch = tf.train.batch([image_reshape, label_reshape], batch_size=FLAGS.batch_size, num_threads=1, capacity=FLAGS.batch_size)

    return image_batch, label_batch


# 初始化权重参数
def init_weight(shape):
    w = tf.Variable(tf.random_normal(shape=shape, mean=0.0, stddev=1.0))
    return w


# 初始化偏置
def init_bias(shape):
    b = tf.Variable(tf.constant(0.0, shape=shape))
    return b


def one_hot(label_batch):
    """
    处理数据dao onehot编码
    :param label_batch:
    :return: label
    """
    arr = []

    # 循环取出每一个样本的目标值列表
    for i in range(FLAGS.batch_size):

        every_label = tf.one_hot(label_batch[i].eval(), depth=FLAGS.label_size)

        print(every_label)

        arr.append(every_label.eval())

    # 将arr处理成tensor类型
    arr = np.array(arr)

    label = tf.convert_to_tensor(arr)

    return label


def model(image_batch):

    # 卷积层 卷积：，32filter,3*3*3，strdes1, padding 激活，池化 2*2, strides2
    with tf.variable_scope("conv1"):

        w_conv1 = init_weight([3, 3, 3, 32])

        b_conv1 = init_bias([32])

        # 把tf.uint8转换成计算类型tf.flot32  [100, 20, 80, 3]
        image = tf.cast(image_batch, tf.float32)

        x_relu1 = tf.nn.relu(tf.nn.conv2d(image, w_conv1, strides=[1, 1, 1, 1], padding="SAME") + b_conv1)

        x_pool1 = tf.nn.max_pool(x_relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    # 卷积层 卷积：，64filter,3*3*32，strdes 1,padding 激活，池化 2*2, strides2
    with tf.variable_scope("conv2"):

        w_conv2 = init_weight([3, 3, 32, 64])

        b_conv2 = init_bias([64])

        # 进行卷积计算， 激活  [100, 10, 40, 32] ---> [100, 10, 40, 64]
        x_relu2 = tf.nn.relu(tf.nn.conv2d(x_pool1, w_conv2, strides=[1, 1, 1, 1], padding="SAME") + b_conv2)

        # 进行池化层 [100, 10, 40, 64]----->[100, 5, 20, 64]
        x_pool2 = tf.nn.max_pool(x_relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    # 全连接层
    with tf.variable_scope("fc"):
        # 初始化权重，偏置
        w_fc = init_weight([5 * 20 * 64, 4 * 26])

        b_fc = init_bias([4 * 26])

        # 将4-D的形状改成二维形状，在进行矩阵运算
        x_fc = tf.reshape(x_pool2, [100, 5 * 20 * 64])

        # 进行全连接计算
        y_predict = tf.matmul(x_fc, w_fc) + b_fc

    return y_predict


if __name__ == "__main__":

    image_batch, label_batch = read_captcha_decode()

    with tf.Session() as sess:

        # 回收线程的管理器
        coord = tf.train.Coordinator()

        threads = tf.train.start_queue_runners(sess, coord=coord)

        # 进行处理目标值的one_hot编码
        # [[15, 25, 14, 14], [23, 21, 10, 11]]--- >[   [[],[],[],[]] , [[],[],[],[]]  ]
        y_true = one_hot(label_batch)

        # print(sess.run([image_batch, label]))

        # 建立卷积神经网络模型
        y_predict = model(image_batch)

        # y_true [100, 4, 26]     y_predict [100, 4*26]
        # scoftmax计算概率值，计算预测值与真实值之间的交叉熵损失，平均值
        with tf.variable_scope("compute"):

            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.reshape(y_true, [100, 4 * 26]), logits=y_predict))

        # 梯度下降优化
        with tf.variable_scope("optimizer"):

            train_op = tf.train.GradientDescentOptimizer(0.0001).minimize(loss)

        # 计算准确率
        # [100, 4*26] -->[100, 4, 26]
        with tf.variable_scope("acc"):
            # [None, 10]
            equal_list = tf.equal(tf.argmax(y_true, 2), tf.argmax(tf.reshape(y_predict, [100, 4, 26]), 2))

            accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))

        # 初始化变量
        sess.run(tf.global_variables_initializer())

        # 指定训练步数去训练
        for i in range(3000):

            sess.run(train_op)

            # 打印准确率
            print("第%d步准确率为：%f" % (i, sess.run(accuracy)))

        # 回收线程
        coord.request_stop()

        coord.join(threads)



