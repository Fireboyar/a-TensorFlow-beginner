import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("max_step", 2000, "训练迭代次数")


# 初始化权重参数
def init_weight(shape):
    w = tf.Variable(tf.random_normal(shape=shape, mean=0.0, stddev=1.0))
    return w


# 初始化偏置
def init_bias(shape):
    b = tf.Variable(tf.constant(0.0, shape=shape))
    return b


def model():
    """
    自定义的卷积神经网络识别
    :return:
    """
    # 准备数据占位符 x [None, 784] , y_true [None, 10]
    with tf.variable_scope("data"):

        x = tf.placeholder(tf.float32, [None, 784])

        y_true = tf.placeholder(tf.int32, [None, 10])

    # 一、卷积层 卷积：，32filter,5*5*1，strdes1,padding 激活，池化 2*2, strides2
    with tf.variable_scope("conv1"):

        # 初始化参数卷积
        w_conv1 = init_weight([5, 5, 1, 32])

        b_conv1 = init_bias([32])

        # 对输入的图片数据进行形状的改变
        x_reshape = tf.reshape(x, [-1, 28, 28, 1])

        # 进行卷积计算， 激活  [None, 28, 28, 1] ---> [None, 28, 28, 32]
        x_relu1 = tf.nn.relu(tf.nn.conv2d(x_reshape, w_conv1, strides=[1, 1, 1, 1], padding="SAME") + b_conv1)

        # 进行池化层[None, 28, 28, 32] --->[None, 14, 14, 32]
        x_pool1 = tf.nn.max_pool(x_relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    # 二、卷积层 卷积：，64filter,5*5*32，strdes1,padding 激活，池化 2*2, strides2
    with tf.variable_scope("conv2"):

        # 初始化参数卷积
        w_conv2 = init_weight([5, 5, 32, 64])

        b_conv2 = init_bias([64])

        # 进行卷积计算， 激活  [None, 14, 14, 32] ---> [None, 14, 14, 64]
        x_relu2 = tf.nn.relu(tf.nn.conv2d(x_pool1, w_conv2, strides=[1, 1, 1, 1], padding="SAME") + b_conv2)

        # 进行池化层 [None, 14, 14, 64]----->[None, 7, 7, 64]
        x_pool2 = tf.nn.max_pool(x_relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    # 三、全连接层 [None, 7*7*64] * [7*7*64, 10] +[10] = [None, 10]
    with tf.variable_scope("fc"):

        # 初始化权重，偏置
        w_fc = init_weight([7 * 7 * 64, 10])

        b_fc = init_bias([10])

        # 将4-D的形状改成二维形状，在进行矩阵运算
        x_fc = tf.reshape(x_pool2, [-1, 7 * 7 * 64])

        # 进行全连接计算
        y_predict = tf.matmul(x_fc, w_fc) + b_fc

    return x, y_true, y_predict


def main(argv):

    mnist = input_data.read_data_sets("./data/mnist/input_data/", one_hot=True)

    # 准备数据和模型
    x, y_true, y_predict = model()

    # scoftmax计算概率值，计算预测值与真实值之间的交叉熵损失，平均值
    with tf.variable_scope("compute"):

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_predict))

    # 梯度下降优化
    with tf.variable_scope("optimizer"):

        train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    # 计算准确率
    with tf.variable_scope("acc"):

        equal_list = tf.equal(tf.argmax(y_true, 1), tf.argmax(y_predict, 1))

        accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))

    init_op = tf.global_variables_initializer()

    # 会话运行训练
    with tf.Session() as sess:

        sess.run(init_op)

        # 循环步数训练
        for i in range(FLAGS.max_step):

            # [50, 784]    [50, 10]
            mnist_x, mnist_y = mnist.train.next_batch(50)

            sess.run(train_op, feed_dict={x: mnist_x, y_true: mnist_y})

            # 打印准确率
            print("第%d步准确率为：%f" % (i, sess.run(accuracy, feed_dict={x: mnist_x, y_true: mnist_y})))


if __name__ == "__main__":
    tf.app.run()