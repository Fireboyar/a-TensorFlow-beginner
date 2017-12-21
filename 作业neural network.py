import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer("max_step", 2000, "训练迭代次数")



def main(argv):

    mnist = input_data.read_data_sets("./data/mnist/input_data/", one_hot=True)

    # 准备数据占位符 x [None, 784] , y_true [None, 10]
    with tf.variable_scope("data"):

        x = tf.placeholder(tf.float32, [None, 784])

        y_true = tf.placeholder(tf.int32, [None, 10])

    # 建立单层神经网络模型，计算预测值one-hot
    with tf.variable_scope("model"):
        # 准备权重和偏置,w [784, 10], bias [10]
        weight = tf.Variable(tf.random_normal([784, 10], mean=0.0, stddev=1.0), name="weights")

        bias = tf.Variable(tf.constant(0.0, shape=[10]), name="biases")

        # 进行神经网络计算matrix:[None, 784] * [784, 10] + [10] = [None, 10]
        y_predict = tf.matmul(x, weight) + bias

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

    # 收集变量
    tf.summary.scalar("losses", loss)
    tf.summary.scalar("acc", accuracy)

    tf.summary.histogram("weights", weight)
    tf.summary.histogram("biases", bias)

    # 定义初始化变量的op
    init_op = tf.global_variables_initializer()

    # 合并变量
    merged = tf.summary.merge_all()

    # 进行训练
    with tf.Session() as sess:

        sess.run(init_op)

        # 建立事件文件
        filewriter = tf.summary.FileWriter("./tmp/summary/test/", graph=sess.graph)

        # 循环训练模型
        for i in range(FLAGS.max_step):

            # [50, 784]    [50, 10]
            mnist_x, mnist_y = mnist.train.next_batch(50)

            sess.run(train_op, feed_dict={x: mnist_x, y_true: mnist_y})

            # 运行合并变量的op
            summary = sess.run(merged, feed_dict={x: mnist_x, y_true: mnist_y})

            # 写入每批次的值
            filewriter.add_summary(summary, i)

            # 打印准确率
            print("第%d步准确率为：%f"%(i, sess.run(accuracy, feed_dict={x: mnist_x, y_true: mnist_y})))

if __name__ == "__main__":
    tf.app.run()