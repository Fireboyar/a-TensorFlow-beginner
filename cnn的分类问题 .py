from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import os



mnist = input_data.read_data_sets("D:\\tmp\\tensorflow\mnist\input_data", one_hot=True)

# 定义回归模型
x = tf.placeholder(tf.float32,[None, 784])
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.matmul(x, w) + b

# 构建回归模型
# 定义损失函数和优化器
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))

# 采用sgd作为优化器
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)




# 存储模型
ckpt_dir = "D:\\tmp\\tensorflow\mnist\logs\mnist_with_summaries"
if  not os.path.exists(ckpt_dir):
    os.mkdir(ckpt_dir)


# 定义一个计数器，为训练轮数计算
globals_step = tf.Variable(0, name='global_step', trainable=False)

saver = tf.train.Saver()
non_storable_variable = tf.Variable(777)




# 训练模型
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

start = globals_step.eval()
print("start from:", start)


for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_ : mnist.test.labels}))


    globals_step.assign(i).eval()
    saver.save(sess, ckpt_dir + "/model.ckpt", global_step= globals_step)













