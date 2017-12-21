    import tensorflow as tf
    import os

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    FLAGS = tf.app.flags.FLAGS

    tf.app.flags.DEFINE_string("job_name", " ", "启动任务的类型ps or worker")
    tf.app.flags.DEFINE_integer("task_index", 0, "指定每种任务的第几天电脑")


    def main(argv):

        global_step = tf.contrib.framework.get_or_create_global_step()

        # 指定集群对象
        cluster = tf.train.ClusterSpec({"ps": ["10.211.55.3:2222"], "worker": ["192.168.35.28:2223"]})

        # 创建ps和worker的服务
        server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

        # 判断启动的任务类型，对应不同的处理方式
        if FLAGS.job_name == "ps":

            # 等待参数更新
            server.join()

        else:

            # 构造设备名称
            worker_device = "/job:worker/task:0/cpu:0"

            # 去进行运行任务，计算，初始化会话等等,如果有gpu就可以指定了
            with tf.device(tf.train.replica_device_setter(
                    worker_device=worker_device,
                    cluster=cluster
            )):

                # 进行一个乘法运算
                var1 = tf.Variable([[1, 2, 3, 4]], name="var1")

                var2 = tf.Variable([[2], [2], [2], [2]], name="var2")

                mat = tf.matmul(var1, var2)

                # 开启分布式会话运行
                with tf.train.MonitoredTrainingSession(
                        master="grpc://192.168.35.28:2223",
                        is_chief=(FLAGS.task_index == 0),
                        config=tf.ConfigProto(log_device_placement=True),
                        hooks=[tf.train.StopAtStepHook(last_step=200)]
                ) as mon_sess:
                    while not mon_sess.should_stop():
                        print(mon_sess.run(mat))


    if __name__ == "__main__":

    tf.app.run()