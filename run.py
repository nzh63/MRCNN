import gc
import numpy as np
import tensorflow.compat.v1 as tf


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def net_MRCNN(x, y):
    W_conv1 = weight_variable([1, 4, 1, 16])
    b_conv1 = bias_variable([16])
    h_conv1 = tf.nn.conv2d(x, W_conv1, strides=[
                           1, 1, 4, 1], padding='VALID') + b_conv1
    h_conv1 = tf.reshape(h_conv1, [-1, 20, 20, 16])
    print(h_conv1)

    W_conv2 = weight_variable([3, 3, 16, 32])
    b_conv2 = bias_variable([32])
    h_conv2 = tf.nn.relu(tf.nn.conv2d(h_conv1, W_conv2, strides=[
                         1, 1, 1, 1], padding='VALID') + b_conv2)
    h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 3, 3, 1], strides=[
                             1, 3, 3, 1], padding='VALID')
    print(h_pool2)

    W_conv3 = weight_variable([3, 3, 32, 48])
    b_conv3 = bias_variable([48])
    h_conv3 = tf.nn.conv2d(h_pool2, W_conv3, strides=[
                           1, 1, 1, 1], padding='VALID') + b_conv3
    W_conv4 = weight_variable([3, 3, 48, 64])
    b_conv4 = bias_variable([64])
    h_conv4 = tf.nn.conv2d(h_conv3, W_conv4, strides=[
                           1, 1, 1, 1], padding='VALID') + b_conv4

    W_fc1 = weight_variable([2*2*64, 80])
    b_fc1 = bias_variable([80])
    h_pool4 = tf.reshape(h_conv4, [-1, 2*2*64])
    h_fc1 = tf.matmul(h_pool4, W_fc1) + b_fc1
    h_fc1_drop = tf.nn.dropout(h_fc1, 0.5)
    print(h_fc1_drop)

    W_fc2 = weight_variable([80, 1])
    b_fc2 = bias_variable([1])
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    print(y_conv)
    loss = tf.reduce_mean(tf.reshape(tf.square(y - y_conv), [-1]))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

    return train_step, loss


def next_batch(data1, data2, num, batch_size):
    idx = np.arange(0, len(data1))
    np.random.shuffle(idx)
    data_shuffle1 = [data1[i] for i in idx]
    data_shuffle1 = np.asarray(data_shuffle1)
    data_shuffle2 = [data2[i] for i in idx]
    data_shuffle2 = np.asarray(data_shuffle2)
    for batch_i in range(num // batch_size):
        yield data_shuffle1[batch_i * batch_size: (batch_i + 1) * batch_size], data_shuffle2[batch_i * batch_size: (batch_i + 1) * batch_size]


batch_size = 64
with tf.Session() as sess:
    x = tf.placeholder("float", shape=[None, 400, 4, 1])
    y = tf.placeholder("float", shape=[None, 1])
    train_step, loss = net_MRCNN(x, y)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=50)
    feature = []
    label = []

    # 读取npy数据
    for i in range(1, 26):
        a = np.load('chr1v2_part%d.npy' % i, allow_pickle=True)
        a = a.reshape([a.shape[0], 400, 4, 1])
        feature.append(a)
        a = np.load('lablev2_part%d.npy' % i, allow_pickle=True)
        # a[a <= 0.5] = 0.01
        # a[a > 0.5] = 0.99
        a = a.reshape([a.shape[0], 1])
        label.append(a)

    # 训练
    for i in range(200):
        train_loss = []
        verification_loss = []
        # 训练集
        for j in range(0, 20):
            for f, l in next_batch(feature[j], label[j], feature[j].shape[0], batch_size):
                t_loss, _ = sess.run([loss, train_step], feed_dict={x: f, y: l})
                train_loss.append(t_loss)
        # 验证集
        for j in range(20, 25):
            for f, l in next_batch(feature[j], label[j], feature[j].shape[0], batch_size):
                verification_loss.append(loss.eval(
                    feed_dict={x: f, y: l}))
        print(i, np.mean(train_loss), np.mean(verification_loss))
        if i % 5 == 0:
            saver.save(sess, "model3/model%d" % i)
