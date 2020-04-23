import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt
import os


os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
with tf.Session() as sess:
    train_ep = 40
    part_id = 127
    feature = np.load('chr1v2_part%d.npy' % part_id, allow_pickle=True)
    feature = feature.reshape([feature.shape[0], 400, 4, 1])
    label = np.load('lablev2_part%d.npy' % part_id, allow_pickle=True)
    saver = tf.train.import_meta_graph('model2/model%d.meta' % train_ep)
    saver.restore(sess, 'model2/model%d' % train_ep)
    x = tf.get_default_graph().get_tensor_by_name('Placeholder:0')
    y = tf.get_default_graph().get_tensor_by_name('Sigmoid:0')
    y_label = tf.get_default_graph().get_tensor_by_name('Placeholder_1:0')
    y_ = sess.run(y, feed_dict={x: feature, y_label: [[None] for i in range(feature.shape[0])]})
    y_ = y_.reshape([y_.shape[0]])

    plt.figure(1)
    plt.title('label - y')
    plt.hist(label-y_, 40)
    plt.figure(2)
    plt.plot(label, y_, "o", markersize=0.5)
    # plt.plot([0,1], [0,1], '-')
    # plt.plot([0,1], [0.5, 0.5], '-')
    # plt.plot([0.5,0.5], [0, 1], '-')
    plt.legend()
    plt.show()
    
    label[label <= 0.5] = 0
    label[label > 0.5] = 2
    y_[y_ <= 0.5] = 0
    y_[y_ > 0.5] = 1

    y_ = y_ + label

    TN = np.sum(y_ == 0)
    TP = np.sum(y_ == 3)
    FN = np.sum(y_ == 2)
    FP = np.sum(y_ == 1)

    SE = TP / (TP + FN)
    SP = TN / (TN + FP)
    ACC = (TP + TN) / (TP + FN + TN + FP)

    print('TN', TN)
    print('TP', TP)
    print('FN', FN)
    print('FP', FP)
    print('SE', SE)
    print('SP', SP)
    print('ACC', ACC)
