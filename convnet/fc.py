import numpy as np
import tensorflow as tf


class FC(object):

    def __init__(self, sess=None):

        self.layers_dic = {}
        self.parameters = []

        self.n_labels = 5
        self.n_input = 64

        # zero-mean input
        with tf.name_scope('input') as scope:
            self.images = tf.placeholder(tf.float32, [None, self.n_input, self.n_input, 3])
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            self.imgs = self.images - mean
            self.layers_dic['imgs'] = self.imgs

        with tf.name_scope('output') as scope:
            self.labels = tf.placeholder(tf.float32, [None, self.n_labels])

        self.convlayers()
        self.fc_layers()

        self.logits = self.fc3l

        self.probs = tf.nn.softmax(self.logits)
        self.cost = tf.reduce_sum((self.probs - self.labels) ** 2)
        self.maxlogit = tf.reduce_max(self.logits, axis=1)

        if sess is not None:
            self.init(sess)
        else:
            print("convnet initialization failed ... ")

    def convlayers(self):

        conv_ch1 = 256

        # # conv1_1
        # with tf.name_scope('conv1_1') as scope:
        #     # kernel = tf.Variable(tf.constant(1.0, shape=[7, 7, 3, conv_ch1], dtype=tf.float32), name='weights')
        #     kernel = tf.Variable(tf.truncated_normal([7, 7, 3, conv_ch1], dtype=tf.float32,
        #                                              stddev=1e-1), name='weights')
        #     conv = tf.nn.conv2d(self.imgs, kernel, [1, 1, 1, 1], padding='SAME')
        #     biases = tf.Variable(tf.constant(0.0, shape=[conv_ch1], dtype=tf.float32),
        #                          trainable=True, name='biases')
        #     out = tf.nn.bias_add(conv, biases)
        #     self.conv1_1 = tf.nn.relu(out, name=scope)
        #
        #     self.parameters += [kernel, biases]
        #     self.layers_dic['conv1_1'] = self.conv1_1

        # # conv1_2
        # with tf.name_scope('conv1_2') as scope:
        #     kernel = tf.Variable(tf.truncated_normal([3, 3, conv_ch1, conv_ch1], dtype=tf.float32,
        #                                              stddev=1e-1), name='weights')
        #     conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 2, 2, 1], padding='SAME')
        #     biases = tf.Variable(tf.constant(0.0, shape=[conv_ch1], dtype=tf.float32),
        #                          trainable=True, name='biases')
        #     out = tf.nn.bias_add(conv, biases)
        #     self.conv1_2 = tf.nn.relu(out, name=scope)
        #
        #     self.parameters += [kernel, biases]
        #     self.layers_dic['conv1_2'] = self.conv1_2
        #
        # # pool1
        # self.pool1 = tf.nn.max_pool(self.conv1_2,
        #                        ksize=[1, 2, 2, 1],
        #                        strides=[1, 2, 2, 1],
        #                        padding='SAME',
        #                        name='pool1')
        # self.layers_dic['pool1'] = self.pool1
        #
        # # conv2_1
        # with tf.name_scope('conv2_1') as scope:
        #     kernel = tf.Variable(tf.truncated_normal([9, 9, conv_ch1, conv_ch1*2], dtype=tf.float32,
        #                                              stddev=1e-1), name='weights')
        #     conv = tf.nn.conv2d(self.pool1, kernel, [1, 2, 2, 1], padding='SAME')
        #     biases = tf.Variable(tf.constant(0.0, shape=[conv_ch1*2], dtype=tf.float32),
        #                          trainable=True, name='biases')
        #     out = tf.nn.bias_add(conv, biases)
        #     self.conv2_1 = tf.nn.relu(out, name=scope)
        #
        #     self.parameters += [kernel, biases]
        #     self.layers_dic['conv2_1'] = self.conv2_1
        #
        # # conv2_2
        # with tf.name_scope('conv2_2') as scope:
        #     kernel = tf.Variable(tf.truncated_normal([3, 3, conv_ch1*2, conv_ch1*2], dtype=tf.float32,
        #                                              stddev=1e-1), name='weights')
        #     conv = tf.nn.conv2d(self.conv2_1, kernel, [1, 2, 2, 1], padding='SAME')
        #     biases = tf.Variable(tf.constant(0.0, shape=[conv_ch1*2], dtype=tf.float32),
        #                          trainable=True, name='biases')
        #     out = tf.nn.bias_add(conv, biases)
        #     self.conv2_2 = tf.nn.relu(out, name=scope)
        #
        #     self.parameters += [kernel, biases]
        #     self.layers_dic['conv2_2'] = self.conv2_2
        #
        # # pool2
        # self.pool2 = tf.nn.max_pool(self.conv2_2,
        #                        ksize=[1, 2, 2, 1],
        #                        strides=[1, 2, 2, 1],
        #                        padding='SAME',
        #                        name='pool2')
        # self.layers_dic['pool2'] = self.pool2
        #
        # # conv3_1
        # with tf.name_scope('conv3_1') as scope:
        #     kernel = tf.Variable(tf.truncated_normal([3, 3, conv_ch1*2, conv_ch1*4], dtype=tf.float32,
        #                                              stddev=1e-1), name='weights')
        #     conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')
        #     biases = tf.Variable(tf.constant(0.0, shape=[conv_ch1*4], dtype=tf.float32),
        #                          trainable=True, name='biases')
        #     out = tf.nn.bias_add(conv, biases)
        #     self.conv3_1 = tf.nn.relu(out, name=scope)
        #
        #     self.parameters += [kernel, biases]
        #     self.layers_dic['conv3_1'] = self.conv3_1
        #
        # # conv3_2
        # with tf.name_scope('conv3_2') as scope:
        #     kernel = tf.Variable(tf.truncated_normal([3, 3, conv_ch1*4, conv_ch1*4], dtype=tf.float32,
        #                                              stddev=1e-1), name='weights')
        #     conv = tf.nn.conv2d(self.conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
        #     biases = tf.Variable(tf.constant(0.0, shape=[conv_ch1*4], dtype=tf.float32),
        #                          trainable=True, name='biases')
        #     out = tf.nn.bias_add(conv, biases)
        #     self.conv3_2 = tf.nn.relu(out, name=scope)
        #
        #     self.parameters += [kernel, biases]
        #     self.layers_dic['conv3_2'] = self.conv3_2
        #
        # # conv3_3
        # with tf.name_scope('conv3_3') as scope:
        #     kernel = tf.Variable(tf.truncated_normal([3, 3, conv_ch1*4, conv_ch1*4], dtype=tf.float32,
        #                                              stddev=1e-1), name='weights')
        #     conv = tf.nn.conv2d(self.conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
        #     biases = tf.Variable(tf.constant(0.0, shape=[conv_ch1*4], dtype=tf.float32),
        #                          trainable=True, name='biases')
        #     out = tf.nn.bias_add(conv, biases)
        #     self.conv3_3 = tf.nn.relu(out, name=scope)
        #
        #     self.parameters += [kernel, biases]
        #     self.layers_dic['conv3_3'] = self.conv3_3
        #
        # # pool3
        # self.pool3 = tf.nn.max_pool(self.conv3_3,
        #                        ksize=[1, 2, 2, 1],
        #                        strides=[1, 2, 2, 1],
        #                        padding='SAME',
        #                        name='pool3')
        # self.layers_dic['pool3'] = self.pool3
        #
        # # conv4_1
        # with tf.name_scope('conv4_1') as scope:
        #     kernel = tf.Variable(tf.truncated_normal([3, 3, conv_ch1*4, conv_ch1*8], dtype=tf.float32,
        #                                              stddev=1e-1), name='weights')
        #     conv = tf.nn.conv2d(self.pool3, kernel, [1, 1, 1, 1], padding='SAME')
        #     biases = tf.Variable(tf.constant(0.0, shape=[conv_ch1*8], dtype=tf.float32),
        #                          trainable=True, name='biases')
        #     out = tf.nn.bias_add(conv, biases)
        #     self.conv4_1 = tf.nn.relu(out, name=scope)
        #
        #     self.parameters += [kernel, biases]
        #     self.layers_dic['conv4_1'] = self.conv4_1
        #
        # # conv4_2
        # with tf.name_scope('conv4_2') as scope:
        #     kernel = tf.Variable(tf.truncated_normal([3, 3, conv_ch1*8, conv_ch1*8], dtype=tf.float32,
        #                                              stddev=1e-1), name='weights')
        #     conv = tf.nn.conv2d(self.conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
        #     biases = tf.Variable(tf.constant(0.0, shape=[conv_ch1*8], dtype=tf.float32),
        #                          trainable=True, name='biases')
        #     out = tf.nn.bias_add(conv, biases)
        #     self.conv4_2 = tf.nn.relu(out, name=scope)
        #
        #     self.parameters += [kernel, biases]
        #     self.layers_dic['conv4_2'] = self.conv4_2
        #
        # # conv4_3
        # with tf.name_scope('conv4_3') as scope:
        #     kernel = tf.Variable(tf.truncated_normal([3, 3, conv_ch1*8, conv_ch1*8], dtype=tf.float32,
        #                                              stddev=1e-1), name='weights')
        #     conv = tf.nn.conv2d(self.conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
        #     biases = tf.Variable(tf.constant(0.0, shape=[conv_ch1*8], dtype=tf.float32),
        #                          trainable=True, name='biases')
        #     out = tf.nn.bias_add(conv, biases)
        #     self.conv4_3 = tf.nn.relu(out, name=scope)
        #
        #     self.parameters += [kernel, biases]
        #     self.layers_dic['conv4_3'] = self.conv4_3
        #
        # # pool4
        # self.pool4 = tf.nn.max_pool(self.conv4_3,
        #                        ksize=[1, 2, 2, 1],
        #                        strides=[1, 2, 2, 1],
        #                        padding='SAME',
        #                        name='pool4')
        # self.layers_dic['pool4'] = self.pool4

        # # conv5_1
        # with tf.name_scope('conv5_1') as scope:
        #     kernel = tf.Variable(tf.truncated_normal([3, 3, conv_ch1*8, conv_ch1*8], dtype=tf.float32,
        #                                              stddev=1e-1), name='weights')
        #     conv = tf.nn.conv2d(self.pool4, kernel, [1, 1, 1, 1], padding='SAME')
        #     biases = tf.Variable(tf.constant(0.0, shape=[conv_ch1*8], dtype=tf.float32),
        #                          trainable=True, name='biases')
        #     out = tf.nn.bias_add(conv, biases)
        #     self.conv5_1 = tf.nn.relu(out, name=scope)
        #
        #     self.parameters += [kernel, biases]
        #     self.layers_dic['conv5_1'] = self.conv5_1
        #
        # # conv5_2
        # with tf.name_scope('conv5_2') as scope:
        #     kernel = tf.Variable(tf.truncated_normal([3, 3, conv_ch1*8, conv_ch1*8], dtype=tf.float32,
        #                                              stddev=1e-1), name='weights')
        #     conv = tf.nn.conv2d(self.conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
        #     biases = tf.Variable(tf.constant(0.0, shape=[conv_ch1*8], dtype=tf.float32),
        #                          trainable=True, name='biases')
        #     out = tf.nn.bias_add(conv, biases)
        #     self.conv5_2 = tf.nn.relu(out, name=scope)
        #
        #     self.parameters += [kernel, biases]
        #     self.layers_dic['conv5_2'] = self.conv5_2
        #
        # # conv5_3
        # with tf.name_scope('conv5_3') as scope:
        #     kernel = tf.Variable(tf.truncated_normal([3, 3, conv_ch1*8, conv_ch1*8], dtype=tf.float32,
        #                                              stddev=1e-1), name='weights')
        #     conv = tf.nn.conv2d(self.conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
        #     biases = tf.Variable(tf.constant(0.0, shape=[conv_ch1*8], dtype=tf.float32),
        #                          trainable=True, name='biases')
        #     out = tf.nn.bias_add(conv, biases)
        #     self.conv5_3 = tf.nn.relu(out, name=scope)
        #
        #     self.parameters += [kernel, biases]
        #     self.layers_dic['conv5_3'] = self.conv5_3
        #
        # # pool5
        # self.pool5 = tf.nn.max_pool(self.conv5_3,
        #                        ksize=[1, 2, 2, 1],
        #                        strides=[1, 2, 2, 1],
        #                        padding='SAME',
        #                        name='pool5')
        # self.layers_dic['pool5'] = self.pool5

        self.convnet_out = self.imgs
        print(self.convnet_out.name)
        print(self.convnet_out.get_shape().as_list())

    def fc_layers(self):

        # fc_len = 4096
        # fc_len = 1024
        fc_len = 5000

        shape = int(np.prod(self.convnet_out.get_shape()[1:]))
        self.pool5_flat = tf.reshape(self.convnet_out, [-1, shape])

        # fc1
        with tf.name_scope('fc1') as scope:

            fc1w = tf.Variable(tf.truncated_normal([shape, fc_len],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
            fc1b = tf.Variable(tf.constant(1.0, shape=[fc_len], dtype=tf.float32),
                                 trainable=True, name='biases')
            fc1l = tf.nn.bias_add(tf.matmul(self.pool5_flat, fc1w), fc1b)
            self.fc1 = tf.nn.relu(fc1l)
            self.parameters += [fc1w, fc1b]
            self.layers_dic['fc1'] = self.fc1

        # # fc2
        # with tf.name_scope('fc2') as scope:
        #     fc2w = tf.Variable(tf.truncated_normal([fc_len, fc_len],
        #                                                  dtype=tf.float32,
        #                                                  stddev=1e-1), name='weights')
        #     fc2b = tf.Variable(tf.constant(1.0, shape=[fc_len], dtype=tf.float32),
        #                          trainable=True, name='biases')
        #     fc2l = tf.nn.bias_add(tf.matmul(self.fc1, fc2w), fc2b)
        #     self.fc2 = tf.nn.relu(fc2l)
        #
        #     self.parameters += [fc2w, fc2b]
        #     self.layers_dic['fc2'] = self.fc2

        self.before_softmax = self.fc1
        flat_len = self.pool5_flat.get_shape().as_list()[1]
        print(self.before_softmax.name)
        print(self.before_softmax.shape)

        # fc3
        with tf.name_scope('fc3') as scope:
            # fc3w = tf.Variable(tf.truncated_normal([fc_len, self.n_labels],
            #                                              dtype=tf.float32,
            #                                              stddev=1e1), name='weights')
            fc3w = tf.Variable(tf.constant(1.0, shape=[fc_len, self.n_labels], dtype=tf.float32), name='weights')
            fc3b = tf.Variable(tf.constant(0.0, shape=[self.n_labels], dtype=tf.float32),
                                 trainable=True, name='biases')
            self.fc3l = tf.nn.bias_add(tf.matmul(self.before_softmax, fc3w), fc3b)

            self.parameters += [fc3w, fc3b]
            self.layers_dic['fc3'] = self.fc3l

    def init(self, sess):
        sess.run(tf.global_variables_initializer())