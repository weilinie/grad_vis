########################################################################################
# Davi Frossard, 2016                                                                  #
# VGG16 implementation in TensorFlow                                                   #
# Details:                                                                             #
# http://www.cs.toronto.edu/~frossard/post/vgg16/                                      #
#                                                                                      #
# Model from https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md     #
# Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow      #
########################################################################################


import numpy as np
import tensorflow as tf


class Vgg_shallow(object):

    # TODO: set the default value for mean

    # [123.68, 116.779, 103.939]

    def __init__(self, weights=None, mean=[123.68, 116.779, 103.939], num_classes=2, plain_init=None, sess=None):

        self.layers_dic = {}
        self.parameters = []
        self.layers_W_dic = {}

        # zero-mean input
        with tf.name_scope('input') as scope:
            self.images = tf.placeholder(tf.float32, [None, 224, 224, 3])
            mean = tf.constant(mean, dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            self.imgs = self.images - mean
            self.layers_dic['imgs'] = self.imgs

        with tf.name_scope('output') as scope:
            self.labels = tf.placeholder(tf.float32, [None, num_classes])

        self.convlayers()

        self.fc_layers(num_classes)

        # probabilities
        self.probs = tf.nn.softmax(self.fc1l)

        # cross entropy loss
        self.cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.fc1l))

        # max logit
        self.maxlogit = tf.reduce_max(self.fc1l, axis=1)

        # accuracy
        self.correct_prediction = tf.equal(tf.argmax(self.fc1l, 1), tf.argmax(self.labels, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        # training step
        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.cost)

        if plain_init and sess is not None:
            self.init(sess)

        elif weights is not None and sess is not None:
            self.load_weights(weights, sess)

        else:
            print("vgg_shallow initialization failed ... ")

    def convlayers(self):

        # conv1_1
        with tf.name_scope('conv1_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([7, 7, 3, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.imgs, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_1 = tf.nn.relu(out, name=scope)

            self.parameters += [kernel, biases]
            self.layers_dic['conv1_1'] = self.conv1_1
            self.layers_W_dic['conv1_1'] = kernel

        # pool1
        self.pool1 = tf.nn.max_pool(self.conv1_1,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool1')

        self.layers_dic['pool1'] = self.pool1


    def fc_layers(self, num_classes):

        from_conv = self.pool1

        # softmax
        with tf.name_scope('fc1') as scope:

            shape = int(np.prod(from_conv.get_shape()[1:]))

            fc1w = tf.Variable(tf.truncated_normal([shape, num_classes],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='weights')

            fc1b = tf.Variable(tf.constant(1.0, shape=[num_classes], dtype=tf.float32),
                                 trainable=True, name='biases')

            flat = tf.reshape(from_conv, [-1, shape])

            self.fc1l = tf.nn.bias_add(tf.matmul(flat, fc1w), fc1b)

            self.parameters += [fc1w, fc1b]
            self.layers_dic['fc1'] = self.fc1l
            self.layers_W_dic['fc1'] = fc1w

    # def load_weights_part(self, n, weight_file, sess):
    #
    #     # load only the first n layers of weights
    #     # randomly initialize the rest
    #
    #     weights = np.load(weight_file)
    #     keys = sorted(weights.keys())
    #     for i, k in enumerate(keys):
    #         if i <= n:
    #             sess.run(self.parameters[i].assign(weights[k]))

    # def load_weights_reverse(self, n, weight_file, sess):
    #
    #     # don't load the first n layers of weights
    #     # randomly initialize them
    #
    #     weights = np.load(weight_file)
    #     keys = sorted(weights.keys())
    #     for i, k in enumerate(keys):
    #         if i > n:
    #             sess.run(self.parameters[i].assign(weights[k]))

    # def load_weights_only(self, n, weight_file, sess):
    #
    #     # don't load a specific layer of weights
    #     # randomly initialize it
    #
    #     weights = np.load(weight_file)
    #     keys = sorted(weights.keys())
    #     for i, k in enumerate(keys):
    #         if i != n and i != n - 1:
    #             sess.run(self.parameters[i].assign(weights[k]))

    def load_weights(self, weight_file, sess):
        print('Restoring the model ... ')
        saver = tf.train.Saver()
        saver.restore(sess, weight_file)

        # weights = np.load(weight_file)
        # keys = sorted(weights.keys())
        # for i, k in enumerate(keys):
        #     sess.run(self.parameters[i].assign(weights[k]))

    def init(self, sess):
        sess.run(tf.global_variables_initializer())