import numpy as np
import tensorflow as tf

class Convnet(object):

    # TODO: set the default value for mean

    def __init__(self, weights=None, mean=[123.68, 116.779, 103.939], Slu = True, num_classes=2, plain_init=None, sess=None):

        self.layers_dic = {}
        self.layers_W_dic = {}

        # zero-mean input
        with tf.name_scope('input') as scope:

            self.images = tf.placeholder(tf.float32, [None, 224, 224, 3])
            mean = tf.constant(mean, dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            self.imgs = self.images - mean
            self.layers_dic['imgs'] = self.imgs

        with tf.name_scope('output') as scope:
            self.labels = tf.placeholder(tf.float32, [None, num_classes])

        self.convlayers(Slu)

        self.fc_layers(num_classes, Slu)

        # probabilities
        self.probs = tf.nn.softmax(self.fc2l)

        # cross entropy loss
        self.cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.fc2l))

        # max logit
        self.maxlogit = tf.reduce_max(self.fc2l, axis=1)

        # accuracy
        self.correct_prediction = tf.equal(tf.argmax(self.fc2l, 1), tf.argmax(self.labels, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        # training step
        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.cost)

        if plain_init and sess is not None:
            self.init(sess)

        elif weights is not None and sess is not None:
            self.load_weights(weights, sess)

        else:
            print("vgg_shallow initialization failed ... ")

    def convlayers(self, Slu):

        num_filters = 64

        # conv1_1
        with tf.name_scope('conv1_1') as scope:

            kernel = tf.Variable(tf.truncated_normal([3, 3, 3, num_filters], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            biases = tf.Variable(tf.constant(0.1, shape=[num_filters], dtype=tf.float32),
                                 trainable=True, name='biases')

            conv = tf.nn.conv2d(self.imgs, kernel, [1, 1, 1, 1], padding='SAME')
            out = tf.nn.bias_add(conv, biases)

            # activation
            if Slu:
                self.conv1_1 = tf.nn.relu(tf.nn.sigmoid(out) - 0.5, name=scope) * 10.
            else:
                self.conv1_1 = tf.nn.relu(out, name=scope)

            self.layers_dic['conv1_1'] = self.conv1_1
            self.layers_W_dic['conv1_1'] = kernel

        # conv1_2
        with tf.name_scope('conv1_2') as scope:

            kernel = tf.Variable(tf.truncated_normal([3, 3, num_filters, num_filters], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            biases = tf.Variable(tf.constant(0.1, shape=[num_filters], dtype=tf.float32),
                                 trainable=True, name='biases')

            conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            out = tf.nn.bias_add(conv, biases)

            # activation
            if Slu:
                self.conv1_2 = tf.nn.relu(tf.nn.sigmoid(out) - 0.5, name=scope) * 10.
            else:
                self.conv1_2 = tf.nn.relu(out, name=scope)

            self.layers_dic['conv1_2'] = self.conv1_2
            self.layers_W_dic['conv1_2'] = kernel

        # pool1
        self.pool1 = tf.nn.max_pool(self.conv1_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool1')
        self.layers_dic['pool1'] = self.pool1

        # conv2_1
        with tf.name_scope('conv2_1') as scope:

            kernel = tf.Variable(tf.truncated_normal([3, 3, num_filters, num_filters], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            biases = tf.Variable(tf.constant(0.1, shape=[num_filters], dtype=tf.float32),
                                 trainable=True, name='biases')

            conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
            out = tf.nn.bias_add(conv, biases)

            # activation
            if Slu:
                self.conv2_1 = tf.nn.relu(tf.nn.sigmoid(out) - 0.5, name=scope) * 10.
            else:
                self.conv2_1 = tf.nn.relu(out, name=scope)

            self.layers_dic['conv2_1'] = self.conv2_1
            self.layers_W_dic['conv2_1'] = kernel

        # conv2_2
        with tf.name_scope('conv2_2') as scope:

            kernel = tf.Variable(tf.truncated_normal([3, 3, num_filters, num_filters], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            biases = tf.Variable(tf.constant(0.1, shape=[num_filters], dtype=tf.float32),
                                 trainable=True, name='biases')

            conv = tf.nn.conv2d(self.conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            out = tf.nn.bias_add(conv, biases)

            # activation
            if Slu:
                self.conv2_2 = tf.nn.relu(tf.nn.sigmoid(out) - 0.5, name=scope) * 10.
            else:
                self.conv2_2 = tf.nn.relu(out, name=scope)

            self.layers_dic['conv2_2'] = self.conv2_2
            self.layers_W_dic['conv2_2'] = kernel

        # pool2
        self.pool2 = tf.nn.max_pool(self.conv2_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool2')
        self.layers_dic['pool2'] = self.pool2

    def fc_layers(self, num_classes, Slu):

        num_hidden = 512

        # fc1
        with tf.name_scope('fc1') as scope:

            shape = int(np.prod(self.pool2.get_shape()[1:]))
            fc1w = tf.Variable(tf.truncated_normal([shape, num_hidden],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
            fc1b = tf.Variable(tf.constant(0.1, shape=[num_hidden], dtype=tf.float32),
                                 trainable=True, name='biases')

            flat = tf.reshape(self.pool2, [-1, shape])
            fc1l = tf.nn.bias_add(tf.matmul(flat, fc1w), fc1b)

            if Slu:
                self.fc1 = tf.nn.relu(tf.nn.sigmoid(fc1l) - 0.5, name=scope) * 10.
            else:
                self.fc1 = tf.nn.relu(fc1l)

            self.layers_dic['fc1'] = self.fc1
            self.layers_W_dic['fc1'] = fc1w

        # fc2
        with tf.name_scope('fc2') as scope:

            fc2w = tf.Variable(tf.truncated_normal([num_hidden, num_classes],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
            fc2b = tf.Variable(tf.constant(0.1, shape=[num_classes], dtype=tf.float32),
                                 trainable=True, name='biases')

            self.fc2l = tf.nn.bias_add(tf.matmul(self.fc1, fc2w), fc2b)

            self.layers_dic['fc2'] = self.fc2l
            self.layers_W_dic['fc2'] = fc2w

    def load_weights(self, weight_file, sess):
        print('Restoring the model ... ')
        saver = tf.train.Saver()
        saver.restore(sess, weight_file)

    def init(self, sess):
        sess.run(tf.global_variables_initializer())