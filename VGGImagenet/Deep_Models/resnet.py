import numpy as np
import tensorflow as tf
import sys
sys.path.append('/home/yang/open-convnet-black-box/VGGImagenet/')
tf.set_random_seed(1234)

class Resnet(object):

    def __init__(self, act_type='relu', pool_type='maxpool', res_blocks=5, reuse=False, num_labels=1000):

        """
        Construct a Resnet object.
        Total layers = 1 + 2n + 2n + 2n +1 = 6n + 2
        Notice: you will need to prepare the weights separately. Either init randomly or load in pre-trained weights.
        :param weights: the path to the trained weights
        :param plain_init: init with random or trained weights
        :param sess: ...
        :param act_type: the activation function (default Relu)
        :param pool_type: the pooling function (default Maxpool)
        :param res_blocks: the number of residual blocks (default 5)
        :param resue: To build a train graph, reuse = False. To build a validation graph resue = True (default True)
        """

        self.act_type = act_type
        self.pool_type = pool_type
        self.res_blocks = res_blocks
        self.reuse = reuse
        self.num_labels = num_labels

        self.layers_dic = {}

        # Placeholders
        with tf.name_scope('input') as scope:
            self.images = tf.placeholder(tf.float32, [None, 224, 224, 3])
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            self.imgs = self.images - mean
            self.layers_dic['imgs'] = self.imgs

        with tf.name_scope('output') as scope:
            self.labels = tf.placeholder(tf.float32, [None, self.num_labels])

        # Build the TF computational graph for the ResNet architecture
        self.logits = self.build()
        self.probs = tf.nn.softmax(self.logits)
        self.cost = tf.reduce_sum((self.probs - self.labels) ** 2)
        self.maxlogit = tf.reduce_max(self.logits, axis=1)

    def build(self):

        # we are stacking the layers
        # and thus we need an easy reference to the last layer of the current graph
        last_layer = self.imgs # starting with the input image of course

        with tf.variable_scope('conv0', reuse=self.reuse):

            conv0 = self.conv_bn_relu_layer(last_layer, [3, 3, 3, 16], 1)

            self.layers_dic['conv0'] = conv0
            last_layer = conv0

        for i in range(self.res_blocks):

            # notice that for each residual block
            # we actually have two layers in it

            name = 'conv1_%d' % i

            with tf.variable_scope(name, reuse=self.reuse):

                if i == 0:
                    conv1 = self.residual_block(last_layer, 16, first_block=True)
                else:
                    conv1 = self.residual_block(last_layer, 16)

                self.layers_dic[name] = conv1
                last_layer = conv1

        for i in range(self.res_blocks):

            # notice that for each residual block
            # we actually have two layers in it

            name = 'conv2_%d' % i

            with tf.variable_scope(name, reuse=self.reuse):

                conv2 = self.residual_block(last_layer, 32)

                self.layers_dic[name] = conv2
                last_layer = conv2

        for i in range(self.res_blocks):

            # notice that for each residual block
            # we actually have two layers in it

            name = 'conv3_%d' % i

            with tf.variable_scope(name, reuse=self.reuse):

                conv3 = self.residual_block(last_layer, 64)

                self.layers_dic[name] = conv3
                last_layer = conv3

        with tf.variable_scope('fc', reuse=self.reuse):

            num_channels = last_layer.get_shape().as_list()[-1]

            bn_layer = self.batch_normalization_layer(last_layer, num_channels) # batch normalization

            relu_layer = tf.nn.relu(bn_layer)

            # print(relu_layer.get_shape().as_list())

            global_pool = tf.reduce_mean(relu_layer, [1, 2])

            # print(global_pool.get_shape().as_list())

            input_dim = global_pool.get_shape().as_list()[-1]

            fc_w = tf.Variable(tf.truncated_normal([input_dim, self.num_labels],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='fc_w')

            fc_b = tf.Variable(tf.constant(1.0, shape=[self.num_labels], dtype=tf.float32), name='fc_b')

            fc_h = tf.matmul(global_pool, fc_w) + fc_b

            self.layers_dic['fc'] = fc_h
            last_layer = fc_h # this is the logits

        return last_layer

    def batch_normalization_layer(self, input_layer, dimension):

        '''
        Helper: batch normalziation
        :param input_layer: 4D tensor
        :param dimension: input_layer.get_shape().as_list()[-1]. The depth of the 4D tensor
        :return: the 4D tensor after being normalized
        '''

        mean, variance = tf.nn.moments(input_layer, axes=[0, 1, 2])

        beta = tf.get_variable('beta', dimension, tf.float32,
                               initializer=tf.constant_initializer(0.0, tf.float32))

        gamma = tf.get_variable('gamma', dimension, tf.float32,
                                initializer=tf.constant_initializer(1.0, tf.float32))

        bn_layer = tf.nn.batch_normalization(input_layer, mean, variance, beta, gamma, 1e-3)

        return bn_layer

    def conv_bn_relu_layer(self, input_layer, filter_shape, stride):
        '''
        Helper: conv, batch normalize and relu the input tensor sequentially
        :param input_layer: 4D tensor
        :param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
        :param stride: stride size for conv
        :return: 4D tensor. Y = Relu(batch_normalize(conv(X)))
        '''

        out_channel = filter_shape[-1]
        filter = tf.Variable(tf.truncated_normal(shape=filter_shape, dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv_layer = tf.nn.conv2d(input_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
        bn_layer = self.batch_normalization_layer(conv_layer, out_channel)
        output = tf.nn.relu(bn_layer)
        return output

    def bn_relu_conv_layer(self, input_layer, filter_shape, stride):

        '''
        Helper: batch normalize, relu and conv the input layer sequentially
        :param input_layer: 4D tensor
        :param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
        :param stride: stride size for conv
        :return: 4D tensor. Y = conv(Relu(batch_normalize(X)))
        '''

        in_channel = input_layer.get_shape().as_list()[-1]
        bn_layer = self.batch_normalization_layer(input_layer, in_channel)
        relu_layer = tf.nn.relu(bn_layer)
        filter = tf.Variable(tf.truncated_normal(shape=filter_shape, dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv_layer = tf.nn.conv2d(relu_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
        return conv_layer

    def residual_block(self, input_layer, output_channel, first_block=False):

        '''
        A Residual Block
        :param input_layer: 4D tensor
        :param output_channel: int. return_tensor.get_shape().as_list()[-1] = output_channel
        :param first_block: if this is the first residual block of the whole network
        :return: 4D tensor.
        '''

        input_channel = input_layer.get_shape().as_list()[-1]

        # when it's time to "shrink" the image size (and double the number of filters), we use stride = 2
        # so no pooling layers
        if input_channel * 2 == output_channel:
            increase_dim = True
            stride = 2
        elif input_channel == output_channel:
            increase_dim = False
            stride = 1
        else:
            raise ValueError('Output and input channel does not match in residual blocks!!!')

        # The first conv layer of the first residual block does not need to be normalized and relu-ed.
        with tf.variable_scope('conv1_in_block'):
            if first_block:
                filter = tf.Variable(tf.truncated_normal(shape=[3, 3, input_channel, output_channel], dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
                conv1 = tf.nn.conv2d(input_layer, filter=filter, strides=[1, 1, 1, 1], padding='SAME')
            else:
                conv1 = self.bn_relu_conv_layer(input_layer, [3, 3, input_channel, output_channel], stride)

        with tf.variable_scope('conv2_in_block'):
            conv2 = self.bn_relu_conv_layer(conv1, [3, 3, output_channel, output_channel], 1)

        # When the channels of input layer and conv2 does not match, we add zero pads to increase the
        #  depth of input layers
        if increase_dim is True:
            pooled_input = tf.nn.avg_pool(input_layer, ksize=[1, 2, 2, 1],
                                          strides=[1, 2, 2, 1], padding='VALID')
            padded_input = tf.pad(pooled_input, [[0, 0], [0, 0], [0, 0], [input_channel // 2,
                                                                          input_channel // 2]])
        else:
            padded_input = input_layer

        output = conv2 + padded_input
        return output

    def init(self, sess):
        sess.run(tf.global_variables_initializer())