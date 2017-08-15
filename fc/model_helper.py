import tensorflow as tf
import numpy as np
from datetime import datetime


def forwardprop(X, w_vars, b_vars, activation, name='Forwardprop'):
    with tf.name_scope(name):
        num_layers = len(w_vars) - 1

        h_before = tf.matmul(X, w_vars[0]) + b_vars[0]
        h = activation(h_before)

        h_before_vars = [h_before]
        h_vars = [h]

        for i in range(num_layers - 1):
            h_before = tf.matmul(h, w_vars[i + 1]) + b_vars[i + 1]
            h = activation(h_before)

            h_before_vars += [h_before]
            h_vars += [h]

        yhat = tf.matmul(h, w_vars[-1]) + b_vars[-1]

    return yhat, h_vars, h_before_vars


def entropy_loss(logits, labels, name='Entropy_loss'):
    with tf.name_scope(name):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='cross_entropy')
        loss = tf.reduce_mean(cross_entropy, name='entropy_mean')

        tf.summary.scalar('loss', loss)

    return loss


def train_opt(loss, lr, opt_type='Adam', name='Train'):
    with tf.name_scope(name):
        if opt_type == 'Adam':
            train_op = tf.train.AdamOptimizer(lr).minimize(loss)
        elif opt_type == 'SGD':
            train_op = tf.train.GradientDescentOptimizer(lr).minimize(loss)
        else:
            train_op = tf.train.GradientDescentOptimizer(lr).minimize(loss)
            print("Unknown opt type! we use default SGD instead!")

    return train_op


def eval_accuracy(logits, labels, name='Eval_accu'):
    with tf.name_scope(name):
        predict = tf.argmax(logits, axis=1)
        correct = tf.argmax(labels, axis=1)
        accu = tf.reduce_mean(tf.to_float(tf.equal(correct, predict)))
        tf.summary.scalar('accu', accu)

    return accu


def placeholder_inputs(input_dim, output_size, name='Inputs'):
    with tf.name_scope(name):
        input_size = input_dim * input_dim * 3
        images_placeholder = tf.placeholder(tf.float32, shape=(None, input_size), name='X')
        labels_placeholder = tf.placeholder(tf.int32, shape=(None, output_size), name='y')

    return images_placeholder, labels_placeholder


def sparse_patterns(input_size, num_pixels, sparse_ratio, num_patterns):
    # this initial sparse vector is common for all the sparse patterns
    n_ch = 3
    sparse_vec = np.array([0. if i < (int(input_size * sparse_ratio) / n_ch) * n_ch else 1. for i in range(input_size)])

    sparse_set = []
    for k in range(num_patterns):  # generate sparse patterns one by one

        # pixel-wise sparsing matrix/pattern preparation
        # add it to sparse_set at the end
        reshaped = tf.reshape(sparse_vec, [num_pixels, 3])
        shuffled = tf.random_shuffle(reshaped)
        reshape_back = tf.reshape(shuffled, [input_size])
        sparse_set += [reshape_back]

    return tf.to_float(tf.stack(sparse_set))


def pick_sparse(sparse_set1, sparse_set2, c):
    # a function in which will randomly pick a sparse pattern from the correct set given the class

    k = sparse_set1.get_shape().as_list()[0]  # num of sparse patterns
    samples = tf.multinomial(tf.log([[1. for _ in range(k)]]), 1)
    pick = tf.cast(samples[0][0], tf.int32)
    return tf.cond(tf.equal(c, 0.0), lambda: sparse_set1[pick], lambda: sparse_set2[pick])


def preprocess(dim,
               X,
               y,
               is_perm=False,
               is_uni_sparse=False,
               sparse_ratio=0.,
               is_finite_sparse=False,
               k=1,
               name='Preprocessing'):
    input_size = dim * dim * 3
    num_pixels = dim * dim  # total num of pixels

    with tf.name_scope(name):

        # permutate the input images pixel-wisely
        # the permutation is the same for each image
        if is_perm:

            # pixel-wise permutation matrix preparation
            identity = tf.eye(input_size, dtype=tf.float32)
            reshaped = tf.reshape(identity, [num_pixels, 3, input_size])
            shuffled = tf.random_shuffle(reshaped)
            permu_matrix = tf.reshape(shuffled, [-1, input_size])

            # permute...
            images_processed = tf.matmul(X, permu_matrix, b_is_sparse=True)

            return images_processed

        # uniformly sparse each input image pixel-wisely
        # sparse ratio = num of zero pixels / num of total pixels
        elif is_uni_sparse:

            # pixel-wise sparsing matrix preparation
            sparse_set = sparse_patterns(input_size, num_pixels, sparse_ratio, 1)

            # sparsing
            images_processed = tf.map_fn(lambda x: tf.multiply(x, sparse_set[0]), X)

            return images_processed


        # for each class, define a finite set of K different sparse patterns
        # for two different classes, their sparse pattern sets are completely different
        # to sparse a image, pick one sparse pattern randomly from its own class sparse pattern set
        elif is_finite_sparse:

            # prepare the sparse pattern sets for each class
            sparse_set1 = sparse_patterns(input_size, num_pixels, sparse_ratio, k)
            sparse_set2 = sparse_patterns(input_size, num_pixels, sparse_ratio, k)

            # convert the label format
            labels = tf.to_float(tf.argmax(y, axis=1))

            # prepare a list of sparse patterns according to the labels
            indices = tf.range(tf.shape(labels)[0])
            s_patterns = tf.map_fn(lambda idx: pick_sparse(sparse_set1, sparse_set2, labels[idx]), indices, dtype=tf.float32)

            images_processed = tf.multiply(X, s_patterns)

            return images_processed

    return X


def init_weights_bias(input_dim, output_size, num_neurons=100, num_layers=1, init_std=1e-3, pb=False):
    with tf.name_scope('hidden1/'):
        input_size = input_dim * input_dim * 3
        w1_hidden = tf.Variable(tf.truncated_normal((input_size, num_neurons), stddev=init_std),
                                dtype=tf.float32, name="w1_hidden")
        tf.summary.histogram('W', w1_hidden)

    w_vars = [w1_hidden]

    for i in range(num_layers - 1):
        with tf.name_scope('hidden{}/'.format(i + 2)):
            wi_hidden = tf.Variable(tf.truncated_normal((num_neurons, num_neurons), stddev=init_std),
                                    dtype=tf.float32, name="w{}_hidden".format(i + 2))
            tf.summary.histogram('W'.format(i), wi_hidden)

        w_vars += [wi_hidden]

    with tf.name_scope('soft/'):
        w_soft = tf.Variable(tf.truncated_normal((num_neurons, output_size), stddev=init_std),
                             dtype=tf.float32, name="w_soft")
        tf.summary.histogram('W', w_soft)

    w_vars += [w_soft]

    # store the init values so that we can calculate the learned "diff" later
    w_vars_init = [tf.Variable(w_vars[i].initialized_value(), name='w_init_{}'.format(i))
                   for i in range(len(w_vars))]

    # init bias either to all zeros or to small positive constant 0.1
    if not pb:

        with tf.name_scope('hidden1/'):
            b1_hidden = tf.Variable(tf.zeros([1, num_neurons]), dtype=tf.float32, name="b1_hidden")
            tf.summary.histogram('B', b1_hidden)

        b_vars = [b1_hidden]

        for i in range(num_layers - 1):
            with tf.name_scope('hidden{}/'.format(i + 2)):
                bi_hidden = tf.Variable(tf.zeros([1, num_neurons]), dtype=tf.float32, name="b{}_hidden".format(i))
                tf.summary.histogram('B'.format(i), bi_hidden)

            b_vars += [bi_hidden]

        with tf.name_scope('soft/'):
            b_soft = tf.Variable(tf.zeros([1, output_size]), dtype=tf.float32, name="soft_bias")
            tf.summary.histogram('B', b_soft)

        b_vars += [b_soft]

    else:

        with tf.name_scope('hidden1/'):
            b1_hidden = tf.Variable(tf.constant(0.1, shape=[1, num_neurons]), dtype=tf.float32, name="b1_hidden")
            tf.summary.histogram('B', b1_hidden)

        b_vars = [b1_hidden]

        for i in range(num_layers - 1):
            with tf.name_scope('hidden{}/'.format(i + 2)):
                bi_hidden = tf.Variable(tf.constant(0.1, shape=[1, num_neurons]), dtype=tf.float32,
                                        name="b{}_hidden".format(i))
                tf.summary.histogram('B'.format(i), bi_hidden)

            b_vars += [bi_hidden]

        with tf.name_scope('soft/'):
            b_soft = tf.Variable(tf.constant(0.1, shape=[1, output_size]), dtype=tf.float32, name="soft_bias")
            tf.summary.histogram('B', b_soft)

        b_vars += [b_soft]

    # we will need this zero bias when calculating the diff and diff_exsoft forwardprob
    b1_hidden_zero = tf.Variable(tf.zeros([1, num_neurons]), dtype=tf.float32, name="b1_hidden_zero")
    b_vars_zero = [b1_hidden_zero]
    for i in range(num_layers - 1):
        b_vars_zero += [tf.Variable(tf.zeros([1, num_neurons]),
                                    dtype=tf.float32, name="b{}_hidden_zero".format(i))]
    b_soft_zero = tf.Variable(tf.zeros([1, output_size]), dtype=tf.float32, name="soft_bias_zero")
    b_vars_zero += [b_soft_zero]

    return w_vars, w_vars_init, b_vars, b_vars_zero


def eval_diff(processed, y, w_vars, w_vars_init, b_vars, b_vars_zero, activation, name='Eval_diff'):
    with tf.name_scope(name):
        with tf.name_scope('diff'):
            # Calculate diff_weights and use it get logits_diff
            w_vars_diff = [w_vars[i] - w_vars_init[i] for i in range(len(w_vars))]
            logits_diff, _, _ = forwardprop(processed, w_vars_diff, b_vars_zero, activation, name='Forwardprop_diff')

            # Evaluate the accuracy
            diff_accuracy = eval_accuracy(logits_diff, y, name='Eval_accu_diff')
            tf.summary.scalar('accu/diff_accu', diff_accuracy)

        with tf.name_scope('diff_exsoft'):
            # Calculate diff_exsoft_weights and use it to get logits_diff_exsoft
            w_vars_diff_exsoft = [w_vars[i] - w_vars_init[i] for i in range(len(w_vars) - 1)]
            w_vars_diff_exsoft += [w_vars[-1]]

            # We also need to exclude the softmax bias
            b_vars_zero_exsoft = [b_vars_zero[i] for i in range(len(b_vars_zero) - 1)]
            b_vars_zero_exsoft += [b_vars[-1]]

            logits_diff_exsoft, _, _ = \
                forwardprop(processed, w_vars_diff_exsoft, b_vars_zero_exsoft, activation,
                            name='Forwardprop_diff_exsoft')

            # Evaluate the accuracy
            diff_exsoft_accuracy = eval_accuracy(logits_diff_exsoft, y, name='Eval_accu_diff_exsoft')
            tf.summary.scalar('accu/diff_accu_exsoft', diff_exsoft_accuracy)


def saliency_map_logits(sess, logits, processed, X, y, images, labels, num_to_viz=5):
    # this is an evaluation block
    # we pass in
    # 1. sess: the model with trained weights
    # 2. logits: the tensor we need for further evaluation operations
    # 3. images_placeholder: the slot where we can feed in the viz set
    # 4. images: to viz set
    # the saliency map is calculated based on logits

    max_logits = tf.reduce_max(logits, axis=1)
    saliencies = tf.gradients(max_logits, processed)

    saliency_maps = tf.reshape(saliencies, [-1, 64, 64, 3])
    saliency_maps_abs = tf.abs(saliency_maps)

    summary_Op1 = tf.summary.image('Saliency_logits', saliency_maps, max_outputs=num_to_viz)
    summary_Op2 = tf.summary.image('Saliency_logits_abs', saliency_maps_abs, max_outputs=num_to_viz)

    return sess.run(tf.summary.merge([summary_Op1, summary_Op2]), feed_dict={X: images, y: labels})


def saliency_map_lgsoft(sess, logits, processed, X, y, images, labels, num_to_viz=5):
    # this is an evaluation block
    # we pass in
    # 1. sess: the model with trained weights
    # 2. logits: the tensor we need for further evaluation operations
    # 3. images_placeholder: the slot where we can feed in the viz set
    # 4. images: to viz set
    # the saliency map is calculated based on softmax

    max_soft = tf.reduce_max(tf.nn.softmax(logits), axis=1)
    saliencies = tf.gradients(tf.log(max_soft), processed)

    saliency_maps = tf.reshape(saliencies, [-1, 64, 64, 3])
    saliency_maps_abs = tf.abs(saliency_maps)

    summary_Op1 = tf.summary.image('Saliency_lgsoft', saliency_maps, max_outputs=num_to_viz)
    summary_Op2 = tf.summary.image('Saliency_lgsoft_abs', saliency_maps_abs, max_outputs=num_to_viz)

    return sess.run(tf.summary.merge([summary_Op1, summary_Op2]), feed_dict={X: images, y: labels})


def viz_weights(sess, X, y, w_vars, h_vars, images, labels, num_to_viz=5):
    # this is an evaluation block
    # no matter how many layers we have, we will always multi them together and viz
    # the num of viz pictures equal to the number of classes

    multi = w_vars[0]
    multi_results = [multi]
    for i in range(len(w_vars) - 1):
        multi = tf.matmul(multi, w_vars[i + 1])
        multi_results += [multi]

    summary_Ops = []
    for i in range(len(multi_results)):
        trans = tf.transpose(multi_results[i])
        pics = tf.reshape(trans, [-1, 64, 64, 3])
        summary_Ops += [tf.summary.image('selected{}_multi_weights_upto_layer{}'.
                                         format(num_to_viz, i), pics, max_outputs=num_to_viz)]

    ###################################################################################################################

    # weights multi with masking matrices in between

    for i in range(num_to_viz):  # loop the viz set, for each input image
        result = w_vars[0]
        for j in range(len(w_vars) - 1):  # loop the layers, for each layer
            masking = tf.diag(tf.sign(h_vars[j][i]))
            temp = tf.matmul(result, masking)
            result = tf.matmul(temp, w_vars[j + 1])
        trans = tf.transpose(result)
        pics = tf.reshape(trans, [-1, 64, 64, 3])
        summary_Ops += [tf.summary.image('viz_img{}_masking_multi'.format(i), pics)]

    return sess.run(tf.summary.merge(summary_Ops), feed_dict={X: images, y: labels})


def input_viz(sess, processed, X, y, images, labels, num_to_viz=5):
    # this is an evaluation block
    # we pass in
    # 1. sess: the model with trained weights
    # 2. images_placeholder: the slot where we can feed in the viz set
    # 3. images: input images to viz

    # check what the inputs look like
    inputs = tf.reshape(processed, [-1, 64, 64, 3])
    summary_op = tf.summary.image('Input', inputs, max_outputs=num_to_viz)

    return sess.run(summary_op, feed_dict={X: images, y: labels})
