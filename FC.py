import os
import sys
import random
import tensorflow as tf
import numpy as np
import scipy
import argparse
import logging
from utils import save_saliency_img
from scipy import misc

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
from data_loader import read_image_data

# Basic model parameters as external flags.
FLAGS = None
# activation function to use
activation = tf.nn.relu

def str2bool(v):
    return v.lower() in ('true', '1')

def normalize_contrast(matrix):
    shifted = matrix - matrix.min()
    return (shifted / np.ptp(shifted) * 255).astype(np.uint8)

def forwardprop(X, w_vars, b_vars, activation, name='Forwardprop'):

    with tf.name_scope(name):

        h_before = tf.matmul(X, w_vars[0]) + b_vars[0]
        h = activation(h_before)

        h_before_vars = [h_before]
        h_vars = [h]

        for i in range(FLAGS.num_layers - 1):

            h_before = tf.matmul(h, w_vars[i + 1]) + b_vars[i + 1]
            h = activation(h_before)

            h_before_vars += [h_before]
            h_vars += [h]

        yhat = tf.matmul(h, w_vars[-1]) + b_vars[-1]

    return yhat, h_vars, h_before_vars

def placeholder_inputs(input_size, output_size, name='Inputs'):

  with tf.name_scope(name):

      images_placeholder = tf.placeholder(tf.float32, shape=(None, input_size), name='X')
      labels_placeholder = tf.placeholder(tf.int32, shape=(None, output_size), name='y')

  return images_placeholder, labels_placeholder

def weights_and_bias(input_size, output_size):

    with tf.name_scope('hidden1/'):
        w1_hidden = tf.Variable(tf.truncated_normal((input_size, FLAGS.num_neurons), stddev=FLAGS.std),
                                dtype=tf.float32, name="w1_hidden")
        tf.summary.histogram('W',w1_hidden)

    w_vars = [w1_hidden]

    for i in range(FLAGS.num_layers - 1):

        with tf.name_scope('hidden{}/'.format(i+2)):
            wi_hidden = tf.Variable(tf.truncated_normal((FLAGS.num_neurons, FLAGS.num_neurons), stddev=FLAGS.std),
                                   dtype=tf.float32, name="w{}_hidden".format(i + 2))
            tf.summary.histogram('W'.format(i), wi_hidden)

        w_vars += [wi_hidden]

    with tf.name_scope('soft/'):
        w_soft = tf.Variable(tf.truncated_normal((FLAGS.num_neurons, output_size), stddev=FLAGS.std),
                         dtype=tf.float32, name="w_soft")
        tf.summary.histogram('W', w_soft)

    w_vars += [w_soft]

    # store the init values so that we can calculate the learned "diff" later
    w_vars_init = [tf.Variable(w_vars[i].initialized_value(), name='w_init_{}'.format(i))
                   for i in range(len(w_vars))]

    # init bias either to all zeros or to small positive constant 0.1
    if not FLAGS.pb:

        with tf.name_scope('hidden1/'):
            b1_hidden = tf.Variable(tf.zeros([1, FLAGS.num_neurons]), dtype=tf.float32, name="b1_hidden")
            tf.summary.histogram('B', b1_hidden)

        b_vars = [b1_hidden]

        for i in range(FLAGS.num_layers - 1):

            with tf.name_scope('hidden{}/'.format(i+2)):
                bi_hidden = tf.Variable(tf.zeros([1, FLAGS.num_neurons]), dtype=tf.float32, name="b{}_hidden".format(i))
                tf.summary.histogram('B'.format(i), bi_hidden)

            b_vars += [bi_hidden]

        with tf.name_scope('soft/'):
            b_soft = tf.Variable(tf.zeros([1, output_size]), dtype=tf.float32, name="soft_bias")
            tf.summary.histogram('B', b_soft)

        b_vars += [b_soft]

    else:

        with tf.name_scope('hidden1/'):
            b1_hidden = tf.Variable(tf.constant(0.1, shape = [1, FLAGS.num_neurons]), dtype=tf.float32, name="b1_hidden")
            tf.summary.histogram('B', b1_hidden)

        b_vars = [b1_hidden]

        for i in range(FLAGS.num_layers - 1):

            with tf.name_scope('hidden{}/'.format(i+2)):
                bi_hidden = tf.Variable(tf.constant(0.1, shape = [1, FLAGS.num_neurons]), dtype=tf.float32, name="b{}_hidden".format(i))
                tf.summary.histogram('B'.format(i), bi_hidden)

            b_vars += [bi_hidden]

        with tf.name_scope('soft/'):
            b_soft = tf.Variable(tf.constant(0.1, shape = [1, output_size]), dtype=tf.float32, name="soft_bias")
            tf.summary.histogram('B', b_soft)

        b_vars += [b_soft]

    # we will need this zero bias when calculating the diff and diff_exsoft forwardprob
    b1_hidden_zero = tf.Variable(tf.zeros([1, FLAGS.num_neurons]), dtype=tf.float32, name="b1_hidden_zero")
    b_vars_zero = [b1_hidden_zero]
    for i in range(FLAGS.num_layers - 1):
        b_vars_zero += [tf.Variable(tf.zeros([1, FLAGS.num_neurons]), dtype=tf.float32, name="b{}_hidden_zero".format(i))]
    b_soft_zero = tf.Variable(tf.zeros([1, output_size]), dtype=tf.float32, name="soft_bias_zero")
    b_vars_zero += [b_soft_zero]

    return w_vars, w_vars_init, b_vars, b_vars_zero

def entropy_loss(logits, labels, name='Entropy_loss'):

    with tf.name_scope(name):

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='cross_entropy')
        cost = tf.reduce_mean(cross_entropy, name='entropy_mean')

    return cost

def train(loss, name='Train'):

    with tf.name_scope(name):

        optimizer = tf.train.AdamOptimizer(FLAGS.lr)
        train_op = optimizer.minimize(loss)

    return train_op

def eval_accuracy(logits, labels, name='Eval_accu'):

    with tf.name_scope(name):

        predict = tf.argmax(logits, axis=1)
        correct = tf.argmax(labels, axis=1)
        accu = tf.reduce_mean(tf.to_float(tf.equal(correct, predict)))

    return accu

def eval_diff(X, y, w_vars, w_vars_init, b_vars, b_vars_zero, name='Eval_diff'):

    with tf.name_scope(name):

        with tf.name_scope('diff'):

            # Calculate diff_weights and use it get logits_diff
            w_vars_diff = [w_vars[i] - w_vars_init[i] for i in range(len(w_vars))]
            logits_diff, _, _ = forwardprop(X, w_vars_diff, b_vars_zero, activation, name='Forwardprop_diff')

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

            logits_diff_exsoft, _, _ =\
                forwardprop(X, w_vars_diff_exsoft, b_vars_zero_exsoft, activation, name='Forwardprop_diff_exsoft')

            # Evaluate the accuracy
            diff_exsoft_accuracy = eval_accuracy(logits_diff_exsoft, y, name='Eval_accu_diff_exsoft')
            tf.summary.scalar('accu/diff_accu_exsoft', diff_exsoft_accuracy)

def saliency_map(sess, logits, images_placeholder, images):

    # this is an evaluation block
    # we pass in
    # 1. sess: the model with trained weights
    # 2. logits: the tensor we need for further evaluation operations
    # 3. images_placeholder: the slot where we can feed in the viz set
    # 4. images: to viz set

    max_logits = tf.reduce_max(logits, axis=1)
    saliencies = tf.gradients(max_logits, images_placeholder)

    saliency_maps = tf.reshape(saliencies, [-1, 64, 64, 3])
    summary_Op = tf.summary.image('Saliency', saliency_maps, max_outputs=FLAGS.num_to_viz, collections=None)

    return sess.run(summary_Op, feed_dict={images_placeholder : images})

def input_viz(sess, images_placeholder, images):

    # this is an evaluation block
    # we pass in
    # 1. sess: the model with trained weights
    # 2. images_placeholder: the slot where we can feed in the viz set
    # 3. images: input images to viz

    # check what the inputs look like
    inputs = tf.reshape(images_placeholder, [-1, 64, 64, 3])
    summary_op = tf.summary.image('Inputs', inputs, max_outputs=FLAGS.num_to_viz, collections=None)

    return sess.run(summary_op, feed_dict={images_placeholder : images})

def run_training(viz_dimension, img_dim, image_folder, summary_name, log_dir, saved_model):

    train_X, test_X, train_y, test_y, train_fn, test_fn = read_image_data(image_folder, 'RGB')

    # just to pick a few to visualize. image is huge
    to_viz = np.random.choice(range(train_X.shape[0]), FLAGS.num_to_viz)

    train_X_to_viz = train_X[to_viz, :]
    train_y_to_viz = train_y[to_viz, :]
    train_fn_to_viz = train_fn[to_viz]
                                                                          
    # Layer's sizes
    input_size = train_X.shape[1]
    output_size = train_y.shape[1]

    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():

        # Placeholders for the images and labels.
        images_placeholder, labels_placeholder = placeholder_inputs(input_size, output_size)

        # Define weights and bias
        w_vars, w_vars_init, b_vars, b_vars_zero = weights_and_bias(input_size, output_size)

        # Forward propagation
        logits, h_vars, h_before_vars = forwardprop(images_placeholder, w_vars, b_vars, activation)

        # Loss
        loss = entropy_loss(logits, labels_placeholder)
        tf.summary.scalar('train/loss', loss)

        # Training operation
        train_op = train(loss)

        # Accuracy
        accuracy = eval_accuracy(logits, labels_placeholder)
        tf.summary.scalar('accu/accu', accuracy)

        if FLAGS.is_diff:

            # Evaluate "diff"
            eval_diff(images_placeholder,
                      labels_placeholder,
                      w_vars,
                      w_vars_init,
                      b_vars,
                      b_vars_zero)

        # Saver
        saver = tf.train.Saver()

        # create sess
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        # merge all the summaries
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(summary_name + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(summary_name + '/test')

        # init
        sess.run(tf.global_variables_initializer())

        if not os.path.exists("saved_model"):
            os.mkdir("saved_model")
        if not os.path.exists("logs"):
            os.mkdir("logs")

        # logging info
        logging.basicConfig(filename=log_dir, level=logging.DEBUG)

        # train & visualization
        step = 0

        for epoch in range(FLAGS.epochs):

            for b in range(int(len(train_X) / FLAGS.bs)):

                #######################################################################################################

                # training
                _, sum_str_train, train_accu =\
                    sess.run([train_op, merged, accuracy],
                                      feed_dict={images_placeholder: train_X[FLAGS.bs * b: FLAGS.bs * b + FLAGS.bs],
                                                 labels_placeholder: train_y[FLAGS.bs * b: FLAGS.bs * b + FLAGS.bs]})

                train_writer.add_summary(sum_str_train, step)

                #######################################################################################################

                # testing
                sum_str_test, test_accu =\
                    sess.run([merged, accuracy],
                                      feed_dict={images_placeholder: test_X,
                                                 labels_placeholder: test_y})

                test_writer.add_summary(sum_str_test, step)

                #######################################################################################################

                msg = "epoch = {}, batch = {}, " \
                      "train accu = {:.4f}, test accu = {:.4f}".format(epoch, b, train_accu, test_accu)
                print(msg)
                logging.info(msg)

                step += 1

        ###############################################################################################################
        #################                      Visualizations and Evaluations                        ##################
        ###############################################################################################################

        # visual train_X_to_viz
        if FLAGS.is_input :
            train_writer.add_summary(input_viz(sess, images_placeholder, train_X_to_viz))

        # saliency map
        if FLAGS.is_saliency :
            train_writer.add_summary(saliency_map(sess, logits, images_placeholder, train_X_to_viz))

        ###############################################################################################################
        ###############################################################################################################

        # save model
        save_path = saver.save(sess, os.path.join("saved_model", saved_model))
        print("Model saved in file: %s" % save_path)

        sess.close()

def main():

    # some dimensions
    # Note: the product of viz_dimension elements has to be equal to the num of hidden neurons
    viz_dimension = (10, 10)
    img_dim = (64, 64, 3)

    # set the random seed
    random.seed(FLAGS.rs)
    tf.set_random_seed(FLAGS.rs)
    np.random.seed(FLAGS.rs)

    # setup the GPUs to use
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpus

    # image dataset path
    image_folder = os.path.join("data/", FLAGS.dataset)

    # summary path and name
    summary_path = os.path.join("summaries", FLAGS.s_path)
    if not os.path.exists(summary_path):
        os.makedirs(summary_path)
    summary_name = os.path.join(summary_path, "fc_num_layers{}_num_neurons{}_bs{}_lr{}_std{}_{}".
                                format(FLAGS.num_layers, FLAGS.num_neurons, FLAGS.bs, FLAGS.lr, FLAGS.std, FLAGS.dataset))

    # always save the training log
    log_dir = os.path.join("logs", "fc_num_layers{}_num_neurons{}_bs{}_lr{}_std{}_{}.log".
                                format(FLAGS.num_layers, FLAGS.num_neurons, FLAGS.bs, FLAGS.lr, FLAGS.std, FLAGS.dataset))

    # always save the trained model
    saved_model = "fc_num_layers{}_num_neurons{}_bs{}_lr{}_std{}_{}.ckpt".\
        format(FLAGS.num_layers, FLAGS.num_neurons, FLAGS.bs, FLAGS.lr, FLAGS.std, FLAGS.dataset)

    # start the training
    run_training(viz_dimension,
                 img_dim,
                 image_folder,
                 summary_name,
                 log_dir,
                 saved_model
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-pb',
                        '--pb',
                        type=str2bool,
                        default=True,
                        help='init bias slightly positive to 0.1, 0 if turned off'
    )
    parser.add_argument('-dataset',
                        '--dataset',
                        type=str,
                        default='2Rec_64_4000_20_1_black',
                        help='specify the dataset to use'
    )
    parser.add_argument('-std',
                        '--std',
                        type=float,
                        default=1e-1,
                        help='specify the init std for the training'
    )
    parser.add_argument('-gpus',
                        '--gpus',
                        type=str,
                        default='7',
                        help='specify which GPU to use'
    )
    parser.add_argument('-epochs',
                        '--epochs',
                        type=int,
                        default=100,
                        help='specify the total # of epochs for the training'
    )
    parser.add_argument('-lr',
                        '--lr',
                        type=float,
                        default=1e-3,
                        help='specify the learning rate for the training'
    )
    parser.add_argument('-bs',
                        '--bs',
                        type=int,
                        default=128,
                        help='specify the batch size for the training'
    )
    parser.add_argument('-p_accu',
                        '--p_accu',
                        type=int,
                        default=5,
                        help='specify for every how many steps print the accuracy')
    parser.add_argument('-num_neurons',
                        '--num_neurons',
                        type=int,
                        default=100,
                        help='specify the # of hidden neurons for each layer'
    )
    parser.add_argument('-num_layers',
                        '--num_layers',
                        type=int,
                        default=1,
                        help='specify the # of hidden layers'
    )
    parser.add_argument('-pa',
                        '--pa',
                        type=int,
                        default=1250, # 50 epochs
                        help='specify the patience which is used in early stop')
    parser.add_argument('-rs',
                        '--rs',
                        type=int,
                        default=42,
                        help='specify the random seed for the training'
    )
    parser.add_argument('-s_path',
                        '--s_path',
                        type=str,
                        default='summaries',
                        help='specify the summary path for the training'
    )
    parser.add_argument('-is_diff',
                        '--is_diff',
                        type=str2bool,
                        default=False,
                        help='specify if turn on diff modual'
    )
    parser.add_argument('-num_to_viz',
                        '--num_to_viz',
                        type=int,
                        default=5,
                        help='specify the # of images to viz'
    )
    parser.add_argument('-is_saliency',
                        '--is_saliency',
                        type=str2bool,
                        default=True,
                        help='specify if turn on saliency map'
    )
    parser.add_argument('-is_input',
                        '--is_input',
                        type=str2bool,
                        default=True,
                        help='specify if viz a few inputs'
    )
    FLAGS, unparsed = parser.parse_known_args()
    main()