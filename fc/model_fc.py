import tensorflow as tf
import numpy as np
import os
import logging

from model_helper import *
from data_loader import read_image_data
from datetime import datetime


class FC_model(object):
    def __init__(self, config):
        self.num_layers = config.num_layers
        self.num_neurons = config.num_neurons
        self.input_dim = config.input_dim
        self.output_dim = config.output_dim

        self.init_std = config.init_std
        self.lr = config.lr
        self.pb = config.pb
        self.opt_type = config.opt_type

        self.is_weights = config.is_weights
        self.is_diff = config.is_diff
        self.batch_size = config.batch_size
        self.epochs = config.epochs

        self.patience = config.patience
        self.p_accu = config.p_accu
        self.rand_label = config.rand_label
        self.dataset = config.dataset

        self.is_saliency = config.is_saliency
        self.num_to_viz = config.num_to_viz
        self.is_total_perm = config.is_total_perm
        self.is_pixel_perm = config.is_pixel_perm

        self.is_rand_sparse = config.is_rand_sparse
        self.is_single_sparse = config.is_single_sparse
        self.is_multi_sparse = config.is_multi_sparse
        self.sparse_ratio = config.sparse_ratio
        self.sparse_set_size = config.sparse_set_size

        self.is_viz_perm_inv = config.is_viz_perm_inv

        if config.act_func == 'relu':
            activation = tf.nn.relu
        elif config.act_func == 'tanh':
            activation = tf.nn.tanh
        else:
            activation = tf.nn.relu
            print("Unknown type of act-func! Use relu as default")

        # Placeholders for the images and labels
        self.imgs, self.labels = placeholder_inputs(self.input_dim, self.output_dim)

        # Initialize weights and bias
        self.w_vars, self.w_vars_init, self.b_vars, self.b_vars_zero = \
            init_weights_bias(self.input_dim, self.output_dim, self.num_neurons, self.num_layers,
                              self.init_std, self.pb)

        # Forward propagation
        self.logits, self.h_vars, h_before_vars = forwardprop(self.imgs, self.w_vars, self.b_vars, activation)

        # Loss
        self.loss = entropy_loss(self.logits, self.labels)

        # Training operation
        self.train_op = train_opt(self.loss, self.lr, self.opt_type)

        # Accuracy
        self.accu = eval_accuracy(self.logits, self.labels)

        if self.is_diff:
            eval_diff(self.imgs, self.labels, self.w_vars,
                      self.w_vars_init, self.b_vars, self.b_vars_zero,
                      activation)

        # merge all the summaries
        self.sum_merged = tf.summary.merge_all()

    def train(self, data_dir, log_dir, model_path, summary_path):

        # if "perm" is false, the two "perm matrices" are identity matrices
        train_X, test_X, train_y, test_y, train_fn, test_fn, total_perm_mat_inv, pixel_perm_mat_inv \
            = read_image_data(data_dir,
                              'RGB',
                              is_total_perm=self.is_total_perm,
                              is_pixel_perm=self.is_pixel_perm,
                              is_rand_sparse=self.is_rand_sparse,
                              is_single_sparse=self.is_single_sparse,
                              is_multi_sparse=self.is_multi_sparse,
                              sparse_ratio=self.sparse_ratio,
                              sparse_set_size=self.sparse_set_size)

        # just to pick a few to visualize. image is huge
        to_viz = np.random.choice(range(train_X.shape[0]), self.num_to_viz)

        train_X_to_viz = train_X[to_viz, :]
        train_y_to_viz = train_y[to_viz, :]
        train_fn_to_viz = train_fn[to_viz]

        # logging info
        logging.basicConfig(filename=os.path.join(log_dir, "train_{}.log".
                                                  format(datetime.now().strftime("%m%d_%H%M%S"))),
                            level=logging.DEBUG)

        # Add two placeholders to indicate the inverse perm matrices
        assert train_X.ndim == 2, "Training images are not flattened!"
        total_perm_mat_inv_tf = tf.placeholder(tf.float32, shape=[train_X.shape[1], train_X.shape[1]])
        pixel_perm_mat_inv_tf = tf.placeholder(tf.float32, shape=[train_X.shape[1], train_X.shape[1]])

        # Saver
        saver = tf.train.Saver()

        gpu_options = tf.GPUOptions(allow_growth=True)
        sess_config = tf.ConfigProto(allow_soft_placement=True,
                                     gpu_options=gpu_options)
        with tf.Session(config=sess_config) as sess:
            # summary writer
            train_writer = tf.summary.FileWriter(summary_path + '/train', sess.graph)
            test_writer = tf.summary.FileWriter(summary_path + '/test')
            sess.run(tf.global_variables_initializer())

            # train & visualization
            step = 0
            for epoch in range(self.epochs):
                for b in range(int(len(train_X) / self.batch_size)):
                    # training
                    train_X_batch = train_X[self.batch_size * b: self.batch_size * b + self.batch_size]
                    train_y_batch = train_y[self.batch_size * b: self.batch_size * b + self.batch_size]
                    _, sum_str_train, train_accu = \
                        sess.run([self.train_op, self.sum_merged, self.accu],
                                 feed_dict={self.imgs: train_X_batch, self.labels: train_y_batch})
                    train_writer.add_summary(sum_str_train, step)

                    # testing
                    sum_str_test, test_accu = \
                        sess.run([self.sum_merged, self.accu],
                                 feed_dict={self.imgs: test_X, self.labels: test_y})
                    test_writer.add_summary(sum_str_test, step)

                    msg = "epoch = {}, batch = {}, " \
                          "train accu = {:.4f}, test accu = {:.4f}".format(epoch, b, train_accu, test_accu)
                    print(msg)
                    logging.info(msg)

                    step += 1

            # viz inputs
            train_writer.add_summary(input_viz(sess, self.imgs, train_X_to_viz, self.num_to_viz))

            # saliency map
            if self.is_saliency:
                # viz saliency map calculated based on logits
                train_writer.add_summary(
                    saliency_map_logits(sess,
                                        self.logits,
                                        self.imgs, train_X_to_viz,
                                        total_perm_mat_inv_tf, total_perm_mat_inv,
                                        pixel_perm_mat_inv_tf, pixel_perm_mat_inv,
                                        self.num_to_viz, self.is_viz_perm_inv))

                # viz saliency map calculated based on log(softmax)
                train_writer.add_summary(
                    saliency_map_lgsoft(sess,
                                        self.logits,
                                        self.imgs, train_X_to_viz,
                                        total_perm_mat_inv_tf, total_perm_mat_inv,
                                        pixel_perm_mat_inv_tf, pixel_perm_mat_inv,
                                        self.num_to_viz, self.is_viz_perm_inv))

            if self.is_weights:
                # viz weights
                train_writer.add_summary(
                    viz_weights(sess,
                                self.imgs,
                                self.w_vars,
                                self.h_vars,
                                train_X_to_viz,
                                total_perm_mat_inv_tf, total_perm_mat_inv,
                                pixel_perm_mat_inv_tf, pixel_perm_mat_inv,
                                self.num_to_viz, self.is_viz_perm_inv))

            # save model
            save_path = saver.save(sess, os.path.join(model_path, "model.ckpt"), global_step=step)
            print("Model saved in file: %s" % save_path)
