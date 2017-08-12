import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os


def normalize_contrast(matrix):
    # each row of matrix is an image
    shifted = tf.subtract(matrix, tf.reduce_min(matrix, axis=1, keep_dims=True))
    normalized = tf.divide(shifted, tf.reduce_max(shifted, axis=1, keep_dims=True)) * 255.
    return normalized


def prep_dirs(FLAGS):
    # image dataset path
    data_path = os.path.join("../data", FLAGS.dataset)

    # summary path and name
    summary_path = os.path.join("../summaries", FLAGS.spath, "fc_nlayers{}_nunits{}_bs{}_lr{}_std{}_rs{}_perm{}_sparse{}__{}".
                                format(FLAGS.num_layers, FLAGS.num_neurons, FLAGS.batch_size,
                                       FLAGS.lr, FLAGS.init_std, FLAGS.rs, int(FLAGS.is_perm),
                                       FLAGS.sparse_ratio, FLAGS.dataset))

    # always save the training log
    log_dir = os.path.join("../logs", "fc_nlayers{}_nunits{}_bs{}_lr{}_std{}_rs{}_perm{}_sparse{}__{}".
                           format(FLAGS.num_layers, FLAGS.num_neurons, FLAGS.batch_size,
                                  FLAGS.lr, FLAGS.init_std, FLAGS.rs, int(FLAGS.is_perm),
                                  FLAGS.sparse_ratio, FLAGS.dataset))

    # always save the trained model
    model_path = os.path.join("../saved_models", "fc_nlayers{}_nunits{}_bs{}_lr{}_std{}_rs{}_perm{}_sparse{}__{}".
                              format(FLAGS.num_layers, FLAGS.num_neurons, FLAGS.batch_size,
                                     FLAGS.lr, FLAGS.init_std, FLAGS.rs, int(FLAGS.is_perm),
                                     FLAGS.sparse_ratio, FLAGS.dataset))

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(summary_path):
        os.makedirs(summary_path)


    return data_path, log_dir, model_path, summary_path


def save_saliency_img(img_original, saliency, max_class, path, title="default saliency"):
    # get out the first map and class from the mini-batch
    saliency = saliency[0]
    max_class = max_class[0]

    # convert saliency from BGR to RGB, and from c01 to 01c
    # saliency = saliency[::-1].transpose(1, 2, 0)

    # convert saliency from BGR to RGB, and from c01 to 01c
    saliency = np.reshape(saliency, [64, 64, 3])

    # plot the original image and the three saliency map variants
    # plt.figure(figsize=(10, 10), facecolor='w')
    plt.figure()
    plt.suptitle("Class: " + str(max_class) + ". Saliency: " + title)
    plt.subplot(2, 2, 1)
    plt.title('input')
    plt.imshow(img_original)
    plt.subplot(2, 2, 2)
    plt.title('abs. saliency')
    plt.imshow(np.abs(saliency).max(axis=-1), cmap='gray')
    plt.subplot(2, 2, 3)
    plt.title('pos. saliency')
    plt.imshow((np.maximum(0, saliency) / saliency.max()))
    plt.subplot(2, 2, 4)
    plt.title('neg. saliency')
    plt.imshow((np.maximum(0, -saliency) / -saliency.min()))
    plt.savefig("{}/saliency_map.png".format(path))
