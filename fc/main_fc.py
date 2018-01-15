import os
import sys
import random
import tensorflow as tf
import numpy as np
from utils import prep_dirs
from config_fc import get_config
from model_fc import FC_model

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))


def main(FLAGS):
    # TODO: viz
    viz_dimension = (10, 10)
    img_dim = (64, 64, 3)

    # set the random seed
    random.seed(FLAGS.rs)
    tf.set_random_seed(FLAGS.rs)
    np.random.seed(FLAGS.rs)

    # setup the GPUs to use
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpus

    data_path, log_dir, model_path, summary_path = prep_dirs(FLAGS)

    fc_model = FC_model(config=FLAGS)

    # start the training
    fc_model.train(data_path, log_dir, model_path, summary_path)


if __name__ == '__main__':
    FLAGS, unparsed = get_config()
    main(FLAGS)
