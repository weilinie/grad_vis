import os
import sys
import random
import tensorflow as tf
import numpy as np
import glob
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt, gridspec
from scipy.misc import imread, imresize
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops
from utils import prep_dirs
from config_fc import get_config
from test_fc_model import FC_model


sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))


@ops.RegisterGradient("GuidedRelu")
def _GuidedReluGrad(op, grad):
    return tf.where(0. < grad, gen_nn_ops._relu_grad(grad, op.outputs[0]), tf.zeros(tf.shape(grad)))


def plot_save(img, sal_map, sal_map_type, save_dir, fn, num_layers):
    # img = np.reshape(img, [224, 224, 3])
    # sal_map = np.reshape(sal_map, [224, 224, 3])

    # normalizations
    sal_map -= np.min(sal_map)
    sal_map /= sal_map.max()

    img = img.astype(float)
    img -= np.min(img)
    img /= img.max()

    fig = plt.figure()

    gs = gridspec.GridSpec(1, 2, wspace=0.2, hspace=0.2)

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(img)
    ax.set_title('Input Image', fontsize=8)
    ax.tick_params(axis='both', which='major', labelsize=6)

    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(sal_map)
    ax.set_title(sal_map_type, fontsize=8)
    ax.tick_params(axis='both', which='major', labelsize=6)

    # saved results path
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print('Saving {}_{}_layers{}.png'.format(sal_map_type, fn, num_layers))
    plt.savefig(os.path.join(save_dir, "{}_{}_layers{}.png".format(sal_map_type, fn, num_layers)))

def main(FLAGS):

    sal_map_type = "GuidedBackprop_maxlogit"
    # sal_map_type = "GuidedBackprop_cost"
    data_dir = "../VGGImagenet/data_imagenet"
    save_dir = "test_gbp_results/10222017"

    image_dict = {'tabby': 281, 'laska': 356, 'mastiff': 243}

    image_name = 'tabby'

    data_path, log_dir, model_path, summary_path = prep_dirs(FLAGS)

    fns = []
    image_list = []
    label_list = []

    # load in the original image and its adversarial examples
    for image_path in glob.glob(os.path.join(data_dir, '{}*.png'.format(image_name))):
        fns.append(os.path.basename(image_path).split('.')[0])
        image = imread(image_path, mode='RGB')
        image = imresize(image, (224, 224)).astype(np.float32)
        image_list.append(image)
        onehot_label = np.array([1 if i == image_dict[image_name] else 0 for i in range(1000)])
        label_list.append(onehot_label)

    batch_img = np.array(image_list)
    batch_label = np.array(label_list)

    batch_size = batch_img.shape[0]

    # tf session
    sess = tf.Session()

    # construct the graph based on the gradient type
    if sal_map_type.split('_')[0] == 'GuidedBackprop':
        eval_graph = tf.get_default_graph()
        with eval_graph.gradient_override_map({'Relu': 'GuidedRelu'}):
            fc_model = FC_model(config=FLAGS)
    elif sal_map_type.split('_')[0] == 'PlainSaliency':
        fc_model = FC_model(config=FLAGS)
    else:
        raise Exception("Unknown saliency_map type - 1")

    sess.run(tf.global_variables_initializer())

    # Calculate gradient
    max_logit = tf.reduce_max(fc_model.logits, axis=1)
    # saliency gradient to input layer
    if sal_map_type.split('_')[1] == "cost":
        sal_map, = tf.gradients(fc_model.loss, fc_model.imgs)
    elif sal_map_type.split('_')[1] == 'maxlogit':
        sal_map, = tf.gradients(max_logit, fc_model.imgs)
    else:
        raise Exception("Unknown saliency_map type - 2")
    # sal_map = tf.reshape(sal_map, [-1, 224, 224, 3])
    sals_val = sess.run(sal_map, feed_dict={fc_model.images: batch_img, fc_model.labels: batch_label})

    for idx in range(batch_size):
        plot_save(batch_img[idx], sals_val[idx], sal_map_type, save_dir, fns[idx], FLAGS.num_layers)

    sess.close()


if __name__ == '__main__':
    FLAGS, unparsed = get_config()
    # setup the GPUs to use
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpus
    main(FLAGS)