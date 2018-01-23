from scipy.misc import imread, imresize
import os
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops
np.set_printoptions(threshold=np.nan)
import glob
from vgg16 import Vgg16
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt, gridspec


def data(image_name, index):

    data_dir = "data_imagenet"

    fns = []
    image_list = []
    label_list = []

    # load in the original image and its adversarial examples
    for image_path in glob.glob(os.path.join(data_dir, '{}.png'.format(image_name))):
        file_name = os.path.basename(image_path).split('.')[0]
        print('File name : {}').format(file_name)
        fns.append(file_name)
        image = imread(image_path, mode='RGB')
        image = imresize(image, (224, 224)).astype(np.float32)
        image_list.append(image)
        onehot_label = np.array([1 if i == index else 0 for i in range(1000)])
        label_list.append(onehot_label)

    batch_img = np.array(image_list)
    batch_label = np.array(label_list)

    return batch_img, batch_label, fns

def prepare_vgg(sal_type, act_type, pool_type, layer_idx, load_weights, sess):

    # construct the graph based on the gradient type we want
    if sal_type == 'GuidedBackprop':
        eval_graph = tf.get_default_graph()
        with eval_graph.gradient_override_map({'Relu': 'GuidedRelu'}):
            vgg = Vgg16(sess=sess, act_type=act_type, pool_type=pool_type)

    elif sal_type == 'Deconv':
        eval_graph = tf.get_default_graph()
        with eval_graph.gradient_override_map({'Relu': 'DeconvRelu'}):
            vgg = Vgg16(sess=sess, act_type=act_type, pool_type=pool_type)

    elif sal_type == 'PlainSaliency':
        vgg = Vgg16(sess=sess, act_type=act_type, pool_type=pool_type)

    else:
        raise Exception("Unknown saliency_map type - 1")

    # different options for loading weights
    if load_weights == 'trained':
        vgg.load_weights('vgg16_weights.npz', sess)

    elif load_weights == 'random':
        vgg.init(sess)

    elif load_weights == 'part':
        # fill the first "idx" layers with the trained weights
        # randomly initialize the rest
        vgg.load_weights_part(layer_idx * 2 + 1, 'vgg16_weights.npz', sess)

    elif load_weights == 'reverse':
        # do not fill the first "idx" layers with the trained weights
        # randomly initialize them
        vgg.load_weights_reverse(layer_idx * 2 + 1, 'vgg16_weights.npz', sess)

    elif load_weights == 'only':
        # do not load a specific layer ("idx") with the trained weights
        # randomly initialize it
        vgg.load_weights_only(layer_idx * 2 + 1, 'vgg16_weights.npz', sess)

    else:
        raise Exception("Unknown load_weights type - 1")

    return vgg

def norm(img):

    img = np.abs(img)
    img /= np.sum(img)
    img /= np.max(img)

    return img

def evaluate(image_name, dict_image, dict_salmap, iterations):

    save_dir = 'results/12042017/deepdream_defense_{}/'.format(image_name)

    for i in range(iterations - 1): # check each step

        img = dict_image[i][0]
        sal = dict_salmap[i + 1][0]

        img = norm(img)
        sal = norm(sal)

        fig = plt.figure()

        gs = gridspec.GridSpec(1, 2, wspace=0.2, hspace=0.2)

        ax = fig.add_subplot(gs[0, 0])
        ax.imshow(img)
        ax.set_title('Iteration_{}'.format(i), fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=6)

        ax = fig.add_subplot(gs[0, 1])
        ax.imshow(sal)
        ax.set_title('SalMap', fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=6)

        # save
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(os.path.join(save_dir, "deepdream_defense_{}_{}.png".format(i, image_name)))

        plt.close()

def main():

    inputs = {} # hold the temp images for each step
    sal_results = {}
    step_size = 1e-1
    iterations = 1000
    image_name = 'tabby'
    label_index = 281 # we assume the image is correctly classified
    gradient_type = 'PlainSaliency'

    # load the image [1, 224, 224, 3]
    batch_img, batch_label, fns = data(image_name, label_index)

    sess = tf.Session()

    # prepare the networks
    vgg = prepare_vgg(gradient_type, 'relu', 'maxpool', None, 'trained', sess) # used for probing

    target = -1 * tf.reduce_mean(tf.square(vgg.logits - batch_label))

    sal = tf.gradients(vgg.probs[0][label_index], vgg.images)[0]

    sign_grad = tf.sign(tf.gradients(target, vgg.images)[0])

    for step in range(iterations):

        if step == 0:
            inputs[step] = batch_img
            continue

        old = inputs[step - 1]
        target_val, grad_val, sal_val, probs_val = sess.run([target, sign_grad, sal, vgg.probs], feed_dict={vgg.images: old})
        print('Prediction : {}, Probability : {}, Diff : {}'.format(np.argmax(probs_val[0]), probs_val[0][label_index], -1 * target_val))
        sal_results[step] = sal_val

        inputs[step] = old + step_size * (0.0 * grad_val + 1.0 * np.sign(sal_val))

    evaluate(image_name, inputs, sal_results, iterations)

    sess.close()


if __name__ == '__main__':
    # setup the GPUs to use
    os.environ['CUDA_VISIBLE_DEVICES'] = '5, 6'
    main()