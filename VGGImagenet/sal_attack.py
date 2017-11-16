from scipy.misc import imread, imresize
import os
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops
np.set_printoptions(threshold=np.nan)
import glob
from vgg16 import Vgg16
from matplotlib import pyplot as plt, gridspec

@ops.RegisterGradient("GuidedRelu")
def _GuidedReluGrad(op, grad):
    return tf.where(0. < grad, gen_nn_ops._relu_grad(grad, op.outputs[0]), tf.zeros(tf.shape(grad)))

@ops.RegisterGradient("DeconvRelu")
def _GuidedReluGrad(op, grad):
    return tf.where(0. < grad, grad, tf.zeros(tf.shape(grad)))

def prepare_vgg(sal_type, layer_idx, load_weights, sess):

    # construct the graph based on the gradient type we want
    if sal_type == 'GuidedBackprop':
        eval_graph = tf.get_default_graph()
        with eval_graph.gradient_override_map({'Relu': 'GuidedRelu'}):
            vgg = Vgg16(sess=sess)

    elif sal_type == 'Deconv':
        eval_graph = tf.get_default_graph()
        with eval_graph.gradient_override_map({'Relu': 'DeconvRelu'}):
            vgg = Vgg16(sess=sess)

    elif sal_type == 'PlainSaliency':
        vgg = Vgg16(sess=sess)

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

def sal_maxlogit(network):

    """
    network will be a vgg object
    """

    sal = tf.gradients(network.maxlogit, network.images)[0]
    return sal

def helper(idx, shape, flat):

    i = tf.mod(idx, shape[0])
    j = tf.mod((idx - i * shape[0]), shape[1])
    k = idx - i * shape[0] - j * shape[1]

    return tf.multiply(flat[idx], tf.cast(tf.convert_to_tensor([i, j, k]), tf.float32))

def center_mass(tensor_batch_sal):

    """
    Another node in the graph to calculate the center mass of the saliency (tensor_batch_sal).
    Notice that the saliency is a node in the graph
    """

    # take the sal out of the batch
    sal = tensor_batch_sal[0]
    shape = sal.get_shape().as_list() # usually (224, 224, 3)

    # normalize the sal
    sal /= tf.norm(sal)

    # flat the tensor
    flat = tf.reshape(sal, [-1])

    len = flat.get_shape().as_list()[0]

    tensor_list = tf.map_fn(lambda x : helper(x, shape, flat), tf.constant(range(len)), dtype=tf.float32)

    tensor = tf.concat(tensor_list, axis=0)

    return tf.reduce_sum(tensor, axis=0)

def data(image_name):

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
        onehot_label = np.array([1 if i == 1 else 0 for i in range(1000)])
        label_list.append(onehot_label)

    batch_img = np.array(image_list)
    batch_label = np.array(label_list)

    return batch_img, batch_label, fns

def normalize(img):

    img = np.abs(img)
    img /= np.sum(img)
    img /= np.max(img)

    return img

def evaluate_and_plot(ori_pre, network, sess, dict_image, dict_dissim, dict_salmap, iterations):

    save_dir = '/results/11152017/attack_sal/'

    for i in range(iterations): # check each step

        if i == 0: # skip the first step, no perturbation at all
            continue

        predictions = sess.run(network.probs, feed_dict={network.images: dict_image[i]})

        if np.argmax(predictions) == ori_pre: # if the prediction doesn't change

            print("We find one!")

            img = normalize(dict_image[i])
            sal = normalize(dict_salmap[i])

            fig = plt.figure()

            gs = gridspec.GridSpec(1, 2, wspace=0.2, hspace=0.2)

            ax = fig.add_subplot(gs[0, 0])
            ax.imshow(img)
            ax.set_title('Ad_Input', fontsize=8)
            ax.tick_params(axis='both', which='major', labelsize=6)

            ax = fig.add_subplot(gs[0, 1])
            ax.imshow(sal)
            ax.set_title('SalMap with dissimilarity = {}'.format(dict_dissim[i]), fontsize=8)
            ax.tick_params(axis='both', which='major', labelsize=6)

            # save
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            plt.savefig(os.path.join(save_dir, "attack_{}.png".format(i)))


def main():

    num_iterations = 20
    step_size = 1e-4
    image_name = 'tabby'

    # load the image
    batch_img, batch_label, fns = data(image_name)

    sess = tf.Session()

    # prepare the network
    vgg = prepare_vgg('PlainSaliency', None, 'trained', sess)

    sal = sal_maxlogit(vgg)
    mass = center_mass(sal)

    ref_mass_val, predictions = sess.run([mass, vgg.probs], feed_dict={vgg.images: batch_img})
    original_pre = np.argmax(predictions)

    # the objective function D
    diff = mass - ref_mass_val
    D = tf.reduce_sum(tf.multiply(diff, diff))

    # gradient
    Dx = tf.gradients(vgg.maxlogit, vgg.images)

    # the signed gradient
    Dx_sign = tf.sign(Dx)

    # record the results for each iteration
    dict_step_to_image = {}
    dict_step_to_dissimilarity = {}
    dict_step_to_salmap = {}

    for step in range(num_iterations):

        print('Step {}'.format(step))

        if step == 0:
            dict_step_to_image[0] = batch_img
            dict_step_to_dissimilarity[0] = 0
            dict_step_to_salmap[0] = sess.run(sal_maxlogit(vgg), feed_dict={vgg.images: batch_img})
            continue

        D_val, Dx_sign_val, sal_map_val \
            = sess.run([D, Dx_sign, sal_maxlogit(vgg)], feed_dict={vgg.images: dict_step_to_image[step - 1]})

        dict_step_to_image[step] = dict_step_to_image[step - 1] + step_size * Dx_sign_val
        dict_step_to_salmap[step] = sal_map_val
        dict_step_to_dissimilarity[step] = D_val

    evaluate_and_plot(original_pre,
                      vgg,
                      sess,
                      dict_step_to_image,
                      dict_step_to_dissimilarity,
                      dict_step_to_salmap,
                      num_iterations)

    sess.close()


if __name__ == '__main__':
    # setup the GPUs to use
    os.environ['CUDA_VISIBLE_DEVICES'] = '5'
    main()