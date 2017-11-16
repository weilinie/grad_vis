from scipy.misc import imread, imresize
import os
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops
from imagenet_classes import class_names
np.set_printoptions(threshold=np.nan)
import glob

from vgg16 import Vgg16
from utils import print_prob, visualize, visualize_yang, simple_plot

image_dict = {'tabby': 281, 'laska': 356, 'mastiff': 243, 'restaurant': 762, 'hook': 600}
sal_type = ['PlainSaliency', 'Deconv', 'GuidedBackprop']
layers = [
    'conv1_1',
    'conv1_2',
    'conv2_1',
    'conv2_2',
    'conv3_1',
    'conv3_2',
    'conv3_3',
    'conv4_1',
    'conv4_2',
    'conv4_3',
    'conv5_1',
    'conv5_2',
    'conv5_3',
    'fc1',
    'fc2',
    'fc3']

@ops.RegisterGradient("GuidedRelu")
def _GuidedReluGrad(op, grad):
    return tf.where(0. < grad, gen_nn_ops._relu_grad(grad, op.outputs[0]), tf.zeros(tf.shape(grad)))

@ops.RegisterGradient("DeconvRelu")
def _GuidedReluGrad(op, grad):
    return tf.where(0. < grad, grad, tf.zeros(tf.shape(grad)))

def data(image_name):

    data_dir = "data_imagenet"

    fns = []
    image_list = []
    label_list = []

    # load in the original image and its adversarial examples
    for image_path in glob.glob(os.path.join(data_dir, '{}.jpg'.format(image_name))):
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

def super_saliency(tensor, inputs, num_to_viz):
    result = []
    shape = int(np.prod(tensor.get_shape()[1:]))
    tensor_flat = tf.reshape(tensor, [-1, shape])
    pick_indices = np.random.choice(shape, num_to_viz)
    for idx in pick_indices:
        result.append(tf.gradients(tensor_flat[:, idx], inputs)[0])
    return tf.stack(result)

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

def job1(vgg, sal_type, sess, init, image_name):

    batch_img, batch_label, fns = data(image_name)

    # first: pick one layer
    # second: pick num_to_viz neurons from this layer
    # third: calculate the saliency map w.r.t self.imgs for each picked neuron
    num_to_viz = 20
    for layer_name in layers:

        save_dir = "results/11132017/new_normalization/job1/{}/{}/{}/{}".format(image_name, init, sal_type, layer_name)

        saliencies = super_saliency(vgg.layers_dic[layer_name], vgg.images, num_to_viz)
        # shape = (num_to_viz, num_input_images, 224, 224, 3)
        saliencies_val = sess.run(saliencies, feed_dict={vgg.images: batch_img, vgg.labels: batch_label})
        # shape = (num_input_images, num_to_viz, 224, 224, 3)
        saliencies_val_trans = np.transpose(saliencies_val, (1, 0, 2, 3, 4))

        visualize_yang(batch_img[0], num_to_viz, saliencies_val_trans[0], layer_name, sal_type, save_dir, fns[0])

def job2(vgg, image_name, sess):

    batch_img, batch_label, fns = data(image_name)

    saliency = tf.gradients(vgg.maxlogit, vgg.images)[0]

    return sess.run(saliency, feed_dict={vgg.images: batch_img, vgg.labels: batch_label})

def sparse_ratio(vgg, sess, layer_name, image_name, h_idx=None, v_idx=None):

    """
    Notice that the sparse ratio will be calculated w.r.t the entire batch!
    """

    # get the target layer tensor
    if h_idx == None and v_idx == None:
        target_tensor = vgg.layers_dic[layer_name]

    # get the target "depth row" from the layer tensor
    # corresponding to one image patch filtered by all the filters
    elif h_idx != None and v_idx != None:
        target_tensor = vgg.layers_dic[layer_name][:, h_idx, v_idx]

    else:
        raise Exception("Error in sparse_ratio !")

    batch_img, batch_label, fns = data(image_name)

    target_tensor_val = sess.run(target_tensor, feed_dict={vgg.images: batch_img, vgg.labels: batch_label})
    target_tensor_val[target_tensor_val > 0] = 1.0
    return np.sum(target_tensor_val) / np.size(target_tensor_val)

def main():

    for image_name in ['Dog_1', 'Dog_2', 'Dog_3', 'Dog_4', 'Dog_5']:
        for sal in sal_type:
            for init in ['trained', 'random']:
                tf.reset_default_graph()
                sess = tf.Session()
                vgg = prepare_vgg(sal, None, init, sess)
                job1(vgg, sal, sess, init, image_name)
                sess.close()

    # for init in ['trained', 'random']:
    #     tf.reset_default_graph()
    #     sess = tf.Session()
    #     vgg = prepare_vgg('PlainSaliency', None, init, sess)
    #     result = sparse_ratio(vgg, sess, 'fc1', 'tabby')
    #     print('The sparse ratio of layer FC1 with {} weights is {}'.format(init, result))
    #     sess.close()

    # for image_name in ['Dog_1', 'Dog_2', 'Dog_3', 'Dog_4', 'Dog_5']:
    #     for sal in sal_type:
    #         save_dir = 'results/11132017/new_normalization/{}/{}/'.format(image_name, sal)
    #         for idx, layer in enumerate(layers):
    #             tf.reset_default_graph()
    #             sess = tf.Session()
    #             vgg = prepare_vgg(sal, idx, 'part', sess)
    #             saliency_val = job2(vgg, image_name, sess)
    #             simple_plot(saliency_val, save_dir, layer)
    #             sess.close()



if __name__ == '__main__':
    # setup the GPUs to use
    os.environ['CUDA_VISIBLE_DEVICES'] = '6'
    main()
