from scipy.misc import imread, imresize
import os
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops
import glob

from vgg16 import Vgg16
from utils import print_prob, visualize, visualize_yang


image_dict = {'tabby': 281, 'laska': 356, 'mastiff': 243}


@ops.RegisterGradient("GuidedRelu")
def _GuidedReluGrad(op, grad):
    # print('relu out: {}'.format(op.outputs[0]))
    return tf.where(0. < grad, gen_nn_ops._relu_grad(grad, op.outputs[0]), tf.zeros(tf.shape(grad)))

def super_saliency(tensor, inputs, num_to_viz):
    result = []
    shape = int(np.prod(tensor.get_shape()[1:]))
    tensor_flat = tf.reshape(tensor, [-1, shape])
    pick_indices = np.random.choice(shape, num_to_viz)
    for idx in pick_indices:
        result.append(tf.gradients(tensor_flat[:, idx], inputs)[0])
    return tf.stack(result)

def main():

    sal_map_type = "GuidedBackprop_maxlogit"
    data_dir = "data_imagenet"
    save_dir = "results"

    # TODO: extend this part to a list
    image_name = 'laska'
    layer_name = 'pool3'

    fns = []
    image_list = []
    label_list = []
    # load in the original image and its adversarial examples
    for image_path in glob.glob(os.path.join(data_dir, '{}*'.format(image_name))):
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

    # construct the graph based on the gradient type we want
    # plain relu vs guidedrelu
    if sal_map_type.split('_')[0] == 'GuidedBackprop':
        eval_graph = tf.get_default_graph()
        with eval_graph.gradient_override_map({'Relu': 'GuidedRelu'}):
            # load the vgg graph with the pre-trained weights
            vgg = Vgg16('vgg16_weights.npz', sess)
    elif sal_map_type.split('_')[0] == 'PlainSaliency':
        vgg = Vgg16('vgg16_weights.npz', sess)
    else:
        raise Exception("Unknown saliency_map type - 1")

    # Get last convolutional layer gradient for generating gradCAM visualization
    target_conv_layer = vgg.pool5
    if sal_map_type.split('_')[1] == "cost":
        conv_grad = tf.gradients(vgg.cost, target_conv_layer)[0]
    elif sal_map_type.split('_')[1] == 'maxlogit':
        conv_grad = tf.gradients(vgg.maxlogit, target_conv_layer)[0]
    else:
        raise Exception("Unknown saliency_map type - 2")
    # normalization
    conv_grad_norm = tf.div(conv_grad, tf.norm(conv_grad) + tf.constant(1e-5))

    # saliency gradient to input layer
    if sal_map_type.split('_')[1] == "cost":
        sal_map = tf.gradients(vgg.cost, vgg.imgs)[0]
    elif sal_map_type.split('_')[1] == 'maxlogit':
        sal_map = tf.gradients(vgg.maxlogit, vgg.imgs)[0]
    else:
        raise Exception("Unknown saliency_map type - 2")

    # predict
    probs = sess.run(vgg.probs, feed_dict={vgg.images: batch_img})

    # sal_map and conv_grad
    sal_map_val, target_conv_layer_val, conv_grad_norm_val =\
        sess.run([sal_map, target_conv_layer, conv_grad_norm],
                 feed_dict={vgg.images: batch_img, vgg.labels: batch_label})

    for idx in range(batch_size):
        print_prob(probs[idx])
        visualize(batch_img[idx], target_conv_layer_val[idx], conv_grad_norm_val[idx], sal_map_val[idx],
                  sal_map_type, save_dir, fns[idx], probs[idx])

    # first: pick one layer
    # second: pick num_to_viz neurons from this layer
    # third: calculate the saliency map w.r.t self.imgs for each picked neuron
    num_to_viz = 5
    saliencies = super_saliency(vgg.layers_dic[layer_name], vgg.imgs, num_to_viz)
    # shape = (num_to_viz, num_input_images, 224, 224, 3)
    saliencies_val = sess.run(saliencies, feed_dict={vgg.images: batch_img, vgg.labels: batch_label})
    # shape = (num_input_images, num_to_viz, 224, 224, 3)
    saliencies_val_trans = np.transpose(saliencies_val, (1, 0, 2, 3, 4))

    for idx in range(batch_size):
        visualize_yang(batch_img[idx], saliencies_val_trans[idx], layer_name, sal_map_type.split('_')[0], save_dir, fns[idx])





if __name__ == '__main__':
    # setup the GPUs to use
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    main()
