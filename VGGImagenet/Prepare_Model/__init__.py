import tensorflow as tf
import os, sys
sys.path.append('/home/yang/open-convnet-black-box/VGGImagenet/Deep_Models/')
from vgg16 import Vgg16
from resnet import Resnet

from keras import backend as K
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50

from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops

@ops.RegisterGradient("GuidedRelu")
def _GuidedReluGrad(op, grad):
    return tf.where(0. < grad, gen_nn_ops._relu_grad(grad, op.outputs[0]), tf.zeros(tf.shape(grad)))

@ops.RegisterGradient("DeconvRelu")
def _GuidedReluGrad(op, grad):
    return tf.where(0. < grad, grad, tf.zeros(tf.shape(grad)))

weight_path = '/home/yang/open-convnet-black-box/VGGImagenet/vgg16_weights.npz'

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
        vgg.load_weights(weight_path, sess)

    elif load_weights == 'random':
        vgg.init(sess)

    elif load_weights == 'part':
        # fill the first "idx" layers with the trained weights
        # randomly initialize the rest
        vgg.load_weights_part(layer_idx * 2 + 1, weight_path, sess)

    elif load_weights == 'reverse':
        # do not fill the first "idx" layers with the trained weights
        # randomly initialize them
        vgg.load_weights_reverse(layer_idx * 2 + 1, weight_path, sess)

    elif load_weights == 'only':
        # do not load a specific layer ("idx") with the trained weights
        # randomly initialize it
        vgg.load_weights_only(layer_idx * 2 + 1, weight_path, sess)

    else:
        raise Exception("Unknown load_weights type - 1")

    return vgg

def prepare_resnet(sal_type, load_weights, sess, res_blocks=5):

    # construct the graph based on the gradient type we want
    if sal_type == 'GuidedBackprop':
        eval_graph = tf.get_default_graph()
        with eval_graph.gradient_override_map({'Relu': 'GuidedRelu'}):
            net = Resnet(res_blocks=res_blocks)

    elif sal_type == 'Deconv':
        eval_graph = tf.get_default_graph()
        with eval_graph.gradient_override_map({'Relu': 'DeconvRelu'}):
            net = Resnet(res_blocks=res_blocks)

    elif sal_type == 'PlainSaliency':
        net = Resnet(res_blocks=res_blocks)

    else:
        raise Exception("Unknown saliency_map type - 1")

    # different options for loading weights
    if load_weights == 'trained':
        raise Exception("Trained Resnet hasn't been implemented yet.")

    elif load_weights == 'random':
        net.init(sess)

    else:
        raise Exception("Unknown load_weights type - 1")

    return net

def prepare_keras_vgg16(sal_type, init, sess):

    K.set_session(sess)  # set Keras to use the given sess

    # construct the graph based on the gradient type we want
    if sal_type == 'GuidedBackprop':
        eval_graph = tf.get_default_graph()
        with eval_graph.gradient_override_map({'Relu': 'GuidedRelu'}):
            if init == 'random':
                vgg16 = VGG16(weights=None)
            else:
                vgg16 = VGG16(weights='imagenet')

    elif sal_type == 'Deconv':
        eval_graph = tf.get_default_graph()
        with eval_graph.gradient_override_map({'Relu': 'DeconvRelu'}):
            if init == 'random':
                vgg16 = VGG16(weights=None)
            else:
                vgg16 = VGG16(weights='imagenet')

    elif sal_type == 'PlainSaliency':
        if init == 'random':
            vgg16 = VGG16(weights=None)
        else:
            vgg16 = VGG16(weights='imagenet')

    else:
        raise Exception("Unknown saliency_map type - 1")

    if init == 'random':
        sess.run(tf.global_variables_initializer())

    return vgg16

def prepare_keras_resnet50(sal_type, init, sess):

    K.set_session(sess)  # set Keras to use the given sess
    K.set_learning_phase(0)

    # construct the graph based on the gradient type we want
    if sal_type == 'GuidedBackprop':
        eval_graph = tf.get_default_graph()
        with eval_graph.gradient_override_map({'Relu': 'GuidedRelu'}):
            if init == 'random':
                resnet50 = ResNet50(weights=None)
            else:
                resnet50 = ResNet50(weights='imagenet')

    elif sal_type == 'Deconv':
        eval_graph = tf.get_default_graph()
        with eval_graph.gradient_override_map({'Relu': 'DeconvRelu'}):
            if init == 'random':
                resnet50 = ResNet50(weights=None)
            else:
                resnet50 = ResNet50(weights='imagenet')

    elif sal_type == 'PlainSaliency':
        if init == 'random':
            resnet50 = ResNet50(weights=None)
        else:
            resnet50 = ResNet50(weights='imagenet')

    else:
        raise Exception("Unknown saliency_map type - 1")

    if init == 'random':
        sess.run(tf.global_variables_initializer())

    return resnet50



