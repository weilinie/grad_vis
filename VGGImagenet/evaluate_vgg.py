from scipy.misc import imread, imresize
import os
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops
import glob

from vgg16 import Vgg16
from utils import print_prob, visualize


image_dict = {'tabby': 281, 'laska': 356, 'mastiff': 243}


@ops.RegisterGradient("GuidedRelu")
def _GuidedReluGrad(op, grad):
    print('relu out: {}'.format(op.outputs[0]))
    return tf.where(0. < grad, gen_nn_ops._relu_grad(grad, op.outputs[0]), tf.zeros(tf.shape(grad)))


def main():
    sal_map_type = "GuidedBackprop_maxlogit"
    data_dir = "data_imagenet"
    save_dir = "results"
    name = 'laska'

    fns = []
    image_list = []
    label_list = []
    for image_path in glob.glob(os.path.join(data_dir, '{}*'.format(name))):
        fns.append(os.path.basename(image_path).split('.')[0])
        image = imread(image_path, mode='RGB')
        image = imresize(image, (224, 224)).astype(np.float32)
        image_list.append(image)

        onehot_label = np.array([1 if i == image_dict[name] else 0 for i in range(1000)])
        label_list.append(onehot_label)

    batch_img = np.array(image_list)
    batch_label = np.array(label_list)

    # # read in the prob image
    # img1 = imread(os.path.join(data_dir, 'laska.png'), mode='RGB')
    # img1 = imresize(img1, (224, 224))  # cut the image to 224 * 224
    # img1 = img1.reshape([1, 224, 224, 3])
    # onehot_label1 = np.array([1 if i == 243 else 0 for i in range(1000)])
    # label1 = onehot_label1.reshape(1, -1)
    #
    # img2 = imread(os.path.join(data_dir, 'demo.png'), mode='RGB')
    # img2 = imresize(img2, (224, 224))  # cut the image to 224 * 224
    # img2 = img2.reshape([1, 224, 224, 3])
    # onehot_label2 = np.array([1 if i == 356 else 0 for i in range(1000)])
    # label2 = onehot_label2.reshape(1, -1)
    #
    # img3 = imread(os.path.join(data_dir, 'adv_x_0.png'), mode='RGB')
    # img3 = imresize(img3, (224, 224))  # cut the image to 224 * 224
    # img3 = img3.reshape([1, 224, 224, 3])
    # onehot_label3 = np.array([1 if i == 243 else 0 for i in range(1000)])
    # label3 = onehot_label3.reshape(1, -1)
    #
    # img4 = imread(os.path.join(data_dir, 'adv_x_1.png'), mode='RGB')
    # img4 = imresize(img4, (224, 224))  # cut the image to 224 * 224
    # img4 = img4.reshape([1, 224, 224, 3])
    # onehot_label4 = np.array([1 if i == 356 else 0 for i in range(1000)])
    # label4 = onehot_label4.reshape(1, -1)
    #
    # batch_img = np.concatenate((img1, img2, img3, img4), 0)
    # batch_label = np.concatenate((label1, label2, label3, label4), 0)

    batch_size = batch_img.shape[0]

    # tf session
    sess = tf.Session()

    # image and label placeholder
    imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    labels = tf.placeholder(tf.float32, [None, 1000])

    if sal_map_type.split('_')[0] == 'GuidedBackprop':
        eval_graph = tf.get_default_graph()
        with eval_graph.gradient_override_map({'Relu': 'GuidedRelu'}):
            # load the vgg graph with the pre-trained weights
            vgg = Vgg16(imgs, 'vgg16_weights.npz', sess)
    elif sal_map_type.split('_')[0] == 'PlainSaliency':
        vgg = Vgg16(imgs, 'vgg16_weights.npz', sess)
    else:
        raise Exception("Unknown saliency_map type - 1")

    # cost
    cost = tf.reduce_sum((vgg.probs - labels) ** 2)

    # Get last convolutional layer gradient for generating gradCAM visualization
    target_conv_layer = vgg.pool5
    conv_grad = tf.gradients(cost, target_conv_layer)[0]

    # saliency gradient to input layer
    if sal_map_type.split('_')[1] == "cost":
        sal_map = tf.gradients(cost, imgs)[0]
    elif sal_map_type.split('_')[1] == 'maxlogit':
        sal_map = tf.gradients(vgg.maxlogit, imgs)[0]
    else:
        raise Exception("Unknown saliency_map type - 2")

    # Normalizing the gradients
    conv_grad_norm = tf.div(conv_grad, tf.norm(conv_grad) + tf.constant(1e-5))

    # predict
    probs = sess.run(vgg.probs, feed_dict={imgs: batch_img})

    # saliency map and gradients
    sal_map_val, target_conv_layer_val, conv_grad_norm_val = sess.run(
        [sal_map, target_conv_layer, conv_grad_norm],
        feed_dict={imgs: batch_img, labels: batch_label})

    for idx in range(batch_size):
        print_prob(probs[idx])
        visualize(batch_img[idx], target_conv_layer_val[idx], conv_grad_norm_val[idx], sal_map_val[idx],
                  sal_map_type, save_dir, fns[idx], probs[idx])


if __name__ == '__main__':
    # setup the GPUs to use
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    main()
