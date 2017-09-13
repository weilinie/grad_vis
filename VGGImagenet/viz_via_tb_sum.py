from scipy.misc import imread, imresize
from imagenet_classes import class_names
import os
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops

from vgg16 import vgg16

"""
This viz technique is deprecated!
"""

@ops.RegisterGradient("GuidedRelu")
def _GuidedReluGrad(op, grad):
    return tf.where(0. < grad, gen_nn_ops._relu_grad(grad, op.outputs[0]), tf.zeros(tf.shape(grad)))


def saliency_gradient(target, imgs, guidedback, name):
    saliencies = tf.gradients(target, imgs)[0]
    saliency_maps = tf.reshape(saliencies, [-1, 224, 224, 3])
    return tf.summary.image('Saliency_guidedback_{}_{}'.format(guidedback, name), saliency_maps, max_outputs=5)


def main():

    guidedback = True

    # read in the prob image
    img1 = imread('laska.png', mode='RGB')
    batch_img = imresize(img1, (224, 224))  # cut the image to 224 * 224
    onehot_label = np.array([1 if i == 244 else 0 for i in range(1000)])
    batch_label = onehot_label.reshape(1, -1)

    # tf session
    sess = tf.Session()

    # summary path
    summary_path = "summary"
    if not os.path.exists(summary_path):
        os.makedirs(summary_path)
    writer = tf.summary.FileWriter(summary_path, sess.graph)

    # image and label placeholder
    imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    labels = tf.placeholder(tf.float32, [None, 1000])

    if guidedback:
        eval_graph = tf.get_default_graph()
        with eval_graph.gradient_override_map({'Relu': 'GuidedRelu'}):
            # load the vgg graph with the pre-trained weights
            vgg = vgg16(imgs, 'vgg16_weights.npz', sess)
    else:
        # load the vgg graph with the pre-trained weights
        vgg = vgg16(imgs, 'vgg16_weights.npz', sess)

    # predict
    prob = sess.run(vgg.probs, feed_dict={vgg.imgs: [batch_img]})[0]
    pred = (np.argsort(prob)[::-1])[0]  # pick the most likely class
    print(class_names[pred], prob[pred])

    # cost
    cost = tf.reduce_sum((vgg.probs - batch_label) ** 2)

    # saliency on maxlogit and cost
    writer.add_summary(sess.run(saliency_gradient(vgg.maxlogit, vgg.imgs, guidedback, 'maxlogit'),
                                feed_dict={vgg.imgs: [batch_img]}))
    writer.add_summary(sess.run(saliency_gradient(cost, vgg.imgs, guidedback, 'cost'),
                                feed_dict={vgg.imgs: [batch_img]}))


if __name__ == '__main__':
    # setup the GPUs to use
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    main()
