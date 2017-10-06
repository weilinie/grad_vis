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
from utils import print_prob, visualize, visualize_yang

image_dict = {'tabby': 281, 'laska': 356, 'mastiff': 243}

@ops.RegisterGradient("GuidedRelu")
def _GuidedReluGrad(op, grad):
    return tf.where(0. < grad, gen_nn_ops._relu_grad(grad, op.outputs[0]), tf.zeros(tf.shape(grad)))

@ops.RegisterGradient("NGuidedRelu")
def _GuidedReluGrad(op, grad):
    return tf.where(0. > grad, gen_nn_ops._relu_grad(grad, op.outputs[0]), tf.zeros(tf.shape(grad)))

def super_saliency(tensor, inputs, num_to_viz):
    result = []
    shape = int(np.prod(tensor.get_shape()[1:]))
    tensor_flat = tf.reshape(tensor, [-1, shape])
    pick_indices = np.random.choice(shape, num_to_viz)
    for idx in pick_indices:
        result.append(tf.gradients(tensor_flat[:, idx], inputs)[0])
    return tf.stack(result)

def main():

    plain_init = False
    sal_map_type = "PlainSaliency_maxlogit"
    data_dir = "data_imagenet"
    save_dir = "results/10062017"

    # TODO: extend this part to a list

    image_name = 'laska'

    layers = [
              'conv1_1',
              'conv1_2',
              'pool1',
              'conv2_1',
              'conv2_2',
              'pool2',
              'conv3_1',
              'conv3_2',
              'conv3_3',
              'pool3',
              'conv4_1',
              'conv4_2',
              'conv4_3',
              'pool4',
              'conv5_1',
              'conv5_2',
              'conv5_3',
              'pool5',
              'fc1',
              'fc2',
              'fc3']

    fns = []
    image_list = []
    label_list = []

    # load in the original image and its adversarial examples
    for image_path in glob.glob(os.path.join(data_dir, '{}*.png'.format(image_name))):
        file_name = os.path.basename(image_path).split('.')[0]
        print('File name : {}').format(file_name)
        fns.append(file_name)
        image = imread(image_path, mode='RGB')
        image = imresize(image, (224, 224)).astype(np.float32)
        image_list.append(image)
        onehot_label = np.array([1 if i == image_dict[image_name] else 0 for i in range(1000)])
        label_list.append(onehot_label)

    batch_img = np.array(image_list)
    batch_label = np.array(label_list)
    batch_fns = fns

    batch_size = batch_img.shape[0]

    # tf session
    sess = tf.Session()

    # construct the graph based on the gradient type we want
    # plain relu vs guidedrelu
    if sal_map_type.split('_')[0] == 'GuidedBackprop':
        eval_graph = tf.get_default_graph()
        with eval_graph.gradient_override_map({'Relu': 'GuidedRelu'}):
                # load the vgg graph
                # plain_init = true -> load the graph with random weights
                # plain_init = false -> load the graph with pre-trained weights
                vgg = Vgg16('vgg16_weights.npz', plain_init, sess)

    elif sal_map_type.split('_')[0] == 'NGuidedBackprop':
        eval_graph = tf.get_default_graph()
        with eval_graph.gradient_override_map({'Relu': 'NGuidedRelu'}):
                # load the vgg graph
                # plain_init = true -> load the graph with random weights
                # plain_init = false -> load the graph with pre-trained weights
                vgg = Vgg16('vgg16_weights.npz', plain_init, sess)

    elif sal_map_type.split('_')[0] == 'PlainSaliency':
        # load the vgg graph
        # plain_init = true -> load the graph with random weights
        # plain_init = false -> load the graph with pre-trained weights
        vgg = Vgg16('vgg16_weights.npz', plain_init, sess)

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
    # num_to_viz = 20
    # for layer_name in layers:
    #
    #     saliencies = super_saliency(vgg.layers_dic[layer_name], vgg.imgs, num_to_viz)
    #     # shape = (num_to_viz, num_input_images, 224, 224, 3)
    #     saliencies_val = sess.run(saliencies, feed_dict={vgg.images: batch_img, vgg.labels: batch_label})
    #     # shape = (num_input_images, num_to_viz, 224, 224, 3)
    #     saliencies_val_trans = np.transpose(saliencies_val, (1, 0, 2, 3, 4))
    #
    #     for idx in range(batch_size):
    #         visualize_yang(batch_img[idx], num_to_viz, saliencies_val_trans[idx], layer_name, sal_map_type.split('_')[0], save_dir, fns[idx])

    fc2_ahats = []
    for idx, image in enumerate(batch_img):
        img = np.reshape(image, (1, 224, 224, 3))
        print('The image name : {}'.format(batch_fns[idx]))
        print('Predict class : {}'.format(
            class_names[
                np.argmax(
                    sess.run(vgg.probs, feed_dict={vgg.images: img})[0]
                )
            ])
        )
        fc2_firing = sess.run(vgg.layers_dic['fc2'], feed_dict={vgg.images: img})[0]
        fc2_ahat = np.sign(fc2_firing)
        fc2_ahats += [fc2_ahat]
        w_softmax = sess.run(vgg.layers_W_dic['fc3'])
        # w_pick = w_softmax[np.where(fc2_ahat == 1)]
        print('Ahat predict class : {}'.format(
            class_names[
                np.argmax(
                    np.dot(w_softmax.T, fc2_ahat)
                )
            ])
        )

    print(batch_fns)
    ori_image_idx = batch_fns.index(image_name)
    print(ori_image_idx)
    fc2_ahats = np.array(fc2_ahats)

    for idx, ahat in enumerate(fc2_ahats):
        if idx == ori_image_idx:
            print('The original image has {} columns'.format(np.sum(ahat)))
        else:
            print('The image name : {}'.format(batch_fns[idx]))

            plus = fc2_ahats[ori_image_idx] + ahat
            stay = np.where(plus == 2.)

            subtract = fc2_ahats[ori_image_idx] - ahat
            delete = np.where(subtract == 1.)

            add = np.where(subtract == -1.)

            print('Comparing to the original image,'
                  ' we have {} columns stay the same,'
                  ' {} columns deleted,'
                  ' {} columns added'.format(
                len(stay[0]),
                len(delete[0]),
                len(add[0]))
            )






if __name__ == '__main__':
    # setup the GPUs to use
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    main()
