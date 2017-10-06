from scipy.misc import imread, imresize
import os, sys
import tensorflow as tf
import glob
from imagenet_classes import class_names
import numpy as np
from sklearn.metrics import jaccard_similarity_score

from vgg16 import Vgg16
from utils import print_prob, visualize, visualize_yang

def compare(layer, image1, image2, sess, pl_holder_X, firing_arg, similarity_arg):

    result = 1 # by default they are different

    if firing_arg == 'plain':
        image1 = np.reshape(image1, (1, 224, 224, 3))
        image2 = np.reshape(image2, (1, 224, 224, 3))
        layer_val_1 = sess.run(layer, feed_dict={pl_holder_X: image1})
        layer_val_2 = sess.run(layer, feed_dict={pl_holder_X: image2})
        if similarity_arg == 'allclose':
            result = int(not np.allclose(layer_val_1, layer_val_2))
        elif similarity_arg == 'jaccard':
            raise ValueError('With real firing values, we cannot calculate the jaccard similarity !!!')
        else:
            raise ValueError('Wrong similarity argument!!!')


    elif firing_arg == 'ahat':
        image1 = np.reshape(image1, (1, 224, 224, 3))
        image2 = np.reshape(image2, (1, 224, 224, 3))
        layer_val_1 = np.sign(sess.run(layer, feed_dict={pl_holder_X: image1}))
        layer_val_2 = np.sign(sess.run(layer, feed_dict={pl_holder_X: image2}))
        if similarity_arg == 'allclose':
            result = int(not np.allclose(layer_val_1, layer_val_2))
        elif similarity_arg == 'jaccard':
            layer_bool_1 = layer_val_1.astype(np.bool)
            layer_bool_2 = layer_val_2.astype(np.bool)
            if layer_bool_1.shape != layer_bool_2.shape:
                raise ValueError("Shape Mismatch !!! ")
            intersection = np.logical_and(layer_bool_1, layer_bool_2)
            union = np.logical_or(layer_bool_1, layer_bool_2)
            result = 1 - intersection.sum() / float(union.sum()) # 0 represent high similarity
        else:
            raise ValueError('Wrong similarity argument!!!')

    else:
        raise ValueError('Wrong firing argument !!!')

    return result


def main():

    folder_name = 'Imagenet_Dogs'

    plain_init = False
    data_dir = "../data/{}/".format(folder_name)
    save_dir = "results/10022017"

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

    for image_path in glob.glob(os.path.join(data_dir, '*.JPEG')):
        file_name = os.path.basename(image_path)
        file_name = file_name.split('.')[0]
        file_name = file_name.split('_')[1]
        fns.append(int(file_name))
        image = imread(image_path, mode='RGB')
        image = imresize(image, (224, 224)).astype(np.float32)
        image_list.append(image)

    batch_fns = np.array(fns)
    sort_indices = np.argsort(batch_fns)
    batch_fns = batch_fns[sort_indices]
    batch_img = np.array(image_list)
    batch_img = batch_img[sort_indices]
    batch_size = batch_img.shape[0]

    reference_image = batch_img[0]

    # tf session
    sess = tf.Session()
    vgg = Vgg16('vgg16_weights.npz', plain_init, sess)

    # firing state for fc1
    firing_states = []
    for image in batch_img: # for each image
        img = np.reshape(image, (1, 224, 224, 3))
        # the input for the FC layers
        # the sparse pattern
        firing_state = np.sign(sess.run(vgg.layers_dic['fc1'], feed_dict={vgg.imgs: img})).flatten()
        firing_states += [firing_state]
    firing_states = np.array(firing_states)
    print('Total number of examples : {}'.format(firing_states.shape[0]))
    print('Unique sparse patterns : {}'.format(np.unique(firing_states, axis=0).shape[0]))


    # firing_arg = 'ahat'  # can be 'plain' or 'ahat'
    # similarity_arg = 'jaccard'  # can be 'allclose' or 'jaccard'

    # f = open('log_{}_{}_{}.txt'.format(folder_name, firing_arg, similarity_arg), 'w')
    # sys.stdout = f

    # for i in range(batch_size):
    #     result = []
    #     firing_states = []
    #     for layer in layers:
    #         temp = compare(vgg.layers_dic[layer],
    #                        reference_image, batch_img[i],
    #                        sess, vgg.imgs,
    #                        firing_arg, similarity_arg)
    #         result.append(temp)
    #     print("Image_idx = {}, Invariance = {}".format(i, result))
    #     probs_val = sess.run(vgg.probs, feed_dict={vgg.imgs: np.reshape(batch_img[i], (1, 224, 224, 3))})
    #     print("Predict class : {}".format(class_names[np.argmax(probs_val)]))

    # f.close()


if __name__ == '__main__':
    # setup the GPUs to use
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    main()