from scipy.misc import imread, imresize
import os, sys
import tensorflow as tf
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
from imagenet_classes import class_names

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

    data_dir = "../data/{}/".format(folder_name)
    save_dir = "results/10052017/"

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

    # reference_image = batch_img[0]

    # tf session
    sess = tf.Session()
    vgg_trained = Vgg16('vgg16_weights.npz', False, sess)

    ratio = 0.8

    firings = []
    for image in batch_img:
        img = np.reshape(image, (1, 224, 224, 3))
        firings += [sess.run(vgg_trained.layers_dic['fc2'], feed_dict={vgg_trained.imgs: img})]
    firings = np.array(firings)

    f = open('log_invariance_{}.txt'.format(ratio), 'w')
    sys.stdout = f

    ahats = np.sign(firings)
    reduce = np.sum(ahats, axis=0)[0]
    common_pattern = np.where(reduce >= batch_size * ratio)

    if len(common_pattern) == 0:
        print('No common pattern!')

    else:

        w_softmax = sess.run(vgg_trained.layers_W_dic['fc3'])
        w_pick = w_softmax[common_pattern]
        print('common pattern class : {}'.format(
            class_names[
                np.argmax(
                    np.sum(w_pick, axis=0)
                )
            ]
        ))

        # for w in w_pick:
        #     print(class_names[np.argmax(w)])

        for idx, ahat in enumerate(ahats):
            ahat_ori = np.copy(ahat[0])

            ahat_left = ahat[0]
            ahat_left[common_pattern] = 0

            ahat_common = ahat_ori - ahat_left

            w_not_pick = w_softmax[np.where(ahat_left == 1)]
            w_pick = w_softmax[np.where(ahat_common == 1)]
            w_ori = w_softmax[np.where(ahat_ori == 1)]

            print('For iamge {},'
                  ' the original firings sum to {},'
                  ' the common firings sum to {},'
                  ' the rest firings sum to : {}'.format(
                idx,
                class_names[
                    np.argmax(
                        np.sum(w_ori, axis=0)
                    )
                ],
                class_names[
                    np.argmax(
                        np.sum(w_pick, axis=0)
                    )
                ],
                class_names[
                    np.argmax(
                        np.sum(w_not_pick, axis=0)
                    )
                ])
            )


    # vgg_not_trained = Vgg16('vgg16_weights.npz', True, sess)

    # firing_arg = 'ahat' # can be 'plain' or 'ahat'
    # similarity_arg = 'jaccard' # can be 'allclose' or 'jaccard'

    # for layer in layers: # for each layer
    #
    #     if 'pool' not in layer:
    #
    #         w_trained = sess.run(vgg_trained.layers_W_dic[layer])
    #         w_not_trained = sess.run(vgg_not_trained.layers_W_dic[layer])
    #         volumn = np.prod(w_trained.shape)
    #
    #         Diff = np.linalg.norm(np.subtract(w_trained, w_not_trained)) / volumn
    #
    #         print('Layer name = {}, Diff = {}'.format(layer, Diff))

        # firing_states_trained = []
        # firing_states_diff = []
        # for image in batch_img: # for each image
        #
        #     img = np.reshape(image, (1, 224, 224, 3))
        #
        #     firing_state_trained =\
        #         np.sign(sess.run(vgg_trained.layers_dic[layer], feed_dict={vgg_trained.imgs: img})).flatten()
        #
        #     firing_state_not_trained = \
        #         np.sign(sess.run(vgg_not_trained.layers_dic[layer], feed_dict={vgg_not_trained.imgs: img})).flatten()
        #
        #     firing_state_diff = np.abs(np.subtract(firing_state_trained, firing_state_not_trained))
        #
        #     firing_states_trained += [firing_state_trained]
        #     firing_states_diff += [firing_state_diff]
        #
        # firing_states_trained = np.array(firing_states_trained)
        # firing_states_diff = np.array(firing_states_diff)
        #
        # print('Layer name : {}'.format(layer))
        # print('Number of images : {}'.format(firing_states_trained.shape[0]))
        # print('Unique sparse patterns : {}'.format(np.unique(firing_states_trained, axis=0).shape[0]))
        # print('Sparse ratio : {}'.format(np.mean(firing_states_trained, axis=1)))
        # print('Diff : {}'.format(np.mean(firing_states_diff, axis=1)))

    # for i in range(batch_size):
    #     result = []
    #     firing_states = []
    #     for layer in layers:
    #         temp = compare(vgg_trained.layers_dic[layer],
    #                        reference_image, batch_img[i],
    #                        sess, vgg_trained.imgs,
    #                        firing_arg, similarity_arg)
    #         result.append(temp)
    #     print("Image_idx = {}, Invariance = {}".format(i, result))
    #     probs_val = sess.run(vgg_trained.probs, feed_dict={vgg_trained.imgs: np.reshape(batch_img[i], (1, 224, 224, 3))})
    #     print("Predict class : {}".format(class_names[np.argmax(probs_val)]))

    f.close()


if __name__ == '__main__':
    # setup the GPUs to use
    os.environ['CUDA_VISIBLE_DEVICES'] = '6'
    main()