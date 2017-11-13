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

def predict_top_n(array, n=5):
    temp = array[0].argsort()[-n:][::-1]
    return [class_names[i] for i in temp]

def experiment_common_features(sess, batch_fns, batch_img, vgg_trained, batch_size, folder_name):

    # firings = []
    # for image in batch_img:
    #     img = np.reshape(image, (1, 224, 224, 3))
    #     firings.append(sess.run(vgg_trained.layers_dic['fc2'], feed_dict={vgg_trained.imgs: img}))
    # firings = np.array(firings)

    firings_fc2 = sess.run(vgg_trained.layers_dic['fc2'], feed_dict={vgg_trained.images: batch_img})

    w_softmax = sess.run(vgg_trained.layers_W_dic['fc3'])

    for ratio in np.arange(0.8, 1.0, 0.1):

        f = open('log_invariance_{}_{}.txt'.format(folder_name, ratio), 'w')

        holder = sys.stdout

        sys.stdout = f

        ahats_fc2 = np.sign(firings_fc2)

        reduce_fc2 = np.sum(ahats_fc2, axis=0)

        common_pattern_idx_fc2 = np.where(reduce_fc2 >= batch_size * ratio)[1]

        print('By setting the threshold to {}, we have {} common columns for fc2.'.format(ratio, len(common_pattern_idx_fc2)))

        print('*********************************************************************************')

        print('Individually, for fc2, each column predicts to:')

        for idx in common_pattern_idx_fc2:
            print('Column {} predicts {}'.format(idx, predict_top_n([w_softmax[idx]])))

        print('*********************************************************************************')

        common_pattern = np.zeros(reduce.shape)
        common_pattern[:, common_pattern_idx_fc2] = 1
        prediction = predict_top_n(np.dot(common_pattern, w_softmax))

        print('The common columns in fc2 together predict : {}'.format(prediction))

        print('*********************************************************************************')

        for idx, ahat in enumerate(ahats_fc2):

            print('Image name = {}'.format(batch_fns[idx]))

            prediction_firing = predict_top_n(np.dot(firings_fc2[idx], w_softmax))
            print('The original prediction is {}.'.format(prediction_firing))

            ahat_ori = np.copy(ahat)
            ahat_left = ahat
            ahat_left[:, common_pattern_idx_fc2] = 0
            ahat_common = ahat_ori - ahat_left

            prediction_ahat = predict_top_n(np.dot(ahat_ori, w_softmax))
            print('The original ahat prediction is {}.'.format(prediction_ahat))

            prediction = predict_top_n(np.dot(ahat_left, w_softmax))
            print('The image specific ahat prediction is {}.'.format(prediction))

            prediction = predict_top_n(np.dot(ahat_common, w_softmax))
            print('The image common ahat prediction is {}.'.format(prediction))

            print('*********************************************************************************')

        f.close()

        sys.stdout = holder

def experiment_ahat_prediction(sess, batch_fns, batch_img, vgg_trained, batch_size, folder_name):

    firings_fc1 = sess.run(vgg_trained.layers_dic['fc1'], feed_dict={vgg_trained.images: batch_img})
    ahats_fc1 = np.sign(firings_fc1)
    


def main():

    is_BR = False

    folders = ['Imagenet_Dogs',
               'Imagenet_Cats']

    for folder_name in folders:

        data_dir = "../data/{}/".format(folder_name)

        # save_dir = "results/10072017/"
        #
        # layers = [
        #           'conv1_1',
        #           'conv1_2',
        #           'pool1',
        #           'conv2_1',
        #           'conv2_2',
        #           'pool2',
        #           'conv3_1',
        #           'conv3_2',
        #           'conv3_3',
        #           'pool3',
        #           'conv4_1',
        #           'conv4_2',
        #           'conv4_3',
        #           'pool4',
        #           'conv5_1',
        #           'conv5_2',
        #           'conv5_3',
        #           'pool5',
        #           'fc1',
        #           'fc2',
        #           'fc3']

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
        batch_img = np.array(image_list)
        batch_size = batch_img.shape[0]

        if is_BR:
            sort_indices = np.argsort(batch_fns)
            batch_fns = batch_fns[sort_indices]
            batch_img = batch_img[sort_indices]

        # tf session
        sess = tf.Session()
        vgg_trained = Vgg16('vgg16_weights.npz', False, sess)
        # vgg_not_trained = Vgg16('vgg16_weights.npz', True, sess)

        # Experiment 1
        experiment_common_features(sess, batch_fns, batch_img, vgg_trained, batch_size, folder_name)

        # Experiment 2
        experiment_ahat_prediction(sess, batch_fns, batch_img, vgg_trained, batch_size, folder_name)

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


if __name__ == '__main__':
    # setup the GPUs to use
    os.environ['CUDA_VISIBLE_DEVICES'] = '5'
    main()