from scipy.misc import imread, imresize
import os
import numpy as np
import tensorflow as tf
import sys
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops
from imagenet_classes import class_names
np.set_printoptions(threshold=np.nan)
import glob

from vgg16 import Vgg16
from utils import print_prob, visualize, visualize_yang

image_dict = {'tabby': 281, 'laska': 356, 'mastiff': 243}

def top5_indices(array):
    return array[1].argsort()[-5:][::-1]

def main():

    plain_init = False
    data_dir = "data_imagenet"

    image_names = ['laska', 'tabby', 'mastiff']

    # tf session
    sess = tf.Session()
    vgg = Vgg16('vgg16_weights.npz', plain_init, sess)
    w_softmax = sess.run(vgg.layers_W_dic['fc3'])

    f = open('log_invariance_adversarial.txt', 'w')
    sys.stdout = f

    for image_name in image_names:

        fns = []
        image_list = []
        # load in the original image and its adversarial examples
        for image_path in glob.glob(os.path.join(data_dir, '{}*.png'.format(image_name))):
            file_name = os.path.basename(image_path).split('.')[0]
            fns.append(file_name)
            image = imread(image_path, mode='RGB')
            image = imresize(image, (224, 224)).astype(np.float32)
            image_list.append(image)

        batch_img = np.array(image_list)
        batch_fns = fns

        fc2_ahats = []
        fc2_firings = []

        for idx, image in enumerate(batch_img):

            img = np.reshape(image, (1, 224, 224, 3))

            fc2_firing = sess.run(vgg.layers_dic['fc2'], feed_dict={vgg.images: img})[0]

            fc2_firings.append(fc2_firing)
            fc2_ahat = np.sign(fc2_firing)
            fc2_ahats.append(fc2_ahat)

        ori_image_idx = batch_fns.index(image_name)

        fc2_ahats = np.array(fc2_ahats)
        fc2_firings = np.array(fc2_firings)

        for idx, ahat in enumerate(fc2_ahats):

            if idx == ori_image_idx:
                print('The original image has {} columns/firngs'.format(np.sum(ahat)))
                print('*******************************************************************************')

            else:

                print('Original prediction : {}'.format(
                    [class_names[i] for i in
                     top5_indices(
                            np.dot(fc2_firings[idx], w_softmax)
                        )
                    ])
                )

                print('Ahat prediction : {}'.format(
                    [class_names[i] for i in
                     top5_indices(
                            np.dot(ahat, w_softmax)
                        )
                    ])
                )

                print('The image name : {}'.format(batch_fns[idx]))

                plus = fc2_ahats[ori_image_idx] + ahat

                remain_idx = np.where(plus == 2.)[0] # the indices of the remaining columns

                remain = np.zeros(ahat.shape)
                remain[remain_idx] = 1

                # the deleting will be positive ones
                # the adding will be negative ones
                subtract = fc2_ahats[ori_image_idx] - ahat

                delete_idx = np.where(subtract == 1.)[0]
                add_idx = np.where(subtract == -1.)[0]

                delete = np.zeros(ahat.shape)
                delete[delete_idx] = 1

                add = np.zeros(ahat.shape)
                add[add_idx] = 1

                print('Compare to the original image, the {} :'.format(batch_fns[idx]))
                print('Added {} extra columns, which predict to {}'.format(
                    len(add_idx),
                    [class_names[i] for i in
                     top5_indices(
                            np.dot(add.T, w_softmax))
                    ])
                )
                print('Deleted {} existing columns, which predict to {}'.format(
                    len(delete_idx),
                    [class_names[i] for i in
                     top5_indices(
                            np.dot(delete.T, w_softmax))
                    ])
                )
                print('The two share {} columns, which predict to {}'.format(
                    len(remain_idx),
                    [class_names[i] for i in
                     top5_indices(
                            np.dot(remain.T, w_softmax))
                    ])
                )

                print('*******************************************************************************')

    f.close()



if __name__ == '__main__':
    # setup the GPUs to use
    os.environ['CUDA_VISIBLE_DEVICES'] = '4'
    main()
