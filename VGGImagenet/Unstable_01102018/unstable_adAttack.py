import os, sys
sys.path.append('/home/yang/open-convnet-black-box/VGGImagenet/')
import tensorflow as tf
from Similarity import similarity
from Prepare_Model import prepare_vgg
from Prepare_Data import list_load
import numpy as np

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
    # 'fc3'
]

def main():

    firing_arg = 'ahat'  # can be 'plain' or 'ahat'
    message = 'L2 Distance' if firing_arg == 'plain' else 'Changing Rate'

    tf.reset_default_graph()
    sess = tf.Session()
    vgg = prepare_vgg('PlainSaliency', None, 'trained', sess)

    batch_img, fns1 = list_load("./../data_imagenet", ["tabby.png"])
    batch_adv, fns2 = list_load("./../data_imagenet", ["tabby_adv_FGSM_topkmis.png"])

    result = []
    for layer in layers:
        temp = similarity(vgg.layers_dic[layer],
                       batch_img, batch_adv,
                       sess, vgg.imgs,
                       firing_arg)
        result.append(temp)

    print("Image_Name = {}, {} for Each Layer = {}".format("tabby.png", message, result))

    sess.close()

if __name__ == '__main__':
    # setup the GPUs to use
    os.environ['CUDA_VISIBLE_DEVICES'] = '5'
    main()