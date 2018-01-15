import os, sys
sys.path.append('/home/yang/open-convnet-black-box/VGGImagenet/')
import numpy as np
import tensorflow as tf
from Prepare_Model import prepare_vgg
from Prepare_Data import list_load
np.set_printoptions(threshold=np.nan)

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
    'fc3'
]

model_type = [
    'trained',
    'random'
]

def sparse_ratio(vgg, sess, layer_name, image_name, h_idx=None, v_idx=None):

    """
    Notice that the sparse ratio will be calculated w.r.t the entire batch!
    """

    # get the target layer tensor
    if h_idx == None and v_idx == None:
        target_tensor = vgg.layers_dic[layer_name]

    # get the target "depth row" from the layer tensor
    # corresponding to one image patch filtered by all the filters
    elif h_idx != None and v_idx != None:
        target_tensor = vgg.layers_dic[layer_name][:, h_idx, v_idx]

    else:
        raise Exception("Error in sparse_ratio !")

    batch_img, fns = list_load("./../data_imagenet", [image_name])

    target_tensor_val = sess.run(target_tensor, feed_dict={vgg.images: batch_img})
    target_tensor_val[target_tensor_val > 0] = 1.0
    return np.sum(target_tensor_val) / np.size(target_tensor_val)

def main():

    for init in model_type:
        tf.reset_default_graph()
        sess = tf.Session()
        vgg = prepare_vgg('PlainSaliency', None, init, sess)
        for layer in layers:
            result = sparse_ratio(vgg, sess, layer, 'tabby.png')
            print('The sparse ratio of layer {} with {} weights is {}'.format(layer, init, result))
        sess.close()

if __name__ == '__main__':
    # setup the GPUs to use
    os.environ['CUDA_VISIBLE_DEVICES'] = '6'
    main()