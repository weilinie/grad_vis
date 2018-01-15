import os, sys
sys.path.append('/home/yang/open-convnet-black-box/VGGImagenet/')
import numpy as np
import tensorflow as tf
from Prepare_Model import prepare_vgg
from Prepare_Data import list_load
np.set_printoptions(threshold=np.nan)
from utils import simple_plot

sal_type = [
    'PlainSaliency',
    # 'Deconv',
    # 'GuidedBackprop'
]

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

images = [
    'Dog_1.JPEG',
    'Dog_2.JPEG',
    'Dog_3.JPEG',
    'Dog_4.JPEG',
    'Dog_5.JPEG'
]

def main():

    for sal in sal_type:
        for idx, layer in enumerate(layers):

            tf.reset_default_graph()
            sess = tf.Session()
            vgg = prepare_vgg(sal, idx, 'part', sess)

            batch_img, fns = list_load("./../data_imagenet", images)

            # TF Graph
            saliency = tf.gradients(vgg.maxlogit, vgg.imgs)[0]
            saliency_vals = sess.run(saliency, feed_dict={vgg.images: batch_img})

            for index, name in enumerate(fns):
                save_dir = 'results/{}/{}/{}'.format(name, sal, layer)
                simple_plot(saliency_vals[index], save_dir, layer)

            sess.close()


if __name__ == '__main__':
    # setup the GPUs to use
    os.environ['CUDA_VISIBLE_DEVICES'] = '6'
    main()
