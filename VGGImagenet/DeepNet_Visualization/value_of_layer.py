import os, sys
sys.path.append('/home/yang/open-convnet-black-box/VGGImagenet/')
sys.path.append('/home/yang/open-convnet-black-box/VGGImagenet/Tools/')
import tensorflow as tf
from Prepare_Model import prepare_vgg
from Prepare_Data import list_load
from Plot import simple_plot

sal_type = [
    # 'PlainSaliency',
    # 'Deconv',
    'GuidedBackprop'
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
    'Dog_5.JPEG',
    'tabby.png',
    'laska.png'
]

def main():

    for sal in sal_type:
        for idx, layer in enumerate(layers):

            tf.reset_default_graph()
            sess = tf.Session()
            vgg = prepare_vgg(sal, idx, 'only', sess)

            batch_img, fns = list_load("./../data_imagenet", images)

            # TF Graph
            saliency = tf.gradients(vgg.maxlogit, vgg.imgs)[0]
            saliency_vals = sess.run(saliency, feed_dict={vgg.images: batch_img})

            for index, name in enumerate(fns):
                save_dir = 'value_only/{}/{}'.format(name, layer)
                simple_plot(saliency_vals[index], name + '_only_' + layer, save_dir)

            sess.close()


if __name__ == '__main__':
    # setup the GPUs to use
    os.environ['CUDA_VISIBLE_DEVICES'] = '5'
    main()
