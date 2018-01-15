import os, sys
sys.path.append('/home/yang/open-convnet-black-box/VGGImagenet/')
import numpy as np
import tensorflow as tf
from Prepare_Model import prepare_vgg
from Prepare_Data import list_load
np.set_printoptions(threshold=np.nan)
from utils import  visualize_yang

sal_type = [
    'PlainSaliency',
    'Deconv',
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

model_type = [
    'trained',
    # 'random'
]

images = [
    'Dog_1.JPEG',
    'Dog_2.JPEG',
    'Dog_3.JPEG',
    'Dog_4.JPEG',
    'Dog_5.JPEG'
]

def super_saliency(tensor, inputs, num_to_viz):
    result = []
    shape = int(np.prod(tensor.get_shape()[1:]))
    tensor_flat = tf.reshape(tensor, [-1, shape])
    pick_indices = np.random.choice(shape, num_to_viz)
    for idx in pick_indices:
        result.append(tf.gradients(tensor_flat[:, idx], inputs)[0])
    return tf.stack(result)

def job(vgg, sal_type, sess, init, batch_img, fns):

    # first: pick one layer
    # second: pick num_to_viz neurons from this layer
    # third: calculate the saliency map w.r.t self.imgs for each picked neuron

    num_to_viz = 20
    for layer_name in layers:

        # shape = (num_to_viz, num_input_images, 224, 224, 3)
        # TF Graph
        saliencies = super_saliency(vgg.layers_dic[layer_name], vgg.imgs, num_to_viz)

        # shape = (num_input_images, num_to_viz, 224, 224, 3)
        saliencies_val = sess.run(saliencies, feed_dict={vgg.images: batch_img})
        saliencies_val_trans = np.transpose(saliencies_val, (1, 0, 2, 3, 4))

        for idx, name in enumerate(fns):
            save_dir = "results/{}/{}/{}/{}".format(name, init, sal_type, layer_name)
            visualize_yang(batch_img[idx], num_to_viz, saliencies_val_trans[idx], layer_name, sal_type, save_dir, name)

def main():

    for sal in sal_type:
        for init in model_type:
            tf.reset_default_graph()
            sess = tf.Session()
            vgg = prepare_vgg(sal, None, init, sess)

            batch_img, fns = list_load("./../data_imagenet", images)
            job(vgg, sal, sess, init, batch_img, fns)

            sess.close()


if __name__ == '__main__':
    # setup the GPUs to use
    os.environ['CUDA_VISIBLE_DEVICES'] = '5'
    main()
