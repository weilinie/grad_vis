import os, sys
sys.path.append('/home/yang/open-convnet-black-box/VGGImagenet/')
sys.path.append('/home/yang/open-convnet-black-box/VGGImagenet/Tools/')

from Prepare_Data import list_load
from Prepare_Model import prepare_keras_resnet50
from Plot import simple_plot

import tensorflow as tf

import numpy as np

sal_type = [
    'PlainSaliency',
    'Deconv',
    'GuidedBackprop'
]

images = [
    'Dog_4.JPEG',
    'tabby.png'
]

model = [
    'random',
    'trained'
]

def super_saliency(tensor, inputs, num_to_viz):

    result = []
    shape = int(np.prod(tensor.get_shape()[1:]))
    tensor_flat = tf.reshape(tensor, [-1, shape])
    pick_indices = np.random.choice(shape, num_to_viz)
    for idx in pick_indices:
        result.append(tf.gradients(tensor_flat[:, idx], inputs)[0])
    return tf.stack(result)

def main():

    batch_img, fns = list_load("./../data_imagenet", images)

    for sal in sal_type: # for each gradient type
        for init in model: # random or trained

            print(sal)
            print(init)

            tf.reset_default_graph() # erase the graph
            sess = tf.Session() # start a new session
            resnet50 = prepare_keras_resnet50(sal, init, sess)
            graph = tf.get_default_graph()

            input = graph.get_tensor_by_name('input_1:0')
            logits = graph.get_tensor_by_name('fc1000/BiasAdd:0')

            num_to_viz = 50

            # shape = (num_to_viz, num_input_images, 224, 224, 3)
            # TF Graph
            saliencies = super_saliency(logits, input, num_to_viz)

            # shape = (num_input_images, num_to_viz, 224, 224, 3)
            saliencies_val = sess.run(saliencies, feed_dict={input: batch_img})
            saliencies_val_trans = np.transpose(saliencies_val, (1, 0, 2, 3, 4))

            for idx, name in enumerate(fns):
                save_dir = "resnet_keras/{}/{}/{}/".format(name, init, sal)
                for index, image in enumerate(saliencies_val_trans[idx]):
                    simple_plot(image, name + str(index), save_dir)

            sess.close()


if __name__ == '__main__':
    # setup the GPUs to use
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    main()