import os, sys
sys.path.append('/home/yang/open-convnet-black-box/VGGImagenet/')
import numpy as np
import tensorflow as tf
from imagenet_classes import class_names
from Prepare_Model import prepare_vgg
from Prepare_Data import list_load
np.set_printoptions(threshold=np.nan)

sal_type = [
    # 'PlainSaliency',
    # 'Deconv',
    'GuidedBackprop'
]

images = [
    'Dog_1.JPEG',
    'Dog_2.JPEG',
    'Dog_3.JPEG',
    'Dog_4.JPEG',
    'Dog_5.JPEG',
    'tabby.png',
    'mastiff.png',
    'laska.png'
]

def main():

    for sal in sal_type:

        tf.reset_default_graph()
        sess = tf.Session()
        vgg = prepare_vgg(sal, None, 'trained', sess)

        batch_img, fns = list_load("./../data_imagenet", images)

        # TF Graph
        saliency = tf.gradients(vgg.maxlogit, vgg.imgs)[0]

        saliency_vals, prob_vals = sess.run([saliency, vgg.probs], feed_dict={vgg.images: batch_img})

        for idx, sal in enumerate(saliency_vals):

            # normalize
            min = np.min(sal)
            sal -= min
            max = np.max(sal)
            sal /= max
            sal *= 225

            print(class_names[np.argmax(prob_vals[idx])])

        prob_vals2 = sess.run(vgg.probs, feed_dict={vgg.images: saliency_vals})

        for prob in prob_vals2:

            print(class_names[np.argmax(prob)])

        sess.close()


if __name__ == '__main__':
    # setup the GPUs to use
    os.environ['CUDA_VISIBLE_DEVICES'] = '5'
    main()