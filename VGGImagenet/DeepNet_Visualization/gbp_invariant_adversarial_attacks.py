import os, sys
sys.path.append('/home/yang/open-convnet-black-box/VGGImagenet/')
sys.path.append('/home/yang/open-convnet-black-box/VGGImagenet/Tools/')
import numpy as np
import tensorflow as tf
from Prepare_Model import prepare_vgg
from Prepare_Data import list_load
from imagenet_classes import class_names
from Plot import simple_plot

sal_type = [
    # 'PlainSaliency',
    'Deconv',
    'GuidedBackprop'
]

model_type = [
    'trained',
    # 'random'
]

images = [
    'laska',
    'laska_adv_SalMap_mis.png',
    'mastiff.png',
    'mastiff_adv_IterGS.png',
    'tabby.png',
    'tabby_adv_FGSM_topkmis.png'
]


def job(vgg, sal_type, sess, init, batch_img, fns):

    # TF Graph
    saliencies = tf.gradients(vgg.maxlogit, vgg.imgs)[0]

    # shape = (num_to_viz, 224, 224, 3)
    saliencies_val = sess.run(saliencies, feed_dict={vgg.images: batch_img})

    # predictions
    probs_val = sess.run(vgg.probs, feed_dict={vgg.images: batch_img})
    predictions = [class_names[np.argmax(vec)] for vec in probs_val]

    save_dir = './adv_results/'
    for idx, name in enumerate(fns):
        simple_plot(batch_img[idx], name, save_dir)
        simple_plot(saliencies_val[idx], name + '_{}'.format(predictions[idx]), save_dir)



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
    os.environ['CUDA_VISIBLE_DEVICES'] = '6'
    main()