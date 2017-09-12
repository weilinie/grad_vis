import foolbox
from foolbox.models import TensorFlowModel
from scipy.misc import imread, imresize
import os
import numpy as np
import tensorflow as tf
from matplotlib import gridspec
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from vgg16 import Vgg16


def show_adv_imagenet(sess, x, y, img, label, preds, preds_adv, adv_x, train_dir):

    preds_val, preds_adv_val, adv_x_val = \
        sess.run([preds, preds_adv, adv_x], feed_dict={x: img, y: label})
    z1 = np.argmax(preds_val, axis=1)
    z2 = np.argmax(preds_adv_val, axis=1)

    if (z2 - z1).all():
        print('Not a successful attack for this batch')

    X_tmps = list([img, adv_x_val, img-adv_x_val])
    print('\nPlotting results')
    fig = plt.figure()
    gs = gridspec.GridSpec(3, 1, wspace=0.1, hspace=0.1)

    label = np.argmax(preds_adv_val, axis=1)
    prob = np.max(preds_adv_val, axis=1)
    for t in range(3):
            ax = fig.add_subplot(gs[t, 1])
            ax.imshow(X_tmps[t][1])
            ax.set_xticks([])
            ax.set_yticks([])
            if t == 2:
                ax.set_xlabel('{0} ({1:.2f})'.format(label, prob), fontsize=8)

    print('\nSaving figure')
    gs.tight_layout(fig)
    saveimg_dir = os.path.join(train_dir)
    if not os.path.exists(saveimg_dir):
        os.makedirs(saveimg_dir)
    plt.savefig(os.path.join(saveimg_dir, 'adv_00.png'))


def main():

    data_dir = "data"
    save_dir = "results_foolbox"


    # read in the prob image
    img1 = imread(os.path.join(data_dir, 'laska.png'), mode='RGB')
    img1 = imresize(img1, (224, 224))  # cut the image to 224 * 224
    img1 = img1.reshape([224, 224, 3])
    label1 = 356
    image_ex, label_ex = foolbox.utils.imagenet_example()

    # image and label placeholder
    imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    labels = tf.placeholder(tf.float32, [None, 1000])
    vgg16 = Vgg16(imgs)
    preds = vgg16.probs
    logits = vgg16.logits

    # tf session
    with tf.Session() as sess:

        vgg16.load_weights('vgg16_weights.npz', sess)
        model = TensorFlowModel(imgs, logits, bounds=(0, 255))
        print(np.argmax(model.predictions(img1)))

        # apply attack on source image
        attack = foolbox.attacks.IterativeGradientSignAttack(model)
        adv_ex = attack(img1[:,:,::-1], label1)
        print(np.argmax(model.predictions(adv_ex)))



        # vgg_adv = Vgg16(adv_x, 'vgg16_weights.npz', sess)
        # preds_adv = vgg_adv.probs
        #
        # # show adversarial images, clean images and adversarial perturbations
        # show_adv_imagenet(sess, imgs, labels, img1, label1, preds, preds_adv, adv_x, save_dir)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    main()



