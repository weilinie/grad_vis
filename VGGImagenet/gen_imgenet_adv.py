from logging import warning

import foolbox
from foolbox.models import TensorFlowModel
from foolbox.criteria import Misclassification, TopKMisclassification
from scipy.misc import imread, imresize, imsave
import os
import numpy as np
import tensorflow as tf
from matplotlib import gridspec
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from vgg16 import Vgg16


image_dict = {'tabby': 281, 'laska': 356, 'mastiff': 243}
attack_list = ['FGSM', 'IterGS', 'SalMap']
criteria_dict = {'topkmis': TopKMisclassification(k=10), 'mis': Misclassification()}

name1 = 'tabby'
attack_type = "FGSM"
criterion_type = "topkmis"


def softmax_np(x, axis=None):
    return np.exp(x) / np.sum(np.exp(x), axis=axis)


def main():

    data_dir = "data_imagenet"
    train_dir = "adv_results_vgg16"

    # read in the prob image
    img1 = imread(os.path.join(data_dir, '{}.png'.format(name1)), mode='RGB')
    img1 = imresize(img1, (224, 224)).astype(np.float32)  # cut the image to 224 * 224
    label1 = image_dict[name1]

    # image placeholder
    imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    vgg16 = Vgg16(imgs)
    logits = vgg16.logits

    # tf session
    with tf.Session() as sess:

        vgg16.load_weights('vgg16_weights.npz', sess)
        fmodel = TensorFlowModel(imgs, logits, bounds=(0, 255))

        print('true label: {}'.format(label1))

        preds = fmodel.predictions(img1)
        label_orig = np.argmax(preds)
        prob_orig = np.max(softmax_np(preds))
        print('labels_orig: {} ({:.2f})'.format(label_orig, prob_orig))

        # apply attack on source image
        print("Attacking {} in {} with criterion ({})...".format(name1, attack_type, criterion_type))
        if attack_type == "FGSM":
            attack = foolbox.attacks.FGSM(fmodel, criterion=criteria_dict[criterion_type])
        elif attack_type == "IterGS":
            attack = foolbox.attacks.IterativeGradientSignAttack(fmodel, criterion=criteria_dict[criterion_type])
        elif attack_type == "SalMap":
            attack = foolbox.attacks.SaliencyMapAttack(fmodel, criterion=criteria_dict[criterion_type])
        else:
            warning("Unknown attack type!")
            attack = foolbox.attacks.FGSM(fmodel, criterion=criteria_dict[criterion_type])

        adversarial = attack(img1, label1)

        preds_adv = fmodel.predictions(adversarial)
        label_adv = np.argmax(preds_adv)
        prob_adv = np.max(softmax_np(preds_adv))
        print('labels_adv: {} ({:.2f})'.format(label_adv, prob_adv))

        X_tmps = [img1 / 255., adversarial / 255., (adversarial - img1) / 255.]

        print('\nPlotting results')
        fig = plt.figure()
        gs = gridspec.GridSpec(3, 1, wspace=0.1, hspace=0.1)

        for t in range(3):
            ax = fig.add_subplot(gs[t, 0])
            ax.imshow(X_tmps[t])
            ax.set_xticks([])
            ax.set_yticks([])
            if t == 2:
                ax.set_xlabel('true_label: {}, label_orig: {} ({:.2f}) to label_adv: {} ({:.2f})'.
                              format(label1, label_orig, prob_orig, label_adv, prob_adv), fontsize=8)

        print('\nSaving figure to ' + train_dir)
        gs.tight_layout(fig)
        saveimg_dir = os.path.join(train_dir)
        if not os.path.exists(saveimg_dir):
            os.makedirs(saveimg_dir)
        plt.savefig(os.path.join(saveimg_dir, '{}_3_{}_{}.png'.format(name1, attack_type, criterion_type)))

        imsave(os.path.join(data_dir, '{}_adv_{}_{}.png'.format(name1, attack_type,criterion_type)), adversarial)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    main()



