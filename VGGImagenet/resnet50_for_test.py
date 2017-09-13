import foolbox
import keras
import numpy as np
from keras.applications.resnet50 import ResNet50
import os
from matplotlib import gridspec
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def softmax_np(x, axis=None):
    return np.exp(x) / np.sum(np.exp(x), axis=axis)

def main():

    train_dir = 'for_test_results'

    # instantiate model
    keras.backend.set_learning_phase(0)
    kmodel = ResNet50(weights='imagenet')
    preprocessing = (np.array([104, 116, 123]), 1)
    fmodel = foolbox.models.KerasModel(kmodel, bounds=(0, 255), preprocessing=preprocessing)

    # get source image and label
    image, label = foolbox.utils.imagenet_example()
    print('label: {}'.format(label))

    preds = fmodel.predictions(image)
    label_orig = np.argmax(preds)
    prob_orig = np.max(softmax_np(preds))
    print('labels_orig: {} ({:.2f})'.format(label_orig, prob_orig))

    # apply attack on source image
    attack = foolbox.attacks.FGSM(fmodel)
    adversarial = attack(image[:,:,::-1], label)

    preds_adv = fmodel.predictions(adversarial)
    label_adv = np.argmax(preds_adv)
    prob_adv = np.max(softmax_np(preds_adv))
    print('labels_adv: {} ({:.2f})'.format(label_adv, prob_adv))

    X_tmps = [image / 255., adversarial[:,:,::-1] / 255., (adversarial[:,:,::-1] - image) / 255.]

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
                          format(label, label_orig, prob_orig, label_adv, prob_adv), fontsize=8)

    print('\nSaving figure')
    gs.tight_layout(fig)
    saveimg_dir = os.path.join(train_dir)
    if not os.path.exists(saveimg_dir):
        os.makedirs(saveimg_dir)
    plt.savefig(os.path.join(saveimg_dir, 'adv_00.png'))


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    main()

