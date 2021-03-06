import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt, gridspec

from imagenet_classes import class_names
from skimage.transform import resize


# returns the top1 string
def print_prob(prob):
    pred = (np.argsort(prob)[::-1])
    # Gegt top1 label
    top1 = [(pred[0], class_names[pred[0]], prob[pred[0]])] # pick the most likely class
    print("Top1: ", top1)

    # Get top5 label
    top5 = [(pred[i], class_names[pred[i]], prob[pred[i]]) for i in range(5)]
    print("Top5: ", top5)


def grad_cam(conv_output, conv_grad):
    output = conv_output  # [7,7,512]
    grads_val = conv_grad  # [7,7,512]

    weights = np.mean(grads_val, axis=(0, 1))  # [512]
    cam = np.ones(output.shape[0: 2], dtype=np.float32)  # [7,7]

    # Taking a weighted average
    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    # Passing through ReLU
    cam = np.maximum(cam, 0)
    cam = cam / np.max(cam)  # scale 0 to 1.0
    cam = resize(cam, (224, 224))
    return cam

def sal_pool_normalization(sal_pool):
        sal_pool -= np.min(sal_pool)
        sal_pool /= sal_pool.max()
        print(sal_pool.shape)

def visualize(image, conv_output, conv_grad, sal_map, sal_map_type, save_dir, fn, prob, layer_name):

    cam = grad_cam(conv_output, conv_grad)

    # normalizations
    sal_map -= np.min(sal_map)
    sal_map /= sal_map.max()

    img = image.astype(float)
    img -= np.min(img)
    img /= img.max()
    # print(img)

    guided_grad_cam = np.dstack((
        sal_map[:, :, 0] * cam,
        sal_map[:, :, 1] * cam,
        sal_map[:, :, 2] * cam,
    ))

    fig = plt.figure()

    gs = gridspec.GridSpec(1, 4, wspace=0.2, hspace=0.2)

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(img)
    ax.set_title('Input', fontsize=8)
    ax.tick_params(axis='both', which='major',  labelsize=6)

    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(cam)
    ax.set_title('{}: Grad-Cam'.format(layer_name), fontsize=8)
    ax.tick_params(axis='both', which='major',  labelsize=6)

    pred = (np.argsort(prob)[::-1])
    ax.set_xlabel('input: {}, pred_class: {} ({:.2f})'.
                  format(fn, class_names[pred[0]], prob[pred[0]]), fontsize=8)

    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(sal_map)
    ax.set_title('{}: {}'.format(layer_name, sal_map_type.split('_')[0]), fontsize=8)
    ax.tick_params(axis='both', which='major',  labelsize=6)

    ax = fig.add_subplot(gs[0, 3])
    ax.imshow(guided_grad_cam)
    ax.set_title('{}: guided Grad-Gam'.format(layer_name), fontsize=8)
    ax.tick_params(axis='both', which='major',  labelsize=6)

    # saved results path
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print('Saving {}_cam_{}.png'.format(sal_map_type, fn))
    plt.savefig(os.path.join(save_dir, "{}_cam_{}.png".format(sal_map_type, fn)))

def visualize_yang(batch_img, num_neurons, neuron_saliencies, layer_name, sal_type, save_dir, fn):

    # min = np.min(neuron_saliencies)
    # neuron_saliencies -= min
    # max = np.max(neuron_saliencies)
    # neuron_saliencies /= max

    for idx in range(num_neurons):

        dir = save_dir + '/{}'.format(layer_name)

        sal_map = neuron_saliencies[idx]

        min = np.min(sal_map)
        sal_map -= min
        max = np.max(sal_map)
        sal_map /= max

        plt.imshow(sal_map)

        if not os.path.exists(dir):
            os.makedirs(dir)
        plt.savefig(os.path.join(dir,
                                 "layer_name={}_idx={}_sal_type={}_image_info={}.png"
                                 .format(layer_name, idx, sal_type, fn)))
        plt.close()

def simple_plot(sal, save_dir, iteration):

    img = sal[0]

    min = np.min(img)
    img -= min
    max = np.max(img)
    img /= max

    plt.imshow(img)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.savefig(os.path.join(save_dir,
                             "iteration_{}.png"
                             .format(iteration)))
    plt.close()

    img *= 255.

    return sal

def diff_plot(rd, tr, dir):

    img_rd = rd[0]
    img_tr = tr[0]

    img_rd /= np.linalg.norm(img_rd)
    img_tr /= np.linalg.norm(img_tr)

    plt.imshow(img_rd - img_tr)

    if not os.path.exists(dir):
        os.makedirs(dir)

    plt.savefig(os.path.join(dir,
                             "diff.png"))

    plt.close()


