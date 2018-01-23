import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt, gridspec

def simple_plot(image, name, save_dir, cmap=None):

    """
    Save a single image with the given name to a given directory
    :param image: the image to save
    :param name: the name to save with 
    :param save_dir: the directory to save to 
    :return: 
    """

    # a very simple normalization
    min = np.min(image)
    image -= min
    max = np.max(image)
    image /= max

    plt.imshow(image, cmap=cmap)

    # make the directory if it's not there yet
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.savefig(os.path.join(save_dir, "{}.png".format(name)))

    plt.close()

def grid_plot(grid_shape, image_list, title, save_dir, save_name):

    """
    Assumption 1: # of grids = # of images
    Assumption 2: row oriented

    :param grid_shape: 2D shape [rows, columns]
    :param image_list: a list of images to be plotted
    :param title: the title for this grid image
    :param save_dir: where to save this grid image
    :param save_name: what name to save with
    :return:
    """

    if np.prod(grid_shape) != len(image_list):
        raise ValueError('Grid Shape does not match the number of images!')

    rows = grid_shape[0]
    columns = grid_shape[1]

    plt.title(title, loc='left')
    fig = plt.figure()
    gs = gridspec.GridSpec(rows, columns, wspace=0.2, hspace=0.2)

    for i in range(rows):
        for j in range(columns):

            ax = fig.add_subplot(gs[i, j])

            image = image_list[i * columns + j]

            # a very simple normalization
            min = np.min(image)
            image -= min
            max = np.max(image)
            image /= max

            ax.imshow(image)
            ax.tick_params(axis='both', which='major',  labelsize=6)



    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.savefig(os.path.join(save_dir, "{}.png".format(save_name)))