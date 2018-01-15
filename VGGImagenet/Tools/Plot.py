import numpy as np
import os
import matplotlib
from matplotlib import pyplot as plt

def simple_plot(image, name, save_dir):

    """
    Save a single image with the given name to a given directory
    :param image: the image to save
    :param name: the name to save with 
    :param save_dir: the directory to save to 
    :return: 
    """

    # a very regular normalization
    min = np.min(image)
    image -= min
    max = np.max(image)
    image /= max

    plt.imshow(image)

    # make the directory if it's not there yet
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.savefig(os.path.join(save_dir, "{}.png".format(name)))

    plt.close()
