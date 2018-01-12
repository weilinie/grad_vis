import numpy as np
import glob
import os, sys
from scipy.misc import imread, imresize

def list_load(data_dir, names, size=(224, 224)):

    # To load a list of images
    # You need to specify the image directory and the image name with the extension
    # We assume the image format is RGB
    # The image will be resized to 224 * 224 * 3 by default

    fns = []
    image_list = []

    for name in names:
        path = os.path.join(data_dir, '{}'.format(name))
        file_name = os.path.basename(path).split('.')[0]
        print('File name : {}').format(file_name)
        fns.append(file_name)

        image = imread(path, mode='RGB')
        image = imresize(image, size).astype(np.float32)
        image_list.append(image)

    batch_img = np.array(image_list) # put into a batch

    return batch_img, fns