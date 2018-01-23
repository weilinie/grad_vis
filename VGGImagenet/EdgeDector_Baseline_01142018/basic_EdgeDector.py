import os, sys
sys.path.append('/home/yang/open-convnet-black-box/VGGImagenet/')
sys.path.append('/home/yang/open-convnet-black-box/VGGImagenet/Tools')
from Prepare_Data import list_load
from Plot import simple_plot
import numpy as np

images = [
    'Dog_1.JPEG',
    'Dog_2.JPEG',
    'Dog_3.JPEG',
    'Dog_4.JPEG',
    'Dog_5.JPEG',
    'tabby.png'
]

def main():

    save_dir = './results/'

    batch_imgs, fns = list_load("./../data_imagenet", images)

    shape = batch_imgs.shape  # [none, 224, 224, 3]

    for idx, image in enumerate(batch_imgs):

        # to gray
        image = 0.2989 * image[:, :, 0] + 0.5870 * image[:, :, 1] + 0.1140 * image[:, :, 2]

        copy = np.copy(image)

        for i in range(shape[2]):

            if i == 0:
                image[:, i] = 0

            else:
                image[:, i] -= copy[:, i - 1]

        simple_plot(image, fns[idx] + '_edge_gray', save_dir)


if __name__ == '__main__':
    # setup the GPUs to use
    os.environ['CUDA_VISIBLE_DEVICES'] = '5'
    main()
