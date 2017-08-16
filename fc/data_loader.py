import os
import glob
import random

import numpy as np
from numpy.linalg import inv
from scipy import misc


def to_one_hot(label):
    num_labels = len(np.unique(label))
    Y_onehot = np.eye(num_labels)[label]
    return Y_onehot


def from_one_hot(one_hot):
    return np.argmax(one_hot, axis=1)

def sparse_pattern_generator(input_size, sparse_ratio, num_pixels, num_ch):

    id = np.array([0 if i < (int(input_size * sparse_ratio) / num_ch) * num_ch else 1 for i in range(input_size)])
    reshaped = np.reshape(id, (num_pixels, 3))
    shuffled = np.random.permutation(reshaped)
    shape_back = np.reshape(shuffled, (input_size))

    return shape_back


def preprocessing(X, y, is_total_perm=False, is_pixel_perm=False, is_rand_sparse=False,
                  is_single_sparse=False, is_multi_sparse=False, sparse_ratio=0.0,
                  sparse_set_size=0, total_perm_mat=0, pixel_perm_mat=0):

    num_ch = 3
    input_size = X.shape[1]
    num_pixels = input_size / num_ch

    # completely permuting each image in the same way
    if is_total_perm:
        print('Total perm ... ')
        X = np.matmul(X, total_perm_mat)

    # permuting each image pixel-wisely in the same way
    elif is_pixel_perm:
        print('Pixel perm ... ')
        X = np.matmul(X, pixel_perm_mat)

    # sparse_ratio = num of zero pixels / num of pixels

    # randomly sparse each image differently
    elif is_rand_sparse:
        print('Random sparse ... ')
        for i in range(X.shape[0]): # for each image
            sparse_pattern = sparse_pattern_generator(input_size, sparse_ratio, num_pixels, num_ch)
            X[i] = np.multiply(X[i], sparse_pattern)

    # randomly sparse each image in the same way
    elif is_single_sparse:
        print('Single sparse ... ')
        sparse_pattern = sparse_pattern_generator(input_size, sparse_ratio, num_pixels, num_ch)
        for i in range(X.shape[0]): # for each image
            X[i] = np.multiply(X[i], sparse_pattern)

    # randomly sparse each image "differently"
    # the sparse pattern used is drawn from a finite set
    # different classes have different sets of sparse patterns
    elif is_multi_sparse:
        print('Multi sparse ... ')
        # prepare the sets
        sparse_set1 = [sparse_pattern_generator(input_size, sparse_ratio, num_pixels, num_ch)
                       for i in range(sparse_set_size)]
        sparse_set2 = [sparse_pattern_generator(input_size, sparse_ratio, num_pixels, num_ch)
                        for i in range(sparse_set_size)]

        for i in range(X.shape[0]): # for each image
            idx = np.random.choice(sparse_set_size)
            if y[i] == 0:
                X[i] = np.multiply(X[i], sparse_set1[idx])
            else:
                X[i] == np.multiply(X[i], sparse_set2[idx])

    return X


def read_image_data(image_folder, image_mode, train_test_ratio=0.8, shuffle=1, is_total_perm=False,
                    is_pixel_perm=False, is_rand_sparse=False, is_single_sparse=False, is_multi_sparse=False,
                    sparse_ratio=0, sparse_set_size=0):
    """ Read the data set and split them into training and test sets """
    X = []
    Label = []
    fns = []

    for image_path in glob.glob(os.path.join(image_folder, "*.png")):
        fns.append(os.path.basename(image_path))
        Label.append(int(os.path.basename(image_path).split("_")[0]))
        X.append(misc.imread(image_path, mode=image_mode).flatten())

    X = (np.array(X) / 255.).astype(np.float32)
    Label = np.array(Label)
    fns = np.array(fns)

    num_ch = 3
    input_size = X.shape[1]
    num_pixels = input_size / num_ch

    # total permutation matrix
    if is_total_perm:
        total_perm_mat = np.random.permutation(np.identity(input_size))
    else:
        total_perm_mat = np.identity(input_size)

    # pixel permutation matrix
    if is_pixel_perm:
        id = np.identity(input_size)
        reshaped = np.reshape(id, (num_pixels, num_ch, input_size))
        permuted = np.random.permutation(reshaped)
        pixel_perm_mat = np.reshape(permuted, (input_size, input_size))
    else:
        pixel_perm_mat = np.identity(input_size)

    X = preprocessing(X, Label, is_total_perm, is_pixel_perm, is_rand_sparse,
                      is_single_sparse, is_multi_sparse, sparse_ratio, sparse_set_size,
                      total_perm_mat, pixel_perm_mat)

    print X.shape

    # Convert into one-hot vectors
    Y_onehot = to_one_hot(Label)

    # randomly shuffle the inputs' order
    all_index = np.arange(X.shape[0])
    for _ in range(shuffle):
        np.random.shuffle(all_index)
    X = X[all_index, :]
    Y_onehot = Y_onehot[all_index, :]
    fns = fns[all_index]

    # prepare the train and test sets
    index_cutoff = int(X.shape[0] * train_test_ratio)

    return X[0:index_cutoff, :], X[index_cutoff:, :], \
           Y_onehot[0:index_cutoff, :], Y_onehot[index_cutoff:, :], \
           fns[0:index_cutoff], fns[index_cutoff:], \
            total_perm_mat, pixel_perm_mat
