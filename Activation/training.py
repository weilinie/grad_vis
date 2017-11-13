from scipy.misc import imread, imresize
import os
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops
import glob
from Convnet import Convnet
from utils import print_prob, visualize, visualize_yang, simple_plot, diff_plot

image_dict = {'tabby': 281, 'laska': 356, 'mastiff': 243, 'restaurant': 762, 'hook': 600}

@ops.RegisterGradient("GuidedRelu")
def _GuidedReluGrad(op, grad):
    return tf.where(0. < grad, gen_nn_ops._relu_grad(grad, op.outputs[0]), tf.zeros(tf.shape(grad)))

@ops.RegisterGradient("DeconvRelu")
def _GuidedReluGrad(op, grad):
    return tf.where(0. < grad, grad, tf.zeros(tf.shape(grad)))

def job1(sal_map_type, vgg, sess):

    data_dir = "./../VGGImagenet/data_imagenet"

    layers = [
              'conv1_1',
              'fc1']

    image_name = 'tabby'
    save_dir = "results/11022017/{}/".format(image_name)

    image_list = []
    label_list = []
    fns = []

    # load in the original image and its adversarial examples
    for image_path in glob.glob(os.path.join(data_dir, '{}.png'.format(image_name))):
        file_name = os.path.basename(image_path).split('.')[0]
        fns.append(file_name)
        print('File name : {}').format(file_name)
        image = imread(image_path, mode='RGB')
        image = imresize(image, (224, 224)).astype(np.float32)
        image_list.append(image)
        onehot_label = np.array([1 if i == image_dict[image_name] else 0 for i in range(2)])
        label_list.append(onehot_label)

    batch_img = np.array(image_list)
    batch_label = np.array(label_list)

    batch_size = batch_img.shape[0]

    # first: pick one layer
    # second: pick num_to_viz neurons from this layer
    # third: calculate the saliency map w.r.t self.imgs for each picked neuron
    num_to_viz = 2
    for layer_name in layers:
        saliencies = super_saliency(vgg.layers_dic[layer_name], vgg.imgs, num_to_viz)
        # shape = (num_to_viz, num_input_images, 224, 224, 3)
        saliencies_val = sess.run(saliencies, feed_dict={vgg.images: batch_img, vgg.labels: batch_label})
        # shape = (num_input_images, num_to_viz, 224, 224, 3)
        saliencies_val_trans = np.transpose(saliencies_val, (1, 0, 2, 3, 4))

        for idx in range(batch_size):
            visualize_yang(batch_img[idx], num_to_viz, saliencies_val_trans[idx], layer_name, sal_map_type.split('_')[0], save_dir, fns[idx])

def job2(sal_map_type, vgg, sess):

    data_dir = "./../VGGImagenet/data_imagenet"

    image_name = 'tabby'
    save_dir = "results/10312017/tabby/iteration_GBP/"

    image_list = []
    label_list = []
    fns = []

    # load in the original image and its adversarial examples
    for image_path in glob.glob(os.path.join(data_dir, '{}.png'.format(image_name))):
        file_name = os.path.basename(image_path).split('.')[0]
        fns.append(file_name)
        print('File name : {}').format(file_name)
        image = imread(image_path, mode='RGB')
        image = imresize(image, (224, 224)).astype(np.float32)
        image_list.append(image)
        onehot_label = np.array([1 if i == image_dict[image_name] else 0 for i in range(2)])
        label_list.append(onehot_label)

    batch_img = np.array(image_list)
    batch_label = np.array(label_list)
    batch_size = batch_img.shape[0]

    # define the saliency map node in the graph
    saliency = tf.gradients(vgg.fc1l[0], vgg.images)[0]

    num_iterations = 20
    saliency_val = None
    for i in range(num_iterations):
        if i == 0:
            saliency_val = sess.run(saliency, feed_dict={vgg.images: batch_img})
            saliency_val = simple_plot(saliency_val, save_dir, i)
        else:
            saliency_val = sess.run(saliency, feed_dict={vgg.images: saliency_val})
            saliency_val = simple_plot(saliency_val, save_dir, i)

def job3(sal_map_type, vgg, sess):

    data_dir = "./../VGGImagenet/data_imagenet"

    image_name = 'tabby'
    save_dir = "results/11022017/{}/diff/".format(image_name)

    image_list = []
    label_list = []
    fns = []

    # load in the original image and its adversarial examples
    for image_path in glob.glob(os.path.join(data_dir, '{}.png'.format(image_name))):
        file_name = os.path.basename(image_path).split('.')[0]
        fns.append(file_name)
        print('File name : {}').format(file_name)
        image = imread(image_path, mode='RGB')
        image = imresize(image, (224, 224)).astype(np.float32)
        image_list.append(image)
        onehot_label = np.array([1 if i == image_dict[image_name] else 0 for i in range(2)])
        label_list.append(onehot_label)

    batch_img = np.array(image_list)
    batch_label = np.array(label_list)
    batch_size = batch_img.shape[0]

    # define the saliency map node in the graph
    saliency = tf.gradients(vgg.fc1l[0], vgg.images)[0]

    return sess.run(saliency, feed_dict={vgg.images: batch_img})

def super_saliency(tensor, inputs, num_to_viz):
    result = []
    shape = int(np.prod(tensor.get_shape()[1:]))
    tensor_flat = tf.reshape(tensor, [-1, shape])
    pick_indices = np.random.choice(shape, num_to_viz)
    for idx in pick_indices:
        result.append(tf.gradients(tensor_flat[:, idx], inputs)[0])
    return tf.stack(result)

def prepare_data(num_classes=2):

    dirs = ["./../data/Imagenet_Dogs", "./../data/Imagenet_Cats"]

    image_list = []
    label_list = []

    # load images
    for c, c_dir in enumerate(dirs):
        for image_path in glob.glob(os.path.join(c_dir, '*.JPEG')):
            image = imread(image_path, mode='RGB')
            image = imresize(image, (224, 224)).astype(np.float32)
            image_list.append(image)
            onehot_label = np.array([1 if i == c else 0 for i in range(num_classes)])
            label_list.append(onehot_label)

    images = np.array(image_list)
    labels = np.array(label_list)

    print(images.shape)
    print(labels.shape)

    # randomly shuffle the inputs
    all_index = np.arange(images.shape[0])
    np.random.shuffle(all_index)
    images = images[all_index, :]
    labels = labels[all_index, :]

    # prepare the training and testing sets
    index_cutoff = int(images.shape[0] * 0.8)

    X_train = images[0:index_cutoff, :]
    X_test = images[index_cutoff:, :]
    Y_train = labels[0:index_cutoff, :]
    Y_test = labels[index_cutoff:, :]

    # # preprocess by substracting the mean
    # train_mean = np.mean(X_train, axis=0)
    # X_train = X_train - train_mean
    # X_test = X_test - train_mean

    return X_train, X_test, Y_train, Y_test

def probing(if_train, if_rdinit):

    tf.reset_default_graph()

    sess = tf.Session()

    sal_map_type = "GuidedBackprop_maxlogit"

    # modify the graph based on the gradient type we want
    if sal_map_type.split('_')[0] == 'GuidedBackprop':

        eval_graph = tf.get_default_graph()
        with eval_graph.gradient_override_map({'Relu': 'GuidedRelu'}):
            vgg_random = Vgg_shallow(weights="models/best_model.ckpt", plain_init= True, sess=sess)
            sal_val_rd = job3(sal_map_type, vgg_random, sess)
            print("random done")
            sess.close()

        tf.reset_default_graph()
        sess = tf.Session()

        eval_graph = tf.get_default_graph()
        with eval_graph.gradient_override_map({'Relu': 'GuidedRelu'}):
            vgg_trained = Vgg_shallow(weights="models/best_model.ckpt", plain_init=False, sess=sess)
            sal_val_tr = job3(sal_map_type, vgg_trained, sess)
            print("trained done")
            sess.close()

        diff_plot(sal_val_rd, sal_val_tr, './')

    elif sal_map_type.split('_')[0] == 'Deconv':
        eval_graph = tf.get_default_graph()
        with eval_graph.gradient_override_map({'Relu': 'DeconvRelu'}):
            vgg = Vgg_shallow(weights="models/best_model.ckpt", plain_init= if_rdinit, sess=sess)
            job1(sal_map_type, vgg, sess)

    else:
        vgg = Vgg_shallow(weights="models/best_model.ckpt", plain_init= if_rdinit, sess=sess)
        job1(sal_map_type, vgg, sess)

def main():

    if_train = True

    Slu = True

    if_rdinit = False

    if if_train:

        # prepare the data
        X_train, X_test, Y_train, Y_test = prepare_data()

        # build the graph
        net = Convnet(Slu = Slu, num_classes=2)

        # Saver
        saver = tf.train.Saver()

        # start the sess
        sess = tf.Session()

        net.init(sess)  # random init for the training

        num_epochs = 5
        bs = 32

        for epoch in range(num_epochs):

            for b in range(int(len(X_train) / bs)):

                X_train_batch = X_train[bs * b: bs * b + bs]
                Y_train_batch = Y_train[bs * b: bs * b + bs]

                _, train_accu = \
                    sess.run([net.train_step, net.accuracy],
                             feed_dict={net.images: X_train_batch, net.labels: Y_train_batch})

                msg = "epoch = {}, batch = {}, " \
                      "train accuracy = {:.4f}".format(epoch, b, train_accu)

                print(msg)

            test_accu = 0.
            for b in range(int(len(X_train) / bs)):

                X_test_batch = X_test[bs * b: bs * b + bs]
                Y_test_batch = Y_test[bs * b: bs * b + bs]

                test_accu += \
                    np.sum(sess.run([net.correct_prediction],
                             feed_dict={net.images: X_test_batch, net.labels: Y_test_batch}))

            msg = "Test accuracy = {:.4f}".format(test_accu/len(X_test))
            print(msg)

        if not os.path.exists("models"):
            os.makedirs("models")

        if Slu:
            activation = 'Slu'
        else:
            activation = 'Relu'

        save_path = saver.save(sess, "models/best_model_{}.ckpt".format(activation))

        print(save_path)

        sess.close()

    # probing(if_train, if_rdinit)

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '5'
    main()