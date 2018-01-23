from scipy.misc import imread, imresize
import os, sys
sys.path.append('/home/yang/open-convnet-black-box/VGGImagenet/')
from Prepare_Model import prepare_vgg
from Prepare_Data import list_load
import numpy as np
import tensorflow as tf
np.set_printoptions(threshold=np.nan)
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt, gridspec

def sal_maxlogit(network, sal_type, target_conv_layer):

    """
    the network will be a vgg object
    saliency map must be consistent with the visualization
    """

    if sal_type == 'abs':
        sal_ori = tf.gradients(network.maxlogit, network.images)[0]
        sal_abs = tf.abs(sal_ori)
        sal_norm1 = sal_abs / tf.reduce_sum(sal_abs)
        return sal_norm1 / tf.reduce_max(sal_norm1)

    elif sal_type == 'plain':
        sal_ori = tf.gradients(network.maxlogit, network.images)[0]
        sal_shift = sal_ori - tf.reduce_min(sal_ori)
        return sal_shift / tf.reduce_max(sal_shift)

    elif sal_type == 'gradcam':

        # the target conv layer
        tcl = network.layers_dic[target_conv_layer] # [none, w, h, num_filters]

        # shape and dimension
        shape = tcl.get_shape().as_list()
        num_filters = shape[-1]
        w = shape[1]
        h = shape[2]
        
        # the gradient w.r.t the target conv layer
        tcl_grad = tf.gradients(network.maxlogit, tcl)[0]
        # normalize
        tcl_grad_norm = tcl_grad / tf.norm(tcl_grad) # [none, w, h, num_filters]

        # the importance of the filters (called weights here)
        weights_temp1 = tf.reduce_mean(tcl_grad_norm, axis=1, keep_dims=True) # [none, 1, h, num_filters]
        weights_temp2 = tf.reduce_mean(weights_temp1, axis=2, keep_dims=True) # [none, 1, 1, num_filters]
        weights = tf.reshape(weights_temp2, [-1, num_filters]) # [none, num_filters]

        # calculate the cam
        indices = tf.constant(range(num_filters))
        # a list of tensors
        cam_list = tf.map_fn(lambda x: weights[0][x] * tcl[0][:, :, x], indices, dtype=tf.float32)
        # stack to one single tensor
        cam_matrix = tf.stack(cam_list) # [num_filters, w, h]
        # reduce sum
        cam_2D = tf.reduce_sum(cam_matrix, axis=0) # [w, h]
        # reshape to include the channel
        cam_reshape = tf.reshape(cam_2D, [w, h, 1]) # [w, h, 1] # need this channel 1 in order to resize
        # Passing through ReLU
        cam_relu = tf.nn.relu(cam_reshape)
        # normalize
        cam_norm = cam_relu / tf.reduce_max(cam_relu)
        # resize to [224, 224, 1]
        cam_resize = tf.image.resize_images(cam_norm, [224, 224]) # [224, 224, 1]
        # reshape back to [224, 224]
        cam = tf.reshape(cam_resize, [224, 224])



        sal_ori = tf.gradients(network.maxlogit, network.images)[0] # [none, 224, 224, 3]

        # the "abs saliency map"
        # sal_abs = tf.abs(sal_ori)
        # # normalize
        # sal_norm = sal_abs / tf.reduce_sum(sal_abs)
        # sal = sal_norm / tf.reduce_max(sal_norm)

        # the plain saliency map
        sal_shifted = sal_ori - tf.reduce_min(sal_ori)
        sal = sal_shifted / tf.reduce_max(sal_shifted)

        # mask two together
        indices = tf.constant(range(3)) # three channels
        result_list = tf.map_fn(lambda i : tf.multiply(sal[0][:, :, i], cam), indices, dtype=tf.float32) # a list of [224, 224]
        grad_cam_matrix = tf.stack(result_list) # [3, 224, 224]
        grad_cam_trans = tf.transpose(grad_cam_matrix, [1, 2, 0])
        grad_cam = tf.expand_dims(grad_cam_trans, axis=0)

        # print(grad_cam.get_shape().as_list())

        return grad_cam / tf.reduce_max(grad_cam)

def helper(idx, shape, flat):

    i = tf.mod(idx, shape[0])
    j = tf.mod((idx - i * shape[0]), shape[1])
    k = idx - i * shape[0] - j * shape[1]

    return tf.multiply(flat[idx], tf.cast(tf.convert_to_tensor([i, j, k]), tf.float32))

def center_mass(tensor_batch_sal):

    """
    Another node in the graph to calculate the center mass of the saliency (tensor_batch_sal).
    Notice that the saliency is a node in the graph
    """

    # take the sal out of the batch
    sal = tensor_batch_sal[0]
    shape = sal.get_shape().as_list() # usually (224, 224, 3)

    # flat the tensor
    flat = tf.reshape(sal, [-1])

    len = flat.get_shape().as_list()[0]

    tensor_list = tf.map_fn(lambda x : helper(x, shape, flat), tf.constant(range(len)), dtype=tf.float32)

    tensor = tf.concat(tensor_list, axis=0)

    return tf.reduce_sum(tensor, axis=0)

def sal_diff(diff_type, network, batch_img, sal, sess):

    if diff_type == 'centermass':

        # the criteria, a function of the saliency map
        mass = center_mass(sal)
        # the reference value of the criteria on the original image without perturbation
        ref_val = sess.run(mass, feed_dict={network.images: batch_img})
        # the difference
        diff = mass - ref_val # this is a graph node
        D = tf.sqrt(tf.reduce_sum(tf.multiply(diff, diff)))

    if diff_type == 'plain':

        # In this case, we define no criteria on the saliency map, but use it directly instead

        # the reference value of the criteria on the original image without perturbation
        ref_val = sess.run(sal, feed_dict={network.images: batch_img})
        # the difference
        diff = sal - ref_val
        D = tf.sqrt(tf.reduce_sum(tf.multiply(diff, diff)))

    return D

def evaluate(image_name, diff_type, gradient_type,
             dict_image, dict_dissim, dict_salmap, dict_predictions, dict_perturb, iterations):

    save_dir = 'results/11212017/sal_attack_{}_{}_{}/'.format(image_name, diff_type, gradient_type)

    for i in range(iterations): # check each step

        if dict_predictions[i] == dict_predictions[0]: # if the prediction doesn't change

            print("We find one!")

            img = dict_image[i][0]
            perturbation = dict_perturb[i][0]

            perturbation_norm = np.max(perturbation)

            img -= np.min(img)
            img /= np.max(img)

            perturbation -= np.min(perturbation)
            perturbation /= np.max(perturbation)

            sal = dict_salmap[i][0]

            fig = plt.figure()

            gs = gridspec.GridSpec(1, 3, wspace=0.2, hspace=0.2)

            ax = fig.add_subplot(gs[0, 0])
            ax.imshow(img)
            ax.set_title('Ad_Input_{}'.format(i), fontsize=8)
            ax.tick_params(axis='both', which='major', labelsize=6)

            ax = fig.add_subplot(gs[0, 1])
            ax.imshow(perturbation)
            ax.set_title('Perturbation with norm = {}'.format(perturbation_norm), fontsize=8)
            ax.tick_params(axis='both', which='major', labelsize=6)

            ax = fig.add_subplot(gs[0, 2])
            ax.imshow(sal)
            ax.set_title('SalMap with dissimilarity = {}'.format(dict_dissim[i]), fontsize=8)
            ax.tick_params(axis='both', which='major', labelsize=6)

            # save
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            plt.savefig(os.path.join(save_dir, "attack{}_{}_{}_{}.png".format(i, image_name, diff_type, gradient_type)))

            plt.close()

def main():

    num_iterations = 100
    step_size = 1e-1
    image_name = 'Dog_1.JPEG'

    # how we would like the "saliency map" be different
    diff_type = 'plain' # 'centermass', 'plain'

    # define the special gradient for the "saliency map" calculation if necessary
    gradient_type = 'PlainSaliency' # 'PlainSaliency', 'GuidedBackprop'

    # how we would like to visualize the result gradient
    viz_type = 'gradcam' # 'abs', 'plain', 'gradcam'

    # for gradcam only
    target_layer = 'pool5'

    # load the image
    batch_img, fns = list_load("./../data_imagenet", [image_name])

    sess = tf.Session()

    # prepare the networks
    vgg_attack = prepare_vgg(gradient_type, 'softplus', 'maxpool', None, 'trained', sess) # used for attack
    vgg = prepare_vgg(gradient_type, 'relu', 'maxpool', None, 'trained', sess) # used for probing

    print('Two Networks Prepared ... ')

    sal = sal_maxlogit(vgg_attack, viz_type, target_layer)
    D = sal_diff(diff_type, vgg_attack, batch_img, sal, sess)

    # gradient
    Dx = tf.gradients(D, vgg_attack.images)[0]

    # the signed gradient
    Dx_sign = tf.sign(Dx)

    # record the results for each iteration
    dict_step_to_image = {}
    dict_step_to_dissimilarity = {}
    dict_step_to_salmap = {}
    dict_step_to_prediction = {}
    dict_step_to_perturbation = {}

    for step in range(num_iterations):

        print('Step {}'.format(step))

        if step == 0:
            dict_step_to_image[0] = batch_img
            dict_step_to_dissimilarity[0] = 0
            dict_step_to_salmap[0] = sess.run(sal_maxlogit(vgg, viz_type, target_layer), feed_dict={vgg.images: batch_img})
            dict_step_to_prediction[0] = np.argmax(sess.run(vgg.probs, feed_dict={vgg.images: batch_img}))
            dict_step_to_perturbation[0] = np.zeros(batch_img.shape)
            continue

        Dx_sign_val, D_val = sess.run([Dx_sign, D], feed_dict={vgg_attack.images: dict_step_to_image[step - 1]})

        sal_map_val, probs_val = sess.run([sal_maxlogit(vgg, viz_type, target_layer), vgg.probs], feed_dict={vgg.images: dict_step_to_image[step - 1]})

        dict_step_to_image[step] = dict_step_to_image[step - 1] + step_size * Dx_sign_val
        dict_step_to_perturbation[step] = step_size * Dx_sign_val
        dict_step_to_salmap[step] = sal_map_val
        dict_step_to_dissimilarity[step] = D_val
        dict_step_to_prediction[step] = np.argmax(probs_val)

    evaluate(image_name,
             diff_type,
             viz_type,
             dict_step_to_image,
             dict_step_to_dissimilarity,
             dict_step_to_salmap,
             dict_step_to_prediction,
             dict_step_to_perturbation,
             num_iterations)

    sess.close()


if __name__ == '__main__':
    # setup the GPUs to use
    os.environ['CUDA_VISIBLE_DEVICES'] = '5, 6'
    main()
