import numpy as np

def similarity(layer, image1, image2, sess, pl_holder_X, firing_arg):

    if firing_arg == 'plain':
        layer_val_1 = sess.run(layer, feed_dict={pl_holder_X: image1})
        layer_val_2 = sess.run(layer, feed_dict={pl_holder_X: image2})
        return np.linalg.norm(layer_val_1 - layer_val_2)

    if firing_arg == 'ahat':
        layer_val_1 = np.sign(sess.run(layer, feed_dict={pl_holder_X: image1}))
        layer_val_2 = np.sign(sess.run(layer, feed_dict={pl_holder_X: image2}))

        layer_bool_1 = layer_val_1.astype(np.bool)
        layer_bool_2 = layer_val_2.astype(np.bool)

        intersection = np.logical_and(layer_bool_1, layer_bool_2)
        return 1 - intersection.sum() / float(layer_val_1.sum()) # the changing rate
