from model_helper import *


class FC_model(object):
    def __init__(self, config):
        self.num_layers = config.num_layers
        self.num_neurons = config.num_neurons
        self.input_dim = config.input_dim
        self.output_dim = config.output_dim

        self.init_std = config.init_std
        self.lr = config.lr
        self.pb = config.pb
        self.opt_type = config.opt_type

        self.is_weights = config.is_weights
        self.is_diff = config.is_diff
        self.batch_size = config.batch_size
        self.epochs = config.epochs

        self.patience = config.patience
        self.p_accu = config.p_accu
        self.rand_label = config.rand_label
        self.dataset = config.dataset

        self.is_saliency = config.is_saliency
        self.num_to_viz = config.num_to_viz
        self.is_total_perm = config.is_total_perm
        self.is_pixel_perm = config.is_pixel_perm

        self.is_rand_sparse = config.is_rand_sparse
        self.is_single_sparse = config.is_single_sparse
        self.is_multi_sparse = config.is_multi_sparse
        self.sparse_ratio = config.sparse_ratio
        self.sparse_set_size = config.sparse_set_size

        self.is_viz_perm_inv = config.is_viz_perm_inv

        if config.act_func == 'relu':
            activation = tf.nn.relu
        elif config.act_func == 'tanh':
            activation = tf.nn.tanh
        else:
            activation = tf.nn.relu
            print("Unknown type of act-func! Use relu as default")

        # Placeholders for the images and labels
        with tf.name_scope("Inputs"):
            self.images = tf.placeholder(tf.float32, shape=(None, self.input_dim, self.input_dim, 3), name='X')
            self.labels = tf.placeholder(tf.int32, shape=(None, self.output_dim), name='y')

        mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
        self.imgs = self.images - mean
        self.imgs_fl = tf.reshape(self.imgs, [-1, self.input_dim * self.input_dim * 3])

        # Initialize weights and bias
        self.w_vars, self.w_vars_init, self.b_vars, self.b_vars_zero = \
            init_weights_bias(self.input_dim, self.output_dim, self.num_neurons, self.num_layers,
                              self.init_std, self.pb)

        # Forward propagation
        self.logits, self.h_vars, h_before_vars = forwardprop(self.imgs_fl, self.w_vars, self.b_vars, activation)

        # Loss
        self.loss = entropy_loss(self.logits, self.labels)

        # Training operation
        self.train_op = train_opt(self.loss, self.lr, self.opt_type)

        # Accuracy
        self.accu = eval_accuracy(self.logits, self.labels)
