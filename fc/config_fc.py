import argparse


def str2bool(v):
    return v.lower() in ('true', '1')


parser = argparse.ArgumentParser()

# training params
parser.add_argument('-is_train', '--is_train', type=str2bool, default=True,
                    help='if train or test')
parser.add_argument('-is_diff', '--is_diff', type=str2bool, default=False,
                    help='decide if raw weight or weight difference')
parser.add_argument('-is_weights', '--is_weights', type=str2bool, default=True,
                    help='true to turn on the visual block')
parser.add_argument('-init_std', '--init_std', type=float, default=1e-1,
                    help='specify the init std for the training')
parser.add_argument('-bs', '--batch_size', type=int, default=128,
                    help='specify the batch size for the training')
parser.add_argument('-p_accu', '--p_accu', type=int, default=5,
                    help='specify for every how many steps print the accuracy')
parser.add_argument('-pa', '--patience', type=int, default=1250,  # 50 epochs
                    help='specify the patience which is used in early stop')
parser.add_argument('-epochs', '--epochs', type=int, default=50,
                    help='specify the total # of epochs for the training')
parser.add_argument('-lr', '--lr', type=float, default=1e-3,
                    help='specify the learning rate for the training')
parser.add_argument('-is_sal', '--is_saliency',  type=str2bool, default=True,
                        help='specify if turn on saliency map')

# model params
parser.add_argument('-nunits', '--num_neurons', type=int, default=100,
                    help='specify the # of hidden neurons for each layer')
parser.add_argument('-nlayers', '--num_layers', type=int, default=1,
                    help='specify the # of hidden layers')
parser.add_argument('-pb', '--pb', type=str2bool, default=True,
                    help='init bias slightly positive to 0.1, 0 if turned off')
parser.add_argument('-rand_label', '--rand_label', type=str2bool, default=False,
                    help='decide if use random labels')
parser.add_argument('-act', '--act_func', type=str, default="relu",
                    help='specify the type of activation function')
parser.add_argument('-opt', '--opt_type', type=str, default='Adam',
                    help='specify the type of the optimizer')

# data params
parser.add_argument('-dataset', '--dataset', type=str, default='2Rec_64_4000_20_1_black',
                    help='specify the dataset to use')
parser.add_argument('-indim', '--input_dim', type=int, default=64,
                    help='specify the dimension of input image')
parser.add_argument('-outdim', '--output_dim', type=int, default=2,
                    help='specify the dimension of class labels')
parser.add_argument('-is_total_perm', '--is_total_perm', type=str2bool, default=False,
                    help='specify if permuting images totally randomly')
parser.add_argument('-is_pixel_perm', '--is_pixel_perm', type=str2bool, default=False,
                    help='specify if permuting images pixel-wisely randomly')
parser.add_argument('-is_rand_sparse', '--is_rand_sparse', type=str2bool, default=False,
                    help='specify if sparsing images with a random pattern')
parser.add_argument('-is_single_sparse', '--is_single_sparse', type=str2bool, default=False,
                    help='specify if sparsing images with a single pattern for each class')
parser.add_argument('-is_multi_sparse', '--is_multi_sparse', type=str2bool, default=False,
                    help='specify if sparsing images with several different patterns for each class')
parser.add_argument('-sparse_ratio', '--sparse_ratio', type=float, default=0.0,
                    help='specify the sparse ratio')
parser.add_argument('-sparse_set_size', '--sparse_set_size', type=int, default=0,
                    help='specify the sparse set size')


# other params
parser.add_argument('-gpus', '--gpus', type=str, default='7',
                    help='specify which GPU to use')
parser.add_argument('-rs', '--rs', type=int, default=42,
                    help='specify the random seed for the training')
parser.add_argument('-nviz', '--num_to_viz', type=int, default=3,
                    help='specify the # of images to viz')
parser.add_argument('-is_viz_perm_inv', '--is_viz_perm_inv', type=str2bool, default=False,
                    help='specify if visualizing weights by multiplying inverse permutation matrix')
parser.add_argument('-spath', '--spath', type=str, default='summary',
                    help='specify the summary path')

def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed