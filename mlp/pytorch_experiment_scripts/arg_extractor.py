import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args():
    """
    Returns a namedtuple with arguments extracted from the command line.
    :return: A namedtuple with arguments
    """
    parser = argparse.ArgumentParser(
        description='Weelcome to the MLP course\'s Pytorch training and inference helper script')

    parser.add_argument('--batch_size', nargs="?", type=int, default=64, help='Batch_size for experiment')
    parser.add_argument('--training_instances', nargs="?", type=int, default=17310, help='Number of training instances')
    parser.add_argument('--test_instances', nargs="?", type=int, default=611, help='Number of test instances')
    parser.add_argument('--val_instances', nargs="?", type=int, default=275, help='Number of validation instances')
    parser.add_argument('--continue_from_epoch', nargs="?", type=int, default=-1, help='Batch_size for experiment')
    parser.add_argument('--seed', nargs="?", type=int, default=7112018,
                        help='Seed to use for random number generator for experiment')
    parser.add_argument('--image_num_channels', nargs="?", type=int, default=1,
                        help='The channel dimensionality of our image-data')
    parser.add_argument('--image_height', nargs="?", type=int, default=97, help='Height of image data')
    parser.add_argument('--image_width', nargs="?", type=int, default=96, help='Width of image data')
    parser.add_argument('--dim_reduction_type', nargs="?", type=str, default='strided_convolution',
                        help='One of [strided_convolution, dilated_convolution, max_pooling, avg_pooling]')
    parser.add_argument('--num_layers', nargs="?", type=int, default=3,
                        help='Number of convolutional layers in the network (excluding '
                             'dimensionality reduction layers)')
    #parser.add_argument('--num_filters', nargs="?", type=int, default=64,
    #                    help='Number of convolutional filters per convolutional layer in the network (excluding '
    #                         'dimensionality reduction layers)')
    parser.add_argument('--num_filters', nargs='+',help='Number of convolutional filters per convolutional layer in the network                         (excluding dimensionality reduction layers)')    
    parser.add_argument('--kernel_size', nargs="?", type=int, default=3,
                        help='Number of convolutional layers in the network (excluding '
                             'dimensionality reduction layers)')
    parser.add_argument('--num_epochs', nargs="?", type=int, default=10, help='The experiment\'s epoch budget')
    parser.add_argument('--experiment_name', nargs="?", type=str, default="exp_audio",
                        help='Experiment name - to be used for building the experiment folder')
    parser.add_argument('--use_gpu', nargs="?", type=str2bool, default=False,
                        help='A flag indicating whether we will use GPU acceleration or not')
    parser.add_argument('--gpu_id', type=str, default="None", help="A string indicating the gpu to use")
    parser.add_argument('--use_cluster', nargs="?", type=str2bool, default=False,
                        help='A flag indicating whether we will use the cluster or not')
    parser.add_argument('--consider_manual',nargs="?",type=str2bool,default=False,
                        help = 'cosider manual')
    parser.add_argument('--weight_decay_coefficient', nargs="?", type=float, default=1e-05,
                        help='Weight decay to use for Adam')
    parser.add_argument('--dropout_rate', nargs="?", type=float, default=0.,
                        help='Dropout rate')
    parser.add_argument('--loss_function', nargs="?", type=str, default='CCE',
                        help='One of [CCE,lq_loss , , ]')  
    parser.add_argument('--q_parameter', nargs="?", type=float, default=0.8,
                        help='q parameter for loss_q, by default 0.8')
    parser.add_argument('--eps_smooth', nargs="?", type=float, default=0.1,
                        help='Label smoothing parameter')
    parser.add_argument('--data_augmentation', nargs="?", type=str2bool, default=False,
                        help='if you want to add data augmentation')
    parser.add_argument('--augmentation_number', nargs="?", type=int, default=0,
                        help='how much data augmentation, from 0 to 4')    
    parser.add_argument('--manual_verified_on', nargs="?", type=str2bool, default=False,
                        help='will use just manual verified data to train')
    parser.add_argument('--mixup', nargs="?", type=str2bool, default=False,
                        help='Mixup')
    parser.add_argument('--alpha', nargs="?", type=float, default=0.3,
                        help='alpha')

    parser.add_argument('--pitch_augmentation', nargs="?", type=str2bool, default=False,
                        help='if you want to add pitch data augmentation')
    parser.add_argument('--augmentation_pitch', nargs="?", type=int, default=0,
                        help='how much data augmentation pitch, from 0 to 2')  

    parser.add_argument('--stack', nargs="?", type=str2bool, default=False,
                        help='both label smooth and mixup')
    parser.add_argument('--shuffle', nargs="?", type=str2bool, default=False,
                        help='shuffle before epoch')

    args = parser.parse_args()
    
    gpu_id = str(args.gpu_id)

    if gpu_id != "None":
        args.gpu_id = gpu_id
        
    print(args)
    return args
