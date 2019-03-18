import sys
import os
#ssys.path.append(os.path.abspath("/home/jordi/mlp_audio/MLPProjectAudio/MLP_CW2/mlp"))

import mlp.data_providers as data_providers
import numpy as np
from mlp.pytorch_experiment_scripts.arg_extractor import get_args
from mlp.pytorch_experiment_scripts.experiment_builder import ExperimentBuilder
from mlp.pytorch_experiment_scripts.model_architectures import ConvolutionalNetwork
import torch


args = get_args()  # get arguments from command line
rng = np.random.RandomState(seed=args.seed)  # set the seeds for the experiment
torch.manual_seed(seed=args.seed) # sets pytorch's seed

# convert number of filters per each layer to a list, and assert 
# it is specified for each layer

num_filters = [int(filt) for filt in args.num_filters[0].split(",")]

assert len(num_filters) == args.num_layers, "Not specified number of filter per each layer!"



train_data = data_providers.AudioDataProvider('train', batch_size=args.batch_size,
                                               rng=rng,shuffle_order=True,data_augmentation= args.data_augmentation,
							augmentation_number= args.augmentation_number,manual_verified_on=args.manual_verified_on,
                                                          pitch_augmentation=args.pitch_augmentation,augmentation_pitch=args.augmentation_pitch)  # initialize our rngs using the argument set seed
val_data = data_providers.AudioDataProvider('valid', batch_size=args.batch_size,
                                             rng=rng,shuffle_order=True)  # initialize our rngs using the argument set seed
test_data = data_providers.AudioDataProvider('test', batch_size=args.batch_size,
                                              rng=rng,shuffle_order=False)  # initialize our rngs using the argument set seed

assert train_data.dict_ == val_data.dict_ == test_data.dict_, "Different dictionaries!"

custom_conv_net = ConvolutionalNetwork(  # initialize our network object, in this case a ConvNet
    input_shape=(args.batch_size, args.image_num_channels, args.image_height, args.image_width),
    dim_reduction_type=args.dim_reduction_type,
        num_output_classes=train_data.num_classes, num_filters=num_filters,kernel_size = args.kernel_size,        num_layers=args.num_layers, dropout=args.dropout_rate,args=args,use_bias=False)

print("definicion convolutional network ok")

conv_experiment = ExperimentBuilder(network_model=custom_conv_net,
                                    experiment_name=args.experiment_name,
                                    num_epochs=args.num_epochs,
                                    weight_decay_coefficient=args.weight_decay_coefficient,
                                    use_gpu=args.use_gpu,
                                    continue_from_epoch=args.continue_from_epoch,
                                    train_data=train_data, val_data=val_data,
                                    test_data=test_data, batch_size = args.batch_size,
                                    training_instances = args.training_instances,
                                    test_instances = args.test_instances,
                                    val_instances = args.val_instances,
                                    image_height = args.image_height,
                                    image_width=args.image_width,
                                    eps_smooth = args.eps_smooth,
                                    num_classes = train_data.num_classes,
                                    loss_function=args.loss_function,
                                    use_cluster = args.use_cluster,
                                    gpu_id=args.gpu_id,
                                    q_ = args.q_parameter,
                                    consider_manual = args.consider_manual,
                                    args = args)  # build an experiment object

experiment_metrics, test_metrics = conv_experiment.run_experiment()  # run experiment and return experiment metrics
