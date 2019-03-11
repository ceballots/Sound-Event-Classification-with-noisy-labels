#!/usr/bin/bash

layers_params_3='48,48,64'
layers_params_4='48,48,64,96'
layers_params_5='48,48,64,64,128'

for layers in 3 4 5;do
for dropout in 0.2 0.6 0.8;do
for kernel in 5;do
temp=layers_params_${layers}
#temp2=${!temp}
python mlp/pytorch_experiment_scripts/train_evaluate_emnist_classification_system.py --num_layers $layers \
--num_filters ${!temp} \
--kernel_size $kernel --batch_size 64 --use_cluster False  --num_epochs 100 --training_instances 17310 --val_instances 611 --test_instance 611 \
--image_height 96 --image_width 115 --use_gpu True --loss_function "CCE" \
--experiment_name exp_name_domingo_l${layers}_d${dropout}_k${kernel} --eps_smooth 0 --dropout_rate $dropout --dim_reduction_type max_pooling
done;
done;
done

