#!/usr/bin/bash

layers_params_3='24,48,48'
layers_params_4='24,48,48,12'
layers_params_5='24,48,48,48,12'

for layers in 3 4 5;do
for dropout in 0.3 0.6 0.6 0.7 0.8;do
for kernel in 3 5;do
python mlp/pytorch_experiment_scripts/train_evaluate_emnist_classification_system.py --num_layers $layers \
--num_filters $layers_params_$layers \
--kernel_size $kernel --batch_size 64 --use_cluster False  --num_epochs 70 --training_instances 17310 --val_instances 611 --test_instance 611 \
--image_height 96 --image_width 115 --use_gpu True --loss_function "CCE" \
--experiment_name exp_name_l${layers}_d${dropout} --eps_smooth 0 --dropout_rate $dropout
done;
done;
done

