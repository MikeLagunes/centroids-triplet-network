#!/bin/sh

# TLESS

# s-triplet

# all

cd /home/miguel_dynium/centroids-triplet-network

python3 train/train_ctn.py --alpha_factor=1e-4 --beta_factor=1e-4 --id='1e-4_1e-4' --n_epoch=33
python3 train/train_ctn.py --alpha_factor=1e-3 --beta_factor=1e-3 --id='1e-3_1e-3' --n_epoch=33
python3 train/train_ctn.py --alpha_factor=1e-2 --beta_factor=1e-3 --id='1e-2_1e-2' --n_epoch=33
python3 train/train_ctn.py --alpha_factor=1e-1 --beta_factor=1e-3 --id='1e-1_1e-3' --n_epoch=33
python3 train/train_ctn.py --alpha_factor=1e-3 --beta_factor=1e-2 --id='1e-3_1e-2' --n_epoch=33
python3 train/train_ctn.py --alpha_factor=1e-3 --beta_factor=1e-1 --id='1e-3_1e-1' --n_epoch=33



#Core50

# python test/test_triplet_resnet_softmax.py --model_path=/media/alexa/DATA/Miguel/checkpoints/ICRA/tris/all/core50/1/triplet_softmax_sgd_all_l1_core50_temp.pkl --test_path=/media/alexa/DATA/Miguel/checkpoints/ICRA/tris/all/core50/1/ --split=train --dataset=core50
# python test/test_triplet_resnet_softmax.py --model_path=/media/alexa/DATA/Miguel/checkpoints/ICRA/tris/all/core50/1/triplet_softmax_sgd_all_l1_core50_temp.pkl --test_path=/media/alexa/DATA/Miguel/checkpoints/ICRA/tris/all/core50/1/ --split=test --dataset=core50

# # TOybox

# python test/test_triplet_resnet_softmax.py --model_path=/media/alexa/DATA/Miguel/checkpoints/ICRA/tris/all/toybox/1/triplet_softmax_sgd_all_l1_toybox_temp.pkl --test_path=/media/alexa/DATA/Miguel/checkpoints/ICRA/tris/all/toybox/1/ --split=train --dataset=toybox
# python test/test_triplet_resnet_softmax.py --model_path=/media/alexa/DATA/Miguel/checkpoints/ICRA/tris/all/toybox/1/triplet_softmax_sgd_all_l1_toybox_temp.pkl --test_path=/media/alexa/DATA/Miguel/checkpoints/ICRA/tris/all/toybox/1/ --split=test --dataset=toybox


# CORE50

# s-triplet

#  novel 
`
# - all


