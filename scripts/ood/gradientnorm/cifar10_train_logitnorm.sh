#!/bin/bash
# sh scripts/ood/logitnorm/cifar10_train_logitnorm.sh


GPUID=$1
CUDA_VISIBLE_DEVICES=$GPUID python main.py \
    --config configs/datasets/cifar10/cifar10.yml \
    configs/networks/resnet18_32x32.yml \
    configs/pipelines/train/train_gradientnorm.yml \
    configs/preprocessors/base_preprocessor.yml \
    --seed 0
