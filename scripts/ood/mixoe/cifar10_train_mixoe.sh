#!/bin/bash
# sh scripts/ood/mixoe/cifar10_train_mixoe.sh
GPUID=$1

SEED=1
CUDA_VISIBLE_DEVICES=$GPUID python main.py \
    --config configs/datasets/cifar10/cifar10.yml \
    configs/datasets/cifar10/cifar10_oe.yml \
    configs/networks/resnet18_32x32.yml \
    configs/pipelines/train/baseline.yml \
    configs/pipelines/train/train_mixoe.yml \
    configs/preprocessors/base_preprocessor.yml \
    --network.pretrained True \
    --network.checkpoint /data/xhn/OpenOOD/results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s${SEED}/best.ckpt \
    --optimizer.lr 0.001 \
    --optimizer.num_epochs 10 \
    --dataset.train.batch_size 128 \
    --dataset.oe.batch_size 128 \
    --seed ${SEED}
