#!/bin/bash
# sh scripts/ood/oe/cifar10_train_oe.sh

# GPU=0
# CPU=1
# node=73
# jobname=openood

# PYTHONPATH='.':$PYTHONPATH \
# srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
# --cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
# --kill-on-bad-exit=1 --job-name=${jobname} \
# -w SG-IDC1-10-51-2-${node} \
# python main.py \
#     --config configs/datasets/cifar10/cifar10.yml \
#     configs/datasets/cifar10/cifar10_oe.yml \
#     configs/networks/resnet18_32x32.yml \
#     configs/pipelines/train/baseline.yml \
#     configs/pipelines/train/train_oe.yml \
#     configs/preprocessors/base_preprocessor.yml \
#     --seed 0

# GPUID=$1

# for seed in 0; do
#     CUDA_VISIBLE_DEVICES=$GPUID python main.py \
#         --config configs/datasets/cifar10/cifar10.yml \
#         configs/datasets/cifar10/cifar10_oe.yml \
#         configs/datasets/cifar10/cifar10_ood.yml \
#         configs/networks/resnet18_32x32.yml \
#         configs/pipelines/train/baseline.yml \
#         configs/pipelines/train/train_oe.yml \
#         --seed ${seed} #&
# done
# # wait



GPUID=$1
for seed in 0 1 2; do
    CUDA_VISIBLE_DEVICES=$GPUID python main.py \
        --config configs/datasets/cifar10/cifar10.yml \
        configs/datasets/cifar10/cifar10_ood.yml \
        configs/datasets/cifar10/cifar10_oe.yml \
        configs/networks/resnet18_32x32.yml \
        configs/pipelines/train/baseline.yml \
        configs/pipelines/train/train_dal.yml \
        configs/preprocessors/base_preprocessor.yml \
        --network.pretrained True \
        --network.checkpoint /data/xhn/current/OpenOOD/checkpoints/cifar10_resnet18_32x32_base_e100_lr0.1_default/s${seed}/best.ckpt \
        --optimizer.lr 0.07 \
        --optimizer.num_epochs 10 \
        --seed ${seed} #&
done
