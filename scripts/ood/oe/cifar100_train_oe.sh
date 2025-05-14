#!/bin/bash
# sh scripts/ood/oe/cifar100_train_oe.sh

# GPU=1
# CPU=1
# node=73
# jobname=openood

# PYTHONPATH='.':$PYTHONPATH \
# srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
# --cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
# --kill-on-bad-exit=1 --job-name=${jobname} \
# -w SG-IDC1-10-51-2-${node} \
# python main.py \
#     --config configs/datasets/cifar100/cifar100.yml \
#     configs/datasets/cifar100/cifar100_oe.yml \
#     configs/networks/resnet18_32x32.yml \
#     configs/pipelines/train/baseline.yml \
#     configs/pipelines/train/train_oe.yml \
#     configs/preprocessors/base_preprocessor.yml \
#     --seed 0



GPUID=$1

# for seed in 0 1 2; do
#     CUDA_VISIBLE_DEVICES=$GPUID python main.py \
#         --config configs/datasets/cifar100/cifar100.yml \
#         configs/datasets/cifar100/cifar100_oe.yml \
#         configs/datasets/cifar100/cifar100_ood.yml \
#         configs/networks/densenet.yml \
#         configs/pipelines/train/baseline.yml \
#         configs/pipelines/train/train_oe.yml \
#         --seed ${seed} #&
# done
# wait


# for seed in 0 1 2; do
#     CUDA_VISIBLE_DEVICES=$GPUID python main.py \
#         --config configs/datasets/cifar100/cifar100.yml \
#         configs/datasets/cifar100/cifar100_oe.yml \
#         configs/datasets/cifar100/cifar100_ood.yml \
#         configs/networks/wrn.yml \
#         configs/pipelines/train/baseline.yml \
#         configs/pipelines/train/train_oe.yml \
#         --network.pretrained True \
#         --network.checkpoint /data/xhn/current/OpenOOD/final_results/wrn_train/cifar100_wrn_base_e100_lr0.1_default/s${seed}/best.ckpt \
#         --optimizer.lr 0.0005 \
#         --optimizer.num_epochs 10 \
#         --dataset.train.batch_size 128 \
#         --dataset.oe.batch_size 128 \
#         --seed ${seed}
# done


GPUID=$1
for seed in 0 1 2; do
    CUDA_VISIBLE_DEVICES=$GPUID python main.py \
        --config configs/datasets/cifar100/cifar100.yml \
        configs/datasets/cifar100/cifar100_ood.yml \
        configs/datasets/cifar100/cifar100_oe.yml \
        configs/networks/resnet18_32x32.yml \
        configs/pipelines/train/baseline.yml \
        configs/pipelines/train/train_dal.yml \
        configs/preprocessors/base_preprocessor.yml \
        --network.pretrained True \
        --network.checkpoint /data/xhn/current/OpenOOD/checkpoints/cifar100_resnet18_32x32_base_e100_lr0.1_default/s${seed}/best.ckpt \
        --optimizer.lr 0.07 \
        --optimizer.num_epochs 10 \
        --seed ${seed} #&
done


# /data/xhn/current/OpenOOD/checkpoints/cifar10_resnet18_32x32_base_e100_lr0.1_default
# /data/xhn/current/OpenOOD/checkpoints/cifar100_resnet18_32x32_base_e100_lr0.1_default
# /data/xhn/current/OpenOOD/checkpoints/imagenet200_resnet18_224x224_base_e90_lr0.1_default