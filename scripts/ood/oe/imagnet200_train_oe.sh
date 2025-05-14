#!/bin/bash
# sh scripts/ood/oe/imagenet200_train_oe.sh

# python main.py \
#     --config configs/datasets/imagenet200/imagenet200.yml \
#     configs/datasets/imagenet200/imagenet200_oe.yml \
#     configs/networks/resnet18_224x224.yml \
#     configs/pipelines/train/baseline.yml \
#     configs/pipelines/train/train_oe.yml \
#     configs/preprocessors/base_preprocessor.yml \
#     --optimizer.num_epochs 90 \
#     --dataset.train.batch_size 128 \
#     --num_gpus 2 --num_workers 16 \
#     --merge_option merge \
#     --seed 0


GPUID=$1

# for seed in 1 2; do
#     CUDA_VISIBLE_DEVICES=$GPUID python main.py \
#         --config configs/datasets/imagenet200/imagenet200.yml \
#         configs/datasets/imagenet200/imagenet200_oe.yml \
#         configs/networks/resnet18_224x224.yml \
#         configs/pipelines/train/baseline.yml \
#         configs/pipelines/train/train_oe.yml \
#         configs/preprocessors/base_preprocessor.yml \
#         --optimizer.num_epochs 90 \
#         --dataset.train.batch_size 256 \
#         --num_gpus 1 --num_workers 16 \
#         --merge_option merge \
#         --seed $seed
# done


GPUID=$1
for seed in 0 1 2; do
    CUDA_VISIBLE_DEVICES=$GPUID python main.py \
        --config configs/datasets/imagenet200/imagenet200.yml \
        configs/datasets/imagenet200/imagenet200_oe.yml \
        configs/networks/resnet18_224x224.yml \
        configs/pipelines/train/baseline.yml \
        configs/pipelines/train/train_dal.yml \
        configs/preprocessors/base_preprocessor.yml \
        --network.pretrained True \
        --network.checkpoint /data/xhn/current/OpenOOD/checkpoints/imagenet200_resnet18_224x224_base_e90_lr0.1_default/s${seed}/best.ckpt \
        --optimizer.lr 0.01 \
        --optimizer.num_epochs 10 \
        --seed ${seed} #&
done
