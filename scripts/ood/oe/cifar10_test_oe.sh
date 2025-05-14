#!/bin/bash
# sh scripts/ood/oe/cifar10_test_oe.sh

# GPU=1
# CPU=1
# node=63
# jobname=openood

# PYTHONPATH='.':$PYTHONPATH \
# srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
# --cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
# --kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \
GPUID=$1 
CUDA_VISIBLE_DEVICES=$GPUID python main.py \
    --config configs/datasets/cifar10/cifar10.yml \
    configs/datasets/cifar10/cifar10_ood.yml \
    configs/networks/resnet18_32x32.yml \
    configs/pipelines/test/test_ood.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/msp.yml \
    --num_workers 8 \
    --network.checkpoint '/data/wfq/OpenOOD/results/cifar10_oe_resnet18_32x32_oe_e100_lr0.1learnable_lam0.5_default/s0/best.ckpt' \
    --mark 0

############################################
# alternatively, we recommend using the
# new unified, easy-to-use evaluator with
# the example script scripts/eval_ood.py
# especially if you want to get results from
# multiple runs
# python scripts/eval_ood.py \
#    --id-data cifar10 \
#    --root /data/wfq/OpenOOD/results/OE/baseline \
#    --postprocessor msp \
#    --save-score --save-csv
