#!/bin/bash
# sh scripts/ood/oe/imagenet200_test_oe.sh

############################################
# alternatively, we recommend using the
# new unified, easy-to-use evaluator with
# the example script scripts/eval_ood.py
# especially if you want to get results from
# multiple runs

# ood
# python scripts/eval_ood.py \
#    --id-data imagenet200 \
#    --root ./results/imagenet200_oe_resnet18_224x224_oe_e90_lr0.1_lam0.5_default \
#    --postprocessor msp \
#    --save-score --save-csv #--fsood

# # full-spectrum ood
# python scripts/eval_ood.py \
#    --id-data imagenet200 \
#    --root ./results/imagenet200_oe_resnet18_224x224_oe_e90_lr0.1_lam0.5_default \
#    --postprocessor msp \
#    --save-score --save-csv --fsood


CUDA_VISIBLE_DEVICES=0 python scripts/eval_ood.py \
   --id-data imagenet200 \
   --root '/data/wfq/OpenOOD/results/Alpha_T_learnable_imagenet200_oe_resnet18_224x224_oe_e90_lr0.1_lam0.5_default' \
   --postprocessor msp \
   --save-score --save-csv