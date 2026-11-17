#!/bin/bash
# Training recipes used in the paper (one GPU; ViT-S/16 and ViT-T/16 on CIFAR-10/100 with DeepAugment, RSPC recipe).
# Data layout expected under ../../data: cifar-10-batches-py/ (or cifar-100-python/), CIFAR-10-DeepAugment/{EDSR,CAE}/<class>/<idx>.png,
# CIFAR-10-C/{corruption}.npy + labels.npy  (same for CIFAR-100).  See README.md.
DATA=../../data
COMMON="--batch-size 256 --data-path $DATA --num_workers 8 --deepaugment"

# baseline (no regularizer)
python main.py $COMMON --model vit_small_patch16_224 --data-set CIFAR10 --deepaugment_base_path $DATA/CIFAR-10-DeepAugment \
  --cifarc_base_path $DATA/CIFAR-10-C --reg 0 --seed 0 --output_dir ../../experiments/vitS_cifar10_baseline_seed0

# + L_DB (full regularizer, lambda = 0.01)
python main.py $COMMON --model vit_small_patch16_224 --data-set CIFAR10 --deepaugment_base_path $DATA/CIFAR-10-DeepAugment \
  --cifarc_base_path $DATA/CIFAR-10-C --reg 0.01 --seed 0 --output_dir ../../experiments/vitS_cifar10_ldb_seed0

# ablation (B): attention-output norm only (kappa = 1)
python main.py $COMMON --model vit_small_patch16_224 --data-set CIFAR10 --deepaugment_base_path $DATA/CIFAR-10-DeepAugment \
  --cifarc_base_path $DATA/CIFAR-10-C --reg 0.01 --kappa_mode one --seed 0 --output_dir ../../experiments/vitS_cifar10_kone_seed0

# simple regularization controls (same lambda, no L_DB): spectral-norm penalty / block-output-norm penalty
python main.py $COMMON --model vit_small_patch16_224 --data-set CIFAR10 --deepaugment_base_path $DATA/CIFAR-10-DeepAugment \
  --cifarc_base_path $DATA/CIFAR-10-C --reg 0 --ctrl_mode specnorm --ctrl_lambda 0.01 --seed 0 --output_dir ../../experiments/vitS_cifar10_specnorm_seed0
python main.py $COMMON --model vit_small_patch16_224 --data-set CIFAR10 --deepaugment_base_path $DATA/CIFAR-10-DeepAugment \
  --cifarc_base_path $DATA/CIFAR-10-C --reg 0 --ctrl_mode blocknorm --ctrl_lambda 0.01 --seed 0 --output_dir ../../experiments/vitS_cifar10_blocknorm_seed0

# to run without torch.compile (slower): add --compile none

# CIFAR-100: --data-set CIFAR100 --deepaugment_base_path $DATA/CIFAR-100-DeepAugment --cifarc_base_path $DATA/CIFAR-100-C ;  ViT-T: --model vit_tiny_patch16_224
