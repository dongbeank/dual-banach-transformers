#!/bin/bash
# GPT-2 (124M, nanoGPT) trained from scratch on WikiText-103, one GPU.
# Recipe (defaults of train.py): AdamWScheduleFree, lr 1e-4, weight decay 0.05, 2000 warm-up steps, batch 8, block 1024,
# 20 epochs, no dropout. WikiText-103 and the GPT-2 tokenizer are downloaded by the datasets / transformers libraries.
OUT=../../experiments

# baseline
python train.py --reg 0   --seed 0 --output_dir $OUT/gpt2_wikitext103_reg0_seed0

# + L_DB with lambda = 0.5, 1, 5
python train.py --reg 0.5 --seed 0 --output_dir $OUT/gpt2_wikitext103_reg05_seed0
python train.py --reg 1   --seed 0 --output_dir $OUT/gpt2_wikitext103_reg1_seed0
python train.py --reg 5   --seed 0 --output_dir $OUT/gpt2_wikitext103_reg5_seed0

# amplification-factor ablations (kappa = 1 / stop-gradient through kappa)
python train.py --reg 5 --kappa_mode one    --seed 0 --output_dir $OUT/gpt2_wikitext103_reg5_kone_seed0
python train.py --reg 5 --kappa_mode detach --seed 0 --output_dir $OUT/gpt2_wikitext103_reg5_kdetach_seed0

# word dropout (p = 0.1) with and without L_DB
python train.py --reg 0 --word_dropout 0.1 --seed 0 --output_dir $OUT/gpt2_wikitext103_wd01_reg0_seed0
python train.py --reg 1 --word_dropout 0.1 --seed 0 --output_dir $OUT/gpt2_wikitext103_wd01_reg1_seed0

# --compile applies torch.compile to the model (same numbers, faster)
