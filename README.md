# Dual Banach regularization for Transformers

Official code for *A Dual Banach Space Perspective on Transformers*. It contains the dual Banach regularizer **L_DB** and the
two training pipelines used in the paper:

| folder | model | task | base code |
|---|---|---|---|
| [`vit/`](vit/) | ViT-S/16, ViT-T/16 | CIFAR-10 / CIFAR-100 classification, CIFAR-10-C / CIFAR-100-C robustness | RSPC (Guo, Stutz, Schiele, CVPR 2023) → DeiT / timm |
| [`gpt2/`](gpt2/) | GPT-2 (124M, nanoGPT) | WikiText-103 language modeling | nanoGPT, HuggingFace `run_clm_no_trainer.py` |

## The regularizer

For a Transformer with L layers, let x^l be the input of layer l and f̃*,l_τ(x^l) its attention output (the input of the FFN
branch, FFN(z) = W2 σ(W1 z), applied after LayerNorm with scale γ^l). The per-layer amplification factor and the dual Banach
regularizer are (equations (35) and (36) of the paper)

```
κ_l  = 1 + ‖W2^l‖ ‖W1^l‖ · max|γ^l| / std( f̃*,l_τ(x^l) )
L_DB = (1/L) · Σ_{l=1..L}  κ_l² · ‖ f̃*,l_τ(x^l) ‖²_ℓ2
```

The term κ_l² ‖f̃*,l_τ(x^l)‖² is computed per token and averaged over tokens, batch and layers. Training minimizes
`task loss + λ · L_DB`; λ is `--reg` in both scripts.

| quantity | how the code evaluates it | ViT (`vit/vision_transformer_reg.py`) | GPT-2 (`gpt2/model.py`) |
|---|---|---|---|
| ‖f̃*,l_τ(x^l)‖²_ℓ2 | per token: squared norm of the attention output after the output projection, averaged over heads, plus ‖x^l_t‖²/H | `attn_norm_core`, `block_reg_core` | `CausalSelfAttention.attention_norm`, `Block.forward` |
| std(f̃*,l_τ(x^l)) | standard deviation of the FFN-branch input over the feature dimension of each token (gradient flows through it) | `block_reg_core` | `Block.kappa_sq` |
| ‖W1^l‖, ‖W2^l‖ | spectral norms of the two FFN weights: one power-iteration step per training step, singular-vector estimates kept as buffers (gradient flows through them) | `_power_iteration` (batched over layers) | `Block.spectral_norm` |
| max\|γ^l\| | LayerNorm scale of the FFN branch, detached | `block_reg_core` | `Block.kappa_sq` |
| κ_l | `kappa_mode` = `full` (as above) / `one` (κ_l ≡ 1) / `detach` (stop-gradient through κ_l) | `block_reg_core` | `Block.forward` |
| mean over layers | returned as the second (ViT) / third (GPT-2) model output | `_forward_core` | `GPT.forward` |

## Setup

```
pip install -r requirements.txt
```

| package | version |
|---|---|
| Python | 3.11 |
| torch / torchvision | 2.11.0 / 0.26.0 (CUDA 12.8) |
| timm, kornia | 1.0.26, 0.8.3 (`vit/`) |
| schedulefree | 1.4.1 |
| datasets, transformers | 4.8.4, 5.4.0 (`gpt2/`, tokenizer and data loading only) |

## Vision Transformers (`vit/`)

| file | content |
|---|---|
| `vision_transformer_reg.py` | `VisionTransformer_RegOpt(model_name, num_classes, kappa_mode, lsigma, compile_mode, ctrl_mode)`: wraps a timm ViT and returns `(logits, L_DB)`; also the two simple regularization controls (`ctrl_mode`) |
| `main.py` | training / evaluation driver (RSPC) with the options below |
| `engine.py` | one epoch (patch-level augmented batch + clean batch, as in RSPC), evaluation, CIFAR-C evaluation |
| `datasets.py` | CIFAR-10/100 + DeepAugment loading, optional fixed stratified validation split |
| `utils.py`, `losses.py`, `samplers.py` | RSPC / DeiT utilities |
| `run_examples.sh` | the commands for baseline, + L_DB, κ ≡ 1 and the two controls |

### Data

| data | layout (`--data-path`, `--deepaugment_base_path`, `--cifarc_base_path`) | source |
|---|---|---|
| CIFAR-10 / CIFAR-100 | torchvision format, downloaded automatically | torchvision |
| DeepAugment (EDSR, CAE) | `<base>/EDSR/<class>/...`, `<base>/CAE/<class>/...` (`ImageFolder`) | RSPC scripts `DeepAugment/*_distort_cifar*.py` or the links in the RSPC README |
| CIFAR-10-C / CIFAR-100-C | `<base>/<corruption>.npy`, `<base>/labels.npy` | Hendrycks & Dietterich |

Each training image is used three times per epoch (clean, EDSR, CAE).

### Training

```
cd vit
python main.py --batch-size 256 --model vit_small_patch16_224 --data-set CIFAR10 --data-path ../../data \
  --deepaugment --deepaugment_base_path ../../data/CIFAR-10-DeepAugment --cifarc_base_path ../../data/CIFAR-10-C \
  --reg 0.01 --seed 0 --output_dir ../../experiments/vitS_cifar10_ldb_seed0
```

Recipe (defaults of `main.py`, unchanged from RSPC):

| hyperparameter | value |
|---|---|
| optimizer | AdamW, weight decay 0.05 |
| learning rate | 2.5e-4 at batch 256 (`--lr 5e-4` × batch / 512), cosine to 1e-5 |
| epochs / batch | 200 / 256 |
| regularization | drop path 0.1, label smoothing 0.1, MixUp 0.8, CutMix 1.0, RandAugment, random erasing 0.25 |
| augmentation | patch-level augmentation (kornia) on one of the two passes per step; DeepAugment training images |
| precision | AMP |
| λ (`--reg`) | 0.01 |

| option | values | meaning |
|---|---|---|
| `--model` | `vit_small_patch16_224`, `vit_tiny_patch16_224` | ViT-S/16, ViT-T/16 |
| `--data-set` | `CIFAR10`, `CIFAR100` | dataset (use the matching DeepAugment and CIFAR-C folders) |
| `--reg` | float | λ, weight of L_DB (0 = baseline) |
| `--kappa_mode` | `full`, `one`, `detach` | amplification factor κ: as defined / κ ≡ 1 (attention-output norm only) / stop-gradient |
| `--lsigma` | float | scales the product ‖W2‖‖W1‖·max\|γ\|/std in κ_l (1.0 in the reported runs) |
| `--ctrl_mode`, `--ctrl_lambda` | `specnorm` or `blocknorm`, float | simple regularization controls (spectral-norm penalty / block-output-norm penalty), used with `--reg 0` |
| `--val_split` | float | holds out a fixed stratified fraction of the training set; checkpoint selection then uses validation accuracy and the test set is evaluated once at the end |
| `--compile` | `model` (default), `model-cg`, `reg`, `none` | `torch.compile` of the forward pass / additionally CUDA graphs / only the regularizer terms / off |
| `--seed`, `--output_dir` | | |

Multi-GPU: `torchrun --nproc_per_node=N main.py ...` (DistributedDataParallel; the paper runs used one GPU).

Outputs in `--output_dir`: `log.txt` (one JSON line per epoch: train loss, L_DB, test accuracy), `last.pth.tar`, `model_best.pth.tar`, and at the end the best
test accuracy over epochs, the final-epoch accuracy and the mean CIFAR-C accuracy of the best checkpoint.

## GPT-2 (`gpt2/`)

| file | content |
|---|---|
| `model.py` | nanoGPT with L_DB: `GPT(config, reg, kappa_mode)` returns `(logits, loss, L_DB)` |
| `train.py` | from-scratch training on a HuggingFace text dataset (default WikiText-103), perplexity after every epoch |
| `run_examples.sh` | the commands for baseline, λ ∈ {0.5, 1, 5}, the κ ablations and word dropout |

### Training

```
cd gpt2
python train.py --reg 1.0 --seed 0 --output_dir ../../experiments/gpt2_wikitext103_reg1_seed0
```

WikiText-103 (`Salesforce/wikitext`, `wikitext-103-raw-v1`) and the GPT-2 tokenizer are downloaded by the `datasets` / `transformers`
libraries on first use; the text is tokenized and concatenated into blocks of 1024 tokens.

Recipe (defaults of `train.py`):

| hyperparameter | value |
|---|---|
| model | 12 layers, 12 heads, width 768, context 1024, vocabulary 50304 (124M parameters), weights initialized from scratch |
| optimizer | AdamWScheduleFree, lr 1e-4, weight decay 0.05, 2000 warm-up steps |
| epochs / batch | 20 / 8 |
| dropout | 0 |
| precision | fp32 |
| λ (`--reg`) | 0.5, 1, 5 |

| option | values | meaning |
|---|---|---|
| `--reg` | float | λ, weight of L_DB (0 = baseline) |
| `--kappa_mode` | `full`, `one`, `detach` | as for ViT |
| `--word_dropout` | float | probability of replacing an input token by `<|endoftext|>` during training |
| `--weight_decay`, `--learning_rate`, `--num_train_epochs`, `--per_device_train_batch_size` | | optimizer settings |
| `--dataset_name`, `--dataset_config_name`, `--block_size` | | any HuggingFace causal-LM dataset |
| `--compile` | flag | `torch.compile` the model |
| `--max_train_steps`, `--checkpointing_steps`, `--resume_from_checkpoint` | | shorter runs, checkpoints |
| `--seed`, `--output_dir` | | |

Outputs in `--output_dir`: `training_log.txt` (one line per epoch: train loss, total loss, validation loss, perplexity),
`pytorch_model.bin` (last epoch) and `all_results.json` (best and final perplexity).

## Using L_DB in your own model

Per layer you need the per-head attention outputs and the output-projection weights (for ‖f̃*,l_τ(x^l)‖²), the input of the FFN
branch (for its standard deviation), the two FFN weight matrices (spectral norms) and the LayerNorm scale of the FFN branch.
`vit/vision_transformer_reg.py` (`attn_norm_core`, `_power_iteration`, `block_reg_core`) and `gpt2/model.py`
(`CausalSelfAttention.attention_norm`, `Block.spectral_norm`, `Block.kappa_sq`) are two self-contained examples, for bidirectional
and causal attention respectively.

## License / acknowledgement

Apache 2.0 (see `LICENSE`). `vit/` is derived from RSPC (https://github.com/guoyongcs/RSPC) and DeiT
(https://github.com/facebookresearch/deit), both Apache 2.0, and uses timm (Apache 2.0). `gpt2/` is derived from nanoGPT
(https://github.com/karpathy/nanoGPT, MIT) and the HuggingFace Transformers examples (Apache 2.0).
