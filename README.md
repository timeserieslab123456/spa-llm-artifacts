# SPA-LLM

Official code for **SPA-LLM** — Phase-Aware Semantic Alignment for LLM-Based Long-Term Time Series Forecasting.

## Overview

SPA-LLM aligns pretrained LLMs with the phase-wise dependency structure of seasonal time series components through two key designs:

- **Semantic-Enhanced Structured Tokenization**: Seasonal components are organized into period-structured patches and compressed from $P$ phases to $P'$ compact tokens via adaptive pooling ($P' = 24$ in all experiments). This preserves phase structure while reducing token length and computational cost. A semantic encoder (MLP) maps the compressed tokens to the LLM embedding dimension.

- **Natural Language Semantic Mixing**: The full GPT2 vocabulary ($V \approx 50,000$) is distilled into $V' = 512$ distilled anchors through low-rank vocabulary distillation (rank $m = 32$). $P'$ learnable phase queries attend over the distilled anchors via cross-attention to produce a phase-aware semantic memory, which is injected into the seasonal tokens before and after the frozen LLM. This injects token-level semantic priors from the LLM semantic space, making phase-aware tokens more compatible with pretrained language representations.

The input series is first decomposed into seasonal and trend components via a moving average operator. The seasonal component goes through Tokenization → Semantic Mixing → frozen GPT2-small (first 4 layers, only LayerNorm trainable) → output projection, while the trend is predicted by a lightweight linear layer. The final forecast is the sum of both branches.

All experiments use an input length of $L = 512$ and prediction horizons $H \in \{96, 192, 336, 720\}$. The model has only **1.73M** trainable parameters. Training uses Adam optimizer with learning rate $1 \times 10^{-4}$ and MSE loss. Random seed is fixed to 22.

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

Main dependencies: PyTorch, transformers, einops, numpy, pandas, scikit-learn, tqdm.

The GPT2-small backbone is downloaded from HuggingFace Hub automatically on first run.
The dataset CSV files, pretrained GPT-2 weights, and trained checkpoints are not included in this repository.

## Dataset

Place datasets under `./datasets/` following this structure:

```
datasets/
├── ETT-small/
│   ├── ETTh1.csv
│   ├── ETTh2.csv
│   ├── ETTm1.csv
│   └── ETTm2.csv
├── electricity/
│   └── electricity.csv
└── Solar/
    └── solar_AL.txt
```

ETT datasets can be downloaded from the [ETDataset repository](https://github.com/zhouhaoyi/ETDataset). ECL and Solar datasets are available from the [Autoformer repository](https://github.com/thuml/Autoformer).

Use the following dataset arguments with the training command below. The forecast horizon can be set with `--pred_len` to 96, 192, 336, or 720.

| Dataset | `--data` | `--root_path` | `--data_path` | `--enc_in` | `--period_len` | `--decomp_kernel` |
|---------|----------|---------------|---------------|------------|----------------|-------------------|
| ETTh1 | `ETTh1` | `./datasets/ETT-small/` | `ETTh1.csv` | 7 | 24 | 25 |
| ETTh2 | `ETTh2` | `./datasets/ETT-small/` | `ETTh2.csv` | 7 | 24 | 25 |
| ETTm1 | `ETTm1` | `./datasets/ETT-small/` | `ETTm1.csv` | 7 | 96 | 97 |
| ETTm2 | `ETTm2` | `./datasets/ETT-small/` | `ETTm2.csv` | 7 | 96 | 97 |
| ECL | `custom_electricity` | `./datasets/electricity/` | `electricity.csv` | 321 | 24 | 25 |
| Solar | `custom_solar` | `./datasets/Solar/` | `solar_AL.txt` | 1 | 144 | 145 |

The Solar loader trains on one of the 137 series at a time. Use `--features M` for multivariate ECL data.

## Training

Example training command for ETTh1 with prediction horizon 96:

```bash
python main.py \
  --data ETTh1 \
  --data_path ETTh1.csv \
  --root_path ./datasets/ETT-small/ \
  --pred_len 96 \
  --period_len 24 \
  --decomp_kernel 25 \
  --pp 24 \
  --pool_size 512 \
  --rank 32 \
  --a1 0.1 \
  --a2 0.1 \
  --batch_size 256 \
  --train_epochs 100 \
  --learning_rate 0.0001 \
  --gpu 0
```

The script uses the selected CUDA GPU when available and otherwise runs on the CPU. Training writes checkpoints and logs under `./checkpoints/`.

## Model Configuration

| Dataset | Prior Period $P$ | Kernel | $P'$ | $V'$ | Rank $m$ |
|---------|-----------------|--------|------|------|----------|
| ETTh1   | 24              | 25     | 24   | 512  | 32       |
| ETTh2   | 24              | 25     | 24   | 512  | 32       |
| ETTm1   | 96              | 97     | 24   | 512  | 32       |
| ETTm2   | 96              | 97     | 24   | 512  | 32       |
| ECL     | 24              | 25     | 24   | 512  | 32       |
| Solar   | 144             | 145    | 24   | 512  | 32       |

All datasets use input length $L = 512$, GPT2-small backbone with the first 4 layers (only LayerNorm trainable). The number of compressed tokens $P'$ is fixed to 24 across all datasets. The distilled vocabulary size $V' = 512$ and low-rank dimension $m = 32$.
