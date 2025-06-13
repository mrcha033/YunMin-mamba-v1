# Hardware-Data-Parameter Co-Design for State Space Models

This repository implements a three-pillar co-design framework for optimizing State Space Models (SSMs) across hardware, data, and parameter dimensions. The project addresses the fundamental bottlenecks in SSM deployment through coordinated optimization strategies.

## Overview

The co-design framework consists of three integrated pillars:

1. **Pillar 1: Correlation-based Scan Permutation (CSP)** - Hardware-level optimization targeting memory access patterns in the SSM scan operation
2. **Pillar 2: Structured Differentiable Masking (SDM)** - Data-level optimization learning sparse channel structures during pre-training
3. **Pillar 3: Sparsity-Guided Hybrid PEFT (SGH-PEFT)** - Parameter-level optimization using structured importance scores to guide fine-tuning strategies

## Directory Structure

```
hardware-data-parameter-codesign/
├── README.md                 # Project overview and setup guide
├── requirements.txt          # Python dependencies
│
├── configs/                  # Experiment configurations
│   ├── pretrain_base.yaml    # M_base and M_SDM hyperparameters
│   └── finetune_glue.yaml    # SGH-PEFT hyperparameters
│
├── data/                     # Data loading and processing
│   ├── wiktext103.py         # WikiText-103 dataloader
│   └── glue.py               # GLUE benchmark dataloader
│
├── models/                   # Model architectures
│   ├── __init__.py
│   └── baseline_ssm.py       # Baseline SSM implementation (M_base)
│
├── scripts/                  # Analysis and evaluation scripts
│   ├── run_csp_analysis.py   # CSP offline analysis
│   ├── evaluate_latency.py   # Latency measurement
│   └── run_finetuning.py     # Fine-tuning with SGH-PEFT
│
├── utils/                    # Utility functions
│   ├── logger.py             # Logging utilities
│   └── profiling.py          # Performance profiling
│
└── pretrain.py               # Main pre-training script
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd hardware-data-parameter-codesign
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Set up Weights & Biases for experiment tracking:
```bash
wandb login
```

## Phase 0: Baseline Establishment

### Step 1: Model Architecture Verification

The baseline SSM model (`M_base`) is implemented in `models/baseline_ssm.py`. To verify the implementation:

```python
from models.baseline_ssm import BaselineSSM
import torch

# Initialize baseline model
model = BaselineSSM(
    d_model=768,
    n_layer=12,
    vocab_size=50257,
    d_state=16,
    d_conv=4
)

# Test forward pass
input_ids = torch.randint(0, 50257, (2, 1024))
outputs = model(input_ids)
print(f"Output shape: {outputs.shape}")  # Should be [2, 1024, 50257]
```

### Step 2: Performance Profiling

Establish baseline metrics for comparison:

```python
from utils.profiling import count_parameters, count_flops, measure_latency

# Parameter count
param_info = count_parameters(model)
print(f"Total parameters: {param_info['total_parameters']:,}")

# FLOPs analysis
flop_info = count_flops(model, (1, 1024))
print(f"Total FLOPs: {flop_info['total_flops']:,}")

# Latency measurement (requires CUDA)
latency_info = measure_latency(model, (1, 1024), device="cuda")
print(f"Mean latency: {latency_info['mean_latency_ms']:.2f}ms")
```

### Step 3: Pre-training Setup

Configure and run baseline pre-training:

```bash
# Edit configs/pretrain_base.yaml as needed
python pretrain.py --config configs/pretrain_base.yaml --output_dir ./checkpoints/baseline
```

## Optimization Pipeline

### Phase A: Hardware-Aware Pre-training

1. **CSP Analysis**: Run correlation analysis to find optimal state permutation
2. **SDM Training**: Pre-train with structured differentiable masking
3. **Baseline Comparison**: Compare M_SDM against M_base

### Phase B: Parameter-Aware Fine-tuning

1. **Importance Scoring**: Extract layer importance from SDM training
2. **SGH-PEFT Application**: Apply hybrid LoRA/IA³ based on importance
3. **GLUE Evaluation**: Evaluate on downstream tasks

## Key Components

### BaselineSSM Architecture

The `BaselineSSM` class implements the core Mamba architecture with:
- Embedding layer and language modeling head
- Stack of `MambaBlock` modules
- Residual connections and layer normalization

### MambaBlock Components

Each `MambaBlock` contains:
- **Input Projection**: Target for SDM channel masking
- **1D Convolution**: Local context modeling
- **SSM Core**: State transition dynamics (target for CSP)
- **Output Projection**: Final linear transformation

### Optimization Targets

The codebase is designed with clear optimization targets:

1. **CSP Targets**: `A_log`, `x_proj` parameters in the SSM core
2. **SDM Targets**: `in_proj` layer channels
3. **SGH-PEFT Targets**: Layer-wise importance scores guide adapter selection

## Configuration

### Pre-training Configuration (`configs/pretrain_base.yaml`)

```yaml
model:
  d_model: 768          # Model dimension
  n_layer: 12           # Number of layers
  d_state: 16           # SSM state dimension
  d_conv: 4             # Convolution kernel size

training:
  batch_size: 32        # Training batch size
  learning_rate: 1e-4   # Learning rate
  max_steps: 100000     # Maximum training steps
```

### Fine-tuning Configuration (`configs/finetune_glue.yaml`)

```yaml
peft:
  lora:
    r: 16               # LoRA rank
    lora_alpha: 32      # LoRA scaling factor
  
  importance_scoring:
    threshold: 0.3      # Threshold for LoRA vs IA³ selection
```

## Evaluation

### Performance Metrics

- **Latency**: Wall-clock inference time
- **Throughput**: Tokens per second
- **Memory**: GPU memory consumption
- **FLOPs**: Computational complexity
- **Accuracy**: Downstream task performance

### Benchmarks

- **Pre-training**: WikiText-103 perplexity
- **Fine-tuning**: GLUE benchmark suite
- **Efficiency**: Parameter count, FLOPs, latency

## Reproduction

To reproduce the results:

1. **Baseline Training**:
```bash
python pretrain.py --config configs/pretrain_base.yaml
```

2. **CSP Analysis**:
```bash
python scripts/run_csp_analysis.py --model_path checkpoints/baseline
```

3. **SDM Training**:
```bash
python pretrain.py --config configs/pretrain_sdm.yaml
```

4. **SGH-PEFT Fine-tuning**:
```bash
python scripts/run_finetuning.py --config configs/finetune_glue.yaml
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{codesign2024,
  title={Hardware-Data-Parameter Co-Design for State Space Models},
  author={Yunmin Cha},
  journal={arXiv preprint},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 