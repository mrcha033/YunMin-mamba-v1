# Experimental Setup Documentation

## Hardware-Data-Parameter Co-Design Framework

This document provides comprehensive implementation details for reproducing the experimental results described in the paper.

## Hardware and Environment

### Target Hardware
- **GPU**: NVIDIA A100 (80GB memory)
- **CUDA Version**: 12.1
- **Framework**: PyTorch 2.1 (cu121)
- **Profiling Tools**: fvcore (FLOPs), PyTorch profiler (Latency)

### Software Environment
- **Python**: 3.9+
- **PyTorch**: 2.1 with CUDA 12.1 support
- **Dependencies**: See `requirements.txt`

## Model Configurations

### Supported Model Sizes
- **Mamba-130M**: 768 dim, 12 layers, ~130M parameters
- **Mamba-370M**: 1024 dim, 24 layers, ~370M parameters

### Model Variants (Ablation Groups)

1. **M_base**: Dense Mamba model (standard baseline)
2. **M_csp**: M_base + CSP (Correlation-based Scan Permutation)
3. **M_sdm**: M_base trained with SDM to learn sparse connectivity
4. **M_sgh**: M_base + SGH-PEFT fine-tuned with proxy-based importance scores (weight magnitude)
5. **M_sdm+sgh**: M_SDM fine-tuned with SGH-PEFT using learned sparsity masks (synergy between SDM & SGH-PEFT)
6. **M_full**: Fully integrated model: CSP applied to SDM-pretrained model and subsequently fine-tuned with SGH-PEFT
7. **M_challenge**: M_base pruned via weight magnitude + fine-tuned with uniform LoRA (Strongest External Baseline)

## Training Hyperparameters

### Phase A: Self-Supervised Pre-training
- **Dataset**: WikiText-103 (Causal Language Modeling)
- **Evaluation Metric**: Perplexity (PPL)
- **Optimizer**: AdamW
- **Learning Rate**: 2e-4
- **Batch Size**: 128
- **Epochs**: 20
- **Warmup Steps**: 10% of total training steps

### Phase B: Fine-tuning
- **Dataset**: GLUE Benchmark (SST-2, MRPC, QNLI, MNLI)
- **Optimizer**: AdamW
- **Learning Rate**: 1e-4
- **Batch Size**: 32
- **Epochs**: Task-dependent
  - SST-2: 5 epochs
  - MNLI: 10 epochs
  - QNLI: 5 epochs
  - MRPC: 8 epochs
- **Early Stopping**: Based on validation accuracy

## Evaluation Metrics

### Performance Metrics
- **Perplexity (PPL)**: Self-supervised model performance (lower is better)
- **Accuracy & F1-Score**: GLUE tasks performance

### Efficiency Metrics
- **Wall-Clock Latency (ms/token)**: Measured on single NVIDIA A100 GPU with standardized batch size
- **Throughput (tokens/sec)**: Measured with large, device-saturating batch size
- **FLOPs**: Computed using fvcore profiling tools
- **Trainable Parameters**: Number of parameters updated during fine-tuning

## Datasets and Evaluation

### Phase A: Self-Supervised Pre-training
- **WikiText-103**: High-quality, long-context articles for SDM to learn meaningful sparsity patterns
- **Evaluation**: Perplexity on validation set

### Phase B: Fine-tuning
- **GLUE Benchmark Tasks**:
  - **SST-2**: Sentiment classification (Accuracy)
  - **MRPC**: Paraphrase identification (F1, Accuracy)
  - **QNLI**: Question-answer inference (Accuracy)
  - **MNLI**: Multi-genre inference (matched/mismatched Accuracy)

## Implementation Details

### Iso-Sparsity Verification
- **M_challenge** sparsity level is set to match the exact sparsity achieved by M_SDM
- Sparsity verification is performed automatically during model generation
- Ensures fair comparison between learned vs. heuristic pruning methodologies

### Hardware Profiling
- **Latency Measurement**: CUDA event-based timing with 100+ iterations
- **Throughput Scaling**: Batch size scaling analysis up to memory limits
- **Memory Profiling**: Peak memory usage tracking
- **Statistical Significance**: Multiple random seeds with confidence intervals

## Reproducibility

### Random Seeds
- **Primary Seed**: 42
- **Statistical Testing**: 5 different seeds for confidence intervals
- **Deterministic Mode**: Enabled for reproducible results

### Configuration Files
- `configs/mamba_130m.yaml`: 130M model configuration
- `configs/mamba_370m.yaml`: 370M model configuration
- `configs/pretrain_sdm.yaml`: SDM pre-training configuration
- `configs/finetune_glue.yaml`: GLUE fine-tuning configuration

## Execution

### Full Experiment Pipeline
```bash
# Run complete experiment
./run_full_experiment.sh 130m 1 experiment_name

# Run with distributed training
./run_full_experiment.sh 370m 4 distributed_experiment
```

### Individual Components
```bash
# Phase A: Pre-training
python pretrain.py --config configs/mamba_130m.yaml

# Phase B: Fine-tuning
python scripts/run_finetuning.py --config configs/finetune_glue.yaml

# Validation
python scripts/run_validation_suite.py --model_group M_full --validate_all
```

## Expected Results

### Performance Benchmarks
- **M_base**: Baseline performance for comparison
- **M_full**: Best performance from complete co-design
- **M_challenge**: Strong external baseline for fair comparison

### Efficiency Gains
- **Latency**: Significant reduction in inference time
- **Throughput**: Improved tokens/second processing
- **Memory**: Reduced memory footprint
- **FLOPs**: Computational efficiency improvements

## Troubleshooting

### Common Issues
1. **Memory Errors**: Reduce batch size or enable gradient checkpointing
2. **CUDA Errors**: Verify CUDA 12.1 and PyTorch 2.1 compatibility
3. **Dataset Loading**: Ensure datasets are cached properly

### Performance Optimization
- Use mixed precision training (bfloat16 on A100)
- Enable torch.compile for additional speedup
- Use distributed training for larger models

## Validation

Use the provided validation script to verify experimental setup:
```bash
python scripts/validate_experimental_setup.py
```

This script automatically checks:
- All model variants can be generated
- Hyperparameters match specifications
- Dataset loading works correctly
- Hardware profiling functions properly 