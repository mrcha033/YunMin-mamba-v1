# YunMin Mamba v1 - Adaptive Hybrid-PEFT Mamba

A research implementation of an Adaptive Hybrid Parameter-Efficient Fine-Tuning (PEFT) approach for Mamba state space models, featuring three core optimization pillars:

1. **Variable-Aware Scan** (Pillar 1) - Dynamic sequence ordering optimization
2. **Learned Masking** (Pillar 2) - Adaptive sparsity through learned masks  
3. **Hybrid PEFT** (Pillar 3) - Combination of LoRA and IA³ adapters

## Features

- 🔬 **Research-Grade Implementation**: Complete ablation study framework with systematic hyperparameter exploration
- 🚀 **Modular Architecture**: Clean separation of concerns with fallback implementations
- 📊 **Comprehensive Evaluation**: Multi-task evaluation suite (language modeling, summarization, QA, code generation)
- 🎯 **Efficiency Optimization**: 3D efficiency scoring (Accuracy/FLOPs/Parameters)
- 📈 **Advanced Monitoring**: Weights & Biases integration with detailed metrics tracking
- 🧪 **Flexible Configuration**: Extensive configuration options for research and production use

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- 8GB+ RAM

### Dependencies

Install the required dependencies using the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Core Dependencies

- **PyTorch**: `torch>=2.0.0` - Core ML framework
- **Transformers**: `transformers>=4.21.0` - HuggingFace transformers library
- **PEFT**: `peft>=0.4.0` - Parameter-Efficient Fine-Tuning library
- **Mamba SSM** (Optional): `mamba-ssm>=1.0.0` - Optimized Mamba implementation
- **Weights & Biases**: `wandb>=0.13.0` - Experiment tracking (optional)

### Optional: Mamba SSM

For optimal performance, install the official Mamba implementation:

```bash
pip install mamba-ssm>=1.0.0
```

**Note**: If `mamba-ssm` is not available, the system will automatically fall back to a simplified implementation.

## Usage

### 1. Basic Training (`train.py`)

Train an Adaptive Mamba model with default configuration:

```bash
python train.py
```

**Key Features:**
- Supports all three optimization pillars
- Automatic PEFT application after warmup
- Comprehensive logging and checkpointing
- Synthetic dataset generation for quick testing

**Expected Outputs:**
- Training logs with loss progression and learning rates
- Model checkpoints in `./training_outputs/`
- Masking statistics and sparsity metrics
- W&B experiment tracking (if enabled)

**Configuration Options:**
The training script uses a `TrainingConfig` dataclass with extensive configuration options:

- Model: `d_model=256`, `n_layers=4`, `vocab_size=1000`
- Training: `batch_size=16`, `learning_rate=1e-4`, `num_epochs=10`
- PEFT: `peft_r=16`, `peft_alpha=32`, `peft_dropout=0.1`
- IA³: set `enable_ia3=True` to insert scaling layers before LoRA
- Masking: `masking_tau=0.5`, `target_sparsity=0.3`

Setting `enable_ia3=True` causes the training script to call
`insert_ia3_modules(model)` before LoRA adapters are attached, enabling the
lightweight IA³ scaling factors on supported layers.

### 2. Research Ablation Study (`research/research_ablation_study.py`)

Conduct systematic ablation studies across pillar combinations and hyperparameters:

```bash
# Pilot study (quick testing)
python research/research_ablation_study.py --mode pilot

# Quick research mode
python research/research_ablation_study.py --mode quick_research

# Full research mode
python research/research_ablation_study.py --mode research
```

**Command Line Options:**
- `--mode {research,quick_research,pilot}`: Research intensity level
- `--disable-grid-search`: Disable hyperparameter grid search
- `--disable-visualization`: Disable plot generation
- `--epochs N`: Override number of training epochs
- `--project PROJECT_NAME`: W&B project name

**Expected Outputs:**
- `comprehensive_results.json`: Detailed experimental data
- `results.csv`: Tabular results for analysis
- `RESEARCH_REPORT.md`: Generated research report with key findings
- `plots/`: Visualization outputs (efficiency surfaces, parameter impacts, etc.)

**Research Modes:**
- **Pilot**: Fast testing (1 epoch, limited hyperparameters)
- **Quick Research**: Moderate exploration (2 epochs, reduced grid)
- **Research**: Full systematic study (3+ epochs, complete hyperparameter grid)

### 3. Model Evaluation (`research/research_evaluate.py`)

Test evaluation metrics and components:

```bash
python research/research_evaluate.py
```

**Features:**
- Multi-task evaluation framework
- ROUGE metrics for summarization
- Exact match and F1 for QA
- Pass@1 for code generation
- Perplexity for language modeling

**Expected Outputs:**
- Evaluation metric demonstrations
- Component testing results
- Validation of evaluation pipeline

## Scan Order Files

### Understanding `scan_order.npy` and `scan_order_inv.npy`

These files contain permutation indices for the Variable-Aware Scan optimization (Pillar 1):

- **`scan_order.npy`**: Forward permutation indices for reordering sequences
- **`scan_order_inv.npy`**: Inverse permutation for recovering original order

### Generation Process

Scan orders are generated dynamically by the `VariableScanOptimizer` during training:

1. **Initialization**: Random or identity permutation
2. **Update Trigger**: Every `scan_update_frequency` steps (default: 1000)
3. **Optimization**: Based on input sequence statistics and gradients
4. **Persistence**: Automatically saved during model checkpointing

### Reproducing Scan Orders

To generate and save scan orders explicitly:

```python
from layers.variable_scan import VariableScanOptimizer
import torch
import numpy as np

# Create optimizer
optimizer = VariableScanOptimizer(d_model=256, update_frequency=1000)

# Generate sample data
x = torch.randn(32, 128, 256)  # (batch, seq_len, d_model)

# Update permutation
optimizer.update_permutation(x)

# Get and save permutations
perm = optimizer.get_permutation().numpy()
inv_perm = optimizer.get_inverse_permutation().numpy()

np.save('scan_order.npy', perm)
np.save('scan_order_inv.npy', inv_perm)
```

## Weights & Biases (W&B) Integration

### Enabling W&B

W&B is automatically enabled if the library is installed. The system tracks:

- Training metrics (loss, learning rate, perplexity)
- Masking statistics (sparsity, mask probabilities)
- Layer-wise contribution analysis
- Hyperparameter sweeps for ablation studies

### Configuration

Set your W&B project in the training configuration:

```python
config = TrainingConfig(
    project_name="your-project-name",
    run_name="experiment-description"
)
```

### Disabling W&B

To disable W&B completely:

1. **Environment Variable**: 
   ```bash
   export WANDB_MODE=disabled
   python train.py
   ```

2. **Uninstall**: 
   ```bash
   pip uninstall wandb
   ```

3. **Modify Code**: The system gracefully handles missing W&B installation.

## Project Structure

```
YunMin-mamba-v1/
├── README.md                     # This file
├── requirements.txt              # Dependencies
├── train.py                      # Main training script
├── model.py                      # Adaptive Mamba model implementation
├── research/                     # Research modules
│   ├── research_ablation_study.py    # Comprehensive ablation study
│   ├── research_evaluate.py          # Evaluation framework
│   └── research_datasets.py          # Dataset utilities
├── ia3_layers.py                 # IA³ adapter implementation
├── layers/                       # Core layer implementations
│   ├── variable_scan.py          # Variable-aware scan (Pillar 1)
│   ├── masked_linear.py          # Learned masking (Pillar 2)
│   └── ...
├── tests/                        # Test suite
├── training_outputs/             # Default output directory
├── runs/                         # Experiment runs
└── batch_results/               # Batch processing results
```

## Expected Outputs

### Training Session
```
Step 50 | Loss: 2.1234 (CE: 2.1200, Reg: 0.003400) | LR: 0.000100
Step 100 | Loss: 1.9876 (CE: 1.9850, Reg: 0.002600) | LR: 0.000098
...
✅ PEFT applied successfully.
Trainable params: 524,288 || all params: 16,777,216 || trainable%: 3.125
```

### Ablation Study Results
```
🔬 Research-Grade Adaptive Mamba Ablation Study
Mode: research
Configuration: research mode
Hyperparameter grid size: 108
Total experiments: 864
Estimated time: ~1296.0 minutes

🎉 Research study completed!
📁 Results: ./research_outputs_20240101_120000
📊 Report: ./research_outputs_20240101_120000/RESEARCH_REPORT.md
📈 Plots: ./research_outputs_20240101_120000/plots
```

### Model Performance Metrics
- **Parameter Reduction**: 70-90% reduction in trainable parameters
- **Sparsity**: 30-70% layer-wise sparsity through learned masking  
- **Efficiency Score**: Accuracy/(FLOPs × Parameters) optimization
- **Memory Usage**: Significant reduction in GPU memory requirements

## Research Applications

This implementation is designed for:

1. **Parameter Efficiency Research**: Systematic study of PEFT methods
2. **Sequence Modeling Optimization**: Variable-aware scan patterns
3. **Sparsity Learning**: Adaptive masking strategies
4. **Multi-Modal Applications**: Extensible to various sequence tasks
5. **Efficiency Analysis**: 3D trade-off space exploration

## Citation

If you use this implementation in your research, please cite:

```bibtex
@misc{yunmin-mamba-v1,
  title={Adaptive Hybrid-PEFT Mamba: A Three-Pillar Approach to Efficient Sequence Modeling},
  author={[Your Name]},
  year={2024},
  url={https://github.com/[your-repo]/YunMin-mamba-v1}
}
```

## License

This project is released under the MIT License. See `LICENSE` file for details.

## Contributing

Contributions are welcome! Please see `CONTRIBUTING.md` for guidelines.

## Support

For questions and issues:
1. Check the GitHub Issues page
2. Review the research report generated by ablation studies
3. Consult the inline documentation in source files 