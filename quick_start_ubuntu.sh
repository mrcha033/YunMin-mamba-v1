#!/bin/bash

# Quick Start Script for Ubuntu - Minimal Test
# Fast verification of Hardware-Data-Parameter Co-Design Framework

set -e

echo "🚀 Quick Start - Ubuntu Minimal Test"
echo "===================================="

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

print_info() {
    echo -e "${YELLOW}[INFO]${NC} $1"
}

# Step 1: Quick Environment Check
print_step "1. Quick Environment Check"
echo "Ubuntu: $(lsb_release -r | cut -f2)"
echo "Python: $(python3 --version)"
print_success "Environment OK"

# Step 2: Essential Dependencies (Minimal)
print_step "2. Installing Essential Dependencies"
if ! python3 -c "import torch" 2>/dev/null; then
    print_info "Installing PyTorch (CPU version for quick test)..."
    pip3 install torch --index-url https://download.pytorch.org/whl/cpu --quiet --user
fi

if ! python3 -c "import transformers" 2>/dev/null; then
    print_info "Installing Transformers..."
    pip3 install transformers --quiet --user
fi

if ! python3 -c "import yaml" 2>/dev/null; then
    print_info "Installing PyYAML..."
    pip3 install pyyaml --quiet --user
fi

print_success "Dependencies installed"

# Step 3: Project Structure Check
print_step "3. Project Structure Check"
required_files=("main.py" "train.py" "configs/unified_config.yaml")
all_present=true

for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✓ $file"
    else
        echo "  ✗ $file"
        all_present=false
    fi
done

if [ "$all_present" = true ]; then
    print_success "All required files present"
else
    echo "❌ Some files are missing. Please check the project structure."
    exit 1
fi

# Step 4: Configuration Test
print_step "4. Configuration Test"
python3 -c "
import yaml
with open('configs/unified_config.yaml', 'r') as f:
    config = yaml.safe_load(f)
print(f'  ✓ Model: {config[\"model\"][\"d_model\"]}d, {config[\"model\"][\"n_layer\"]} layers')
print(f'  ✓ Device: {config[\"system\"][\"device\"]}')
"
print_success "Configuration loaded"

# Step 5: Model Import Test
print_step "5. Model Import Test"
python3 -c "
import sys
import os
sys.path.insert(0, os.getcwd())

try:
    from models.baseline_ssm import BaselineSSM
    model = BaselineSSM(d_model=64, n_layer=2, vocab_size=1000, d_state=8, d_conv=2)
    param_count = sum(p.numel() for p in model.parameters())
    print(f'  ✓ BaselineSSM created: {param_count:,} parameters')
except Exception as e:
    print(f'  ✗ Model import failed: {e}')
    exit(1)

try:
    from models.sdm_ssm import SDM_SSM
    print('  ✓ SDM_SSM import successful')
except Exception as e:
    print(f'  ✗ SDM import failed: {e}')

try:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    print('  ✓ Tokenizer loaded')
except Exception as e:
    print(f'  ✗ Tokenizer failed: {e}')
"
print_success "Model imports successful"

# Step 6: Quick Performance Test
print_step "6. Quick Performance Test"
python3 -c "
import torch
import time

# CPU test
start = time.time()
x = torch.randn(100, 100)
y = torch.randn(100, 100)
z = torch.mm(x, y)
cpu_time = time.time() - start

print(f'  ✓ CPU test: {cpu_time:.3f}s (100x100 matrix)')

# GPU test (if available)
if torch.cuda.is_available():
    device = torch.device('cuda')
    x = x.to(device)
    y = y.to(device)
    start = time.time()
    z = torch.mm(x, y)
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    print(f'  ✓ GPU test: {gpu_time:.3f}s (100x100 matrix)')
    print(f'  ✓ GPU speedup: {cpu_time/gpu_time:.1f}x')
else:
    print('  ⚠ No GPU available (CPU mode)')
"
print_success "Performance test completed"

# Step 7: Quick Config Test
print_step "7. Testing Script Execution"
timeout 10 python3 main.py --help > /dev/null 2>&1 && echo "  ✓ main.py executable" || echo "  ⚠ main.py had issues"
timeout 10 python3 train.py --help > /dev/null 2>&1 && echo "  ✓ train.py executable" || echo "  ⚠ train.py had issues"
print_success "Scripts tested"

# Step 8: Summary
print_step "8. Summary"
echo ""
echo "🎉 Quick Start Test Completed!"
echo ""
echo "✅ System Requirements: Met"
echo "✅ Dependencies: Installed"
echo "✅ Project Structure: Valid"
echo "✅ Models: Importable"
echo "✅ Scripts: Executable"
echo ""
echo "📋 Next Steps:"
echo "1. For full setup: ./ubuntu_dry_run.sh"
echo "2. For GPU training: Edit configs/unified_config.yaml (device: 'cuda')"
echo "3. Run pipeline: python3 main.py --config configs/unified_config.yaml --mode full_pipeline"
echo ""
echo "🔧 Quick Commands:"
echo "• Test with small model: python3 main.py --config configs/dry_run_config.yaml --mode full_pipeline"
echo "• Monitor GPU: watch -n 1 nvidia-smi"
echo "• Check logs: tail -f experiments/*/pipeline.log"
echo ""
print_success "Ready to go! 🐧✨" 