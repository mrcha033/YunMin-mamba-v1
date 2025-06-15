# Tasks to Fix Experimental Setup

## Priority 1: Training Hyperparameter Misalignments

### Task 1.1: Fix Phase A (Pre-training) Hyperparameters
**File**: `configs/pretrain_sdm.yaml`
**Issues**:
- SDM epochs: Currently 15, should be 20 (as per description)
- Batch size inconsistency: Some configs show 24, should be 128

**Fix**:
```yaml
training:
  pretrain:
    batch_size: 128           # Match description
    max_epochs: 20           # Match description (currently 15)
    learning_rate: 2e-4      # ✅ Already correct
    warmup_steps_ratio: 0.1  # ✅ Already correct (10% of total)
```

### Task 1.2: Fix Phase B (Fine-tuning) Hyperparameters  
**File**: `configs/finetune_glue.yaml`
**Issues**:
- Learning rate: Currently 3e-4, should be 1e-4
- Batch size: Currently 16, should be 32
- Need to verify task-specific epochs match description

**Fix**:
```yaml
training:
  finetune:
    learning_rate: 1e-4      # Fix from 3e-4
    batch_size: 32           # Fix from 16
    micro_batch_size: 8      # Adjust accordingly
    # Verify task-specific epochs:
    # SST-2: 5 epochs ✅
    # MNLI: 10 epochs (need to verify current config)

## Priority 2: Validation and Verification Tasks

### Task 2.1: Verify M_challenge Iso-sparsity Implementation
**File**: `scripts/run_challenge_baseline.py` 
**Requirement**: M_challenge sparsity must exactly match M_SDM sparsity
**Verification needed**:
- Confirm sparsity calculation logic matches between SDM and challenge baseline
- Add explicit sparsity ratio logging and validation

### Task 2.2: Add Missing Implementation Details
**Files**: Various config files
**Add missing specs from description**:
- Warmup steps: 10% of total training steps ✅ (already implemented)
- Early stopping based on validation accuracy ✅ (already implemented) 
- AdamW optimizer ✅ (already specified)

## Priority 3: Documentation and Consistency

### Task 3.1: Update Hardware Environment Documentation
**File**: `README.md` or add `EXPERIMENTAL_SETUP.md`
**Add explicit hardware specs**:
```markdown
## Hardware and Environment
- GPU: NVIDIA A100 (80GB memory)
- CUDA version: 12.1  
- Framework: PyTorch 2.1 (cu121)
- Profiling Tools: fvcore (FLOPs), PyTorch profiler (Latency)
```

### Task 3.2: Validate GLUE Task Epochs
**File**: `configs/mamba_130m.yaml`
**Current task epochs**:
```yaml
epochs:
  sst2: 5     # ✅ Matches description
  mnli: 8     # ❓ Description says 10 epochs  
  qnli: 5     # ✅ Not explicitly specified in description
  mrpc: 8     # ✅ Not explicitly specified in description
```
**Action**: Verify MNLI should be 10 epochs per description

## Priority 4: Testing and Validation

### Task 4.1: Add Experimental Validation Script
**New File**: `scripts/validate_experimental_setup.py`
**Purpose**: Automatically verify all hyperparameters and configurations match the experimental description
**Features**:
- Check all 7 model variants can be generated
- Validate hyperparameters match specification
- Confirm dataset loading works for both WikiText-103 and GLUE
- Test hardware profiling on target A100 specs

### Task 4.2: Add Sparsity Verification
**Enhancement**: `scripts/run_validation_suite.py`
**Add check**: Verify M_challenge and M_SDM have identical sparsity ratios
```python
def verify_iso_sparsity(sdm_model_path, challenge_model_path):
    """Verify challenge baseline matches SDM sparsity exactly."""
    # Implementation needed
```

## Summary of Files to Modify:

1. **configs/pretrain_sdm.yaml** - Fix epochs (15→20) and batch size consistency
2. **configs/finetune_glue.yaml** - Fix learning rate (3e-4→1e-4) and batch size (16→32)  
3. **configs/mamba_130m.yaml** - Fix PyTorch version (2.2→2.1), verify MNLI epochs (8→10)
4. **configs/mamba_370m.yaml** - Same version fixes
5. **scripts/run_challenge_baseline.py** - Add sparsity verification logging
6. **CREATE: scripts/validate_experimental_setup.py** - New validation script

## Estimated Impact:
- **High**: Hyperparameter fixes ensure reproducibility matches paper specification
- **Medium**: Version consistency prevents environment-related issues  
- **Low**: Documentation improvements aid reproducibility

All identified issues are addressable and don't require major architectural changes to the experimental framework. 