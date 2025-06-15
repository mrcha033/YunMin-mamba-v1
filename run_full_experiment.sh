#!/bin/bash

# Hardware-Data-Parameter Co-Design Framework
# Full-Scale Experiment Execution Script
#
# This script executes the complete experimental pipeline:
# 1. Phase A: Pre-training (WikiText-103)
# 2. Phase B: Fine-tuning (GLUE benchmark)
# 3. Comprehensive validation and analysis
#
# Usage:
#   ./run_full_experiment.sh [MODEL_SIZE] [NUM_GPUS] [EXPERIMENT_NAME]
#
# Examples:
#   ./run_full_experiment.sh 130m 1 baseline_experiment
#   ./run_full_experiment.sh 370m 4 full_scale_validation

set -e  # Exit on any error

# =============================================================================
# Configuration and Setup
# =============================================================================

# Default parameters
MODEL_SIZE=${1:-"130m"}
NUM_GPUS=${2:-1}
EXPERIMENT_NAME=${3:-"full_experiment_$(date +%Y%m%d_%H%M%S)"}

# Validate model size
if [[ ! "$MODEL_SIZE" =~ ^(130m|370m)$ ]]; then
    echo "Error: MODEL_SIZE must be '130m' or '370m'"
    exit 1
fi

# Directories
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="${PROJECT_ROOT}/experiments/${EXPERIMENT_NAME}"
CHECKPOINTS_DIR="${EXPERIMENT_DIR}/checkpoints"
LOGS_DIR="${EXPERIMENT_DIR}/logs"
RESULTS_DIR="${EXPERIMENT_DIR}/results"

# Create directories
mkdir -p "${CHECKPOINTS_DIR}"/{baseline,csp,sdm,sgh,challenge,sdm_sgh,full}
mkdir -p "${LOGS_DIR}"
mkdir -p "${RESULTS_DIR}"

# Configuration files
if [[ "${MODEL_SIZE}" == "370m" ]]; then
    CONFIG_FILE="${PROJECT_ROOT}/configs/mamba_370m_memory_optimized.yaml"
else
    CONFIG_FILE="${PROJECT_ROOT}/configs/mamba_${MODEL_SIZE}.yaml"
fi

# Logging setup
LOG_FILE="${LOGS_DIR}/experiment.log"
exec > >(tee -a "${LOG_FILE}")
exec 2>&1

echo "============================================================================="
echo "HARDWARE-DATA-PARAMETER CO-DESIGN FRAMEWORK"
echo "Full-Scale Experiment Execution"
echo "============================================================================="
echo "Model Size: ${MODEL_SIZE}"
echo "Number of GPUs: ${NUM_GPUS}"
echo "Experiment Name: ${EXPERIMENT_NAME}"
echo "Experiment Directory: ${EXPERIMENT_DIR}"
echo "Configuration: ${CONFIG_FILE}"
echo "Started at: $(date)"
echo "============================================================================="

# =============================================================================
# Environment Setup
# =============================================================================

echo "Setting up environment..."

# Check if config file exists
if [[ ! -f "${CONFIG_FILE}" ]]; then
    echo "Error: Configuration file ${CONFIG_FILE} not found"
    exit 1
fi

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
else
    echo "Warning: nvidia-smi not found. GPU information unavailable."
fi

# Set CUDA visible devices
if [[ ${NUM_GPUS} -gt 1 ]]; then
    export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
    DISTRIBUTED_ARGS="--nproc_per_node=${NUM_GPUS}"
else
    export CUDA_VISIBLE_DEVICES=0
    DISTRIBUTED_ARGS=""
fi

echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"

# =============================================================================
# Phase A: Pre-training Pipeline
# =============================================================================

echo ""
echo "============================================================================="
echo "PHASE A: PRE-TRAINING PIPELINE"
echo "============================================================================="

# Step A1: Baseline Pre-training
echo "Step A1: Baseline Pre-training..."
BASELINE_CHECKPOINT="${CHECKPOINTS_DIR}/baseline/model.pt"

if [[ ! -f "${BASELINE_CHECKPOINT}" ]]; then
    echo "Starting baseline pre-training..."
    
    if [[ ${NUM_GPUS} -gt 1 ]]; then
        python -m torch.distributed.launch ${DISTRIBUTED_ARGS} \
            pretrain.py \
            --config "${CONFIG_FILE}" \
            --output_dir "${CHECKPOINTS_DIR}/baseline" \
            --experiment_name "${EXPERIMENT_NAME}_baseline" \
            --distributed
    else
        python pretrain.py \
            --config "${CONFIG_FILE}" \
            --output_dir "${CHECKPOINTS_DIR}/baseline" \
            --experiment_name "${EXPERIMENT_NAME}_baseline"
    fi
    
    echo "✅ Baseline pre-training completed"
else
    echo "✅ Baseline checkpoint found, skipping pre-training"
fi

# Step A2: CSP Analysis
echo "Step A2: CSP Analysis..."
CSP_CHECKPOINT="${CHECKPOINTS_DIR}/csp/model_csp.pt"

if [[ ! -f "${CSP_CHECKPOINT}" ]]; then
    echo "Running CSP analysis..."
    
    python scripts/run_csp_analysis.py \
        --model_path "${BASELINE_CHECKPOINT}" \
        --output_path "${CSP_CHECKPOINT}" \
        --num_samples 1000
    
    echo "✅ CSP analysis completed"
else
    echo "✅ CSP checkpoint found, skipping analysis"
fi

# Step A3: SDM Pre-training
echo "Step A3: SDM Pre-training..."
SDM_CHECKPOINT="${CHECKPOINTS_DIR}/sdm/model_sdm.pt"

if [[ ! -f "${SDM_CHECKPOINT}" ]]; then
    echo "Starting SDM pre-training..."
    
    # Create SDM-specific config
    SDM_CONFIG="${EXPERIMENT_DIR}/sdm_config.yaml"
    cp "${CONFIG_FILE}" "${SDM_CONFIG}"
    
    if [[ ${NUM_GPUS} -gt 1 ]]; then
        python -m torch.distributed.launch ${DISTRIBUTED_ARGS} \
            pretrain_sdm.py \
            --config "${SDM_CONFIG}" \
            --init_from "${CSP_CHECKPOINT}" \
            --output_dir "${CHECKPOINTS_DIR}/sdm" \
            --experiment_name "${EXPERIMENT_NAME}_sdm" \
            --distributed
    else
        python pretrain_sdm.py \
            --config "${SDM_CONFIG}" \
            --init_from "${CSP_CHECKPOINT}" \
            --output_dir "${CHECKPOINTS_DIR}/sdm" \
            --experiment_name "${EXPERIMENT_NAME}_sdm"
    fi
    
    echo "✅ SDM pre-training completed"
else
    echo "✅ SDM checkpoint found, skipping pre-training"
fi

# Step A4: SDM Analysis
echo "Step A4: SDM Analysis..."
python scripts/analyze_sdm.py

echo "✅ Phase A completed successfully"

# =============================================================================
# Phase B: Fine-tuning Pipeline
# =============================================================================

echo ""
echo "============================================================================="
echo "PHASE B: FINE-TUNING PIPELINE"
echo "============================================================================="

# GLUE tasks to evaluate
GLUE_TASKS=("sst2" "mrpc" "qnli" "mnli")

# Additional model checkpoints
SGH_CHECKPOINT="${CHECKPOINTS_DIR}/sgh/model_sgh.pt"
CHALLENGE_CHECKPOINT="${CHECKPOINTS_DIR}/challenge/model_challenge.pt"
SDM_SGH_CHECKPOINT="${CHECKPOINTS_DIR}/sdm_sgh/model_sdm_sgh.pt"
FULL_CHECKPOINT="${CHECKPOINTS_DIR}/full/model_full.pt"

# Step B1: Generate M_SGH (SGH-PEFT with proxy importance)
echo "Step B1: Generating M_SGH..."

if [[ ! -f "${SGH_CHECKPOINT}" ]]; then
    python scripts/run_sgh_proxy.py \
        --checkpoint "${BASELINE_CHECKPOINT}" \
        --config "${CONFIG_FILE}" \
        --output "${SGH_CHECKPOINT}"
    echo "✅ M_SGH model generated"
else
    echo "✅ M_SGH checkpoint found, skipping generation"
fi

# Step B2: Generate M_sdm_sgh (SDM pretraining followed by SGH-PEFT)
echo "Step B2: Generating M_sdm_sgh..."

if [[ ! -f "${SDM_SGH_CHECKPOINT}" ]]; then
    python scripts/run_sdm_then_sgh.py \
        --sdm_checkpoint "${SDM_CHECKPOINT}" \
        --config "${CONFIG_FILE}" \
        --output "${SDM_SGH_CHECKPOINT}"
    echo "✅ M_sdm_sgh model generated"
else
    echo "✅ M_sdm_sgh checkpoint found, skipping generation"
fi

# Step B3: Generate M_challenge (magnitude pruning + uniform LoRA)
echo "Step B3: Generating M_challenge..."

if [[ ! -f "${CHALLENGE_CHECKPOINT}" ]]; then
    python scripts/run_challenge_baseline.py \
        --checkpoint "${BASELINE_CHECKPOINT}" \
        --sdm_checkpoint "${SDM_CHECKPOINT}" \
        --config "${CONFIG_FILE}" \
        --output "${CHALLENGE_CHECKPOINT}"
    echo "✅ M_challenge model generated"
else
    echo "✅ M_challenge checkpoint found, skipping generation"
fi

# Step B4: Generate M_full model
echo "Step B4: Generating M_full model..."

# Step B0: Generate M_challenge baseline
echo "Step B0: Generating M_challenge baseline..."
if [[ ! -f "${CHALLENGE_CHECKPOINT}" ]]; then
    python scripts/create_challenge_baseline.py \
        --base_model "${BASELINE_CHECKPOINT}" \
        --output_path "${CHALLENGE_CHECKPOINT}" \
        --config "${CONFIG_FILE}" \
        --sdm_checkpoint "${SDM_CHECKPOINT}"
    echo "✅ M_challenge baseline generated"
else
    echo "✅ M_challenge checkpoint found, skipping generation"
fi

if [[ ! -f "${FULL_CHECKPOINT}" ]]; then
    echo "Running full pipeline to generate M_full..."
    
    python scripts/run_full_pipeline.py \
        --base_model "${BASELINE_CHECKPOINT}" \
        --output_dir "${CHECKPOINTS_DIR}/full" \
        --config "${CONFIG_FILE}"
    
    echo "✅ M_full model generated"
else
    echo "✅ M_full checkpoint found, skipping generation"
fi

# Step B5: Fine-tune on GLUE tasks
echo "Step B5: Fine-tuning on GLUE tasks..."

for task in "${GLUE_TASKS[@]}"; do
    echo "Fine-tuning on ${task}..."
    
    TASK_CHECKPOINT="${CHECKPOINTS_DIR}/full/${task}_finetuned.pt"
    
    if [[ ! -f "${TASK_CHECKPOINT}" ]]; then
        python scripts/run_finetuning.py \
            --config "${CONFIG_FILE}" \
            --sdm_model "${FULL_CHECKPOINT}" \
            --task "${task}" \
            --output_dir "${CHECKPOINTS_DIR}/full"
        
        echo "✅ ${task} fine-tuning completed"
    else
        echo "✅ ${task} checkpoint found, skipping fine-tuning"
    fi
done

echo "✅ Phase B completed successfully"

# =============================================================================
# Phase C: Comprehensive Validation
# =============================================================================

echo ""
echo "============================================================================="
echo "PHASE C: COMPREHENSIVE VALIDATION"
echo "============================================================================="

# Step C1: Generate all model variants
echo "Step C1: Generating all model variants..."

# Model variants to validate
declare -A MODEL_VARIANTS=(
    ["M_base"]="${BASELINE_CHECKPOINT}"
    ["M_CSP"]="${CSP_CHECKPOINT}"
    ["M_SDM"]="${SDM_CHECKPOINT}"
    ["M_SGH"]="${SGH_CHECKPOINT}"
    ["M_challenge"]="${CHALLENGE_CHECKPOINT}"
    ["M_sdm_sgh"]="${SDM_SGH_CHECKPOINT}"
    ["M_full"]="${FULL_CHECKPOINT}"
)

# Step C2: Run validation suite for each model
echo "Step C2: Running validation suite..."

for model_name in "${!MODEL_VARIANTS[@]}"; do
    model_path="${MODEL_VARIANTS[$model_name]}"
    
    if [[ -f "${model_path}" ]]; then
        echo "Validating ${model_name}..."
        
        python scripts/run_validation_suite.py \
            --model_group "${model_name}" \
            --checkpoint "${model_path}" \
            --config "${CONFIG_FILE}" \
            --validate_all \
            --output_dir "${RESULTS_DIR}"
        
        echo "✅ ${model_name} validation completed"
    else
        echo "⚠️ ${model_name} checkpoint not found: ${model_path}"
    fi
done

# Step C3: Generate analysis and plots
echo "Step C3: Generating analysis and plots..."

python scripts/analyze_results.py \
    --results_dir "${RESULTS_DIR}" \
    --output_dir "${RESULTS_DIR}/plots"

echo "✅ Analysis and plots generated"

# Step C4: Generate final report
echo "Step C4: Generating final report..."

python scripts/generate_final_report.py \
    --experiment_dir "${EXPERIMENT_DIR}" \
    --config "${CONFIG_FILE}" \
    --output_file "${RESULTS_DIR}/final_report.json"

echo "✅ Final report generated"

# =============================================================================
# Cleanup and Summary
# =============================================================================

echo ""
echo "============================================================================="
echo "EXPERIMENT COMPLETED SUCCESSFULLY"
echo "============================================================================="

# Calculate total time
END_TIME=$(date)
echo "Started at: $(head -n 20 "${LOG_FILE}" | grep "Started at:" | cut -d: -f2-)"
echo "Completed at: ${END_TIME}"

# Display results summary
echo ""
echo "Results Summary:"
echo "  Experiment Directory: ${EXPERIMENT_DIR}"
echo "  Checkpoints: ${CHECKPOINTS_DIR}"
echo "  Results: ${RESULTS_DIR}"
echo "  Logs: ${LOG_FILE}"

# Display key files
echo ""
echo "Key Output Files:"
if [[ -f "${RESULTS_DIR}/final_report.json" ]]; then
    echo "  📋 Final Report: ${RESULTS_DIR}/final_report.json"
fi

if [[ -d "${RESULTS_DIR}/plots" ]]; then
    echo "  📊 Plots Directory: ${RESULTS_DIR}/plots"
    ls -la "${RESULTS_DIR}/plots"/*.png 2>/dev/null | head -5
fi

# Display validation results
echo ""
echo "Validation Results:"
for result_file in "${RESULTS_DIR}"/*_validation.json; do
    if [[ -f "${result_file}" ]]; then
        model_name=$(basename "${result_file}" _validation.json)
        echo "  ✅ ${model_name}: ${result_file}"
    fi
done

# Check for any errors
if grep -q "Error\|Failed\|❌" "${LOG_FILE}"; then
    echo ""
    echo "⚠️ Some errors were detected during execution. Please check the log file:"
    echo "   ${LOG_FILE}"
    echo ""
    echo "Recent errors:"
    grep -n "Error\|Failed\|❌" "${LOG_FILE}" | tail -5
fi

echo ""
echo "🎉 Full-scale experiment completed successfully!"
echo "📊 Results are ready for publication and analysis."
echo ""
echo "Next steps:"
echo "  1. Review final report: ${RESULTS_DIR}/final_report.json"
echo "  2. Examine plots: ${RESULTS_DIR}/plots/"
echo "  3. Analyze validation results in: ${RESULTS_DIR}/"
echo "  4. Check logs for any issues: ${LOG_FILE}"
echo ""
echo "============================================================================="