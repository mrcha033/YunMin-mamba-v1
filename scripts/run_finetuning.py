"""
SGH-PEFT Fine-tuning Script - Pillar 3: Sparsity-Guided Hybrid PEFT

This script implements the complete fine-tuning pipeline using SGH-PEFT:
1. Load pre-trained SDM model (M_SDM) 
2. Compute layer-wise importance scores from z_logits
3. Apply hybrid LoRA/IA³ adapters based on importance
4. Fine-tune on downstream tasks (GLUE)
The scheduler supports ``warmup_steps_ratio`` to derive warmup steps from ``max_steps``.
"""

import argparse
import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from accelerate import Accelerator
import wandb
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

import sys
# Add project root to path with higher priority
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from models.sdm_ssm import SDM_SSM
from models.sgh_peft import (
    SGHPEFTModel, SGHPEFTConfig, create_sgh_peft_model, 
    compute_layer_importance_scores
)
from data.glue import get_glue_dataloader
from utils.logger import setup_logger, setup_wandb, log_model_info
from utils.profiling import count_parameters


def load_sdm_model(checkpoint_path: str, config: dict) -> SDM_SSM:
    """
    Load pre-trained SDM model from checkpoint.
    
    Args:
        checkpoint_path: Path to SDM checkpoint
        config: Model configuration
        
    Returns:
        Loaded SDM model with learned sparsity patterns
    """
    print(f"Loading SDM model from {checkpoint_path}")
    
    # Create model
    model = SDM_SSM(
        d_model=config['model']['d_model'],
        n_layer=config['model']['n_layer'],
        vocab_size=config['model']['vocab_size'],
        d_state=config['model']['d_state'],
        d_conv=config['model']['d_conv'],
        gumbel_temp=1.0  # Not used during inference
    )
    
    # Load checkpoint
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Load state dict
        model.load_state_dict(state_dict, strict=False)
        print(f"✓ Loaded SDM model with {count_parameters(model)['total_parameters']:,} parameters")
        
        # Log sparsity information if available
        if hasattr(model, 'get_sparsity_summary'):
            sparsity_summary = model.get_sparsity_summary()
            print(f"✓ Model sparsity: {sparsity_summary['overall_sparsity']:.2%}")
            print(f"✓ Compression ratio: {sparsity_summary['compression_ratio']:.2f}x")
        
    else:
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
    
    return model


def create_task_specific_head(num_labels: int, d_model: int) -> nn.Module:
    """
    Create task-specific classification head for GLUE tasks.
    
    Args:
        num_labels: Number of output labels
        d_model: Model dimension
        
    Returns:
        Classification head module
    """
    return nn.Sequential(
        nn.Dropout(0.1),
        nn.Linear(d_model, d_model // 2),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(d_model // 2, num_labels)
    )


class SGHPEFTForSequenceClassification(nn.Module):
    """
    SGH-PEFT model adapted for sequence classification tasks.
    
    This wraps the SGH-PEFT model with a task-specific head for GLUE fine-tuning.
    """
    
    def __init__(self, sgh_peft_model: SGHPEFTModel, num_labels: int):
        super().__init__()
        
        self.backbone = sgh_peft_model
        self.num_labels = num_labels
        
        # Task-specific classification head
        self.classifier = create_task_specific_head(num_labels, sgh_peft_model.embedding.embedding_dim)
        
        # Pooling strategy for sequence classification
        self.pooling_strategy = "mean"  # Options: "mean", "max", "cls"
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for sequence classification.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask (optional)
            
        Returns:
            Classification logits
        """
        # Get sequence representations from SGH-PEFT backbone
        sequence_output = self.backbone(input_ids)  # (batch, seq_len, d_model)
        
        # Pool sequence representations
        if self.pooling_strategy == "mean":
            if attention_mask is not None:
                # Masked mean pooling
                masked_output = sequence_output * attention_mask.unsqueeze(-1)
                pooled_output = masked_output.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
            else:
                pooled_output = sequence_output.mean(dim=1)
        elif self.pooling_strategy == "max":
            pooled_output = sequence_output.max(dim=1)[0]
        else:  # cls
            pooled_output = sequence_output[:, 0]  # Use first token
        
        # Classification
        logits = self.classifier(pooled_output)
        
        return logits


def evaluate_model(model: nn.Module, dataloader: DataLoader, device: torch.device) -> dict:
    """
    Evaluate model on validation/test set.
    
    Args:
        model: Model to evaluate
        dataloader: Evaluation dataloader
        device: Device to run evaluation on
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    
    all_predictions = []
    all_labels = []
    total_loss = 0.0
    num_samples = 0
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            
            # Collect predictions and labels
            predictions = torch.argmax(logits, dim=-1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            total_loss += loss.item()
            num_samples += input_ids.size(0)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    avg_loss = total_loss / len(dataloader)
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'loss': avg_loss,
        'num_samples': num_samples
    }


def save_sgh_peft_checkpoint(
    model: SGHPEFTForSequenceClassification,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    step: int,
    metrics: dict,
    output_dir: str,
    config: dict
):
    """
    Save SGH-PEFT checkpoint with adaptation information.
    
    Args:
        model: SGH-PEFT model
        optimizer: Optimizer state
        scheduler: Scheduler state
        step: Current step
        metrics: Training metrics
        output_dir: Output directory
        config: Configuration
    """
    checkpoint_dir = os.path.join(output_dir, f"sgh-peft-checkpoint-{step}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Get adaptation summary
    adaptation_summary = model.backbone.get_adaptation_summary()
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'step': step,
        'metrics': metrics,
        'adaptation_summary': adaptation_summary,
        'config': config,
        'sgh_peft_applied': True
    }
    
    torch.save(checkpoint, os.path.join(checkpoint_dir, "pytorch_model.bin"))
    
    # Save config
    with open(os.path.join(checkpoint_dir, "config.yaml"), 'w') as f:
        yaml.dump(config, f)
    
    print(f"SGH-PEFT checkpoint saved at step {step}")


def parse_args():
    parser = argparse.ArgumentParser(description="SGH-PEFT Fine-tuning")
    parser.add_argument("--config", type=str, default="configs/finetune_sgh_peft.yaml",
                        help="Path to fine-tuning configuration file")
    parser.add_argument("--sdm_model", type=str, required=True,
                        help="Path to pre-trained SDM model checkpoint")
    parser.add_argument("--task", type=str, default="sst2",
                        help="GLUE task name (sst2, mrpc, qnli, mnli)")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/sgh_peft",
                        help="Output directory for checkpoints")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    args = parse_args()
    config = load_config(args.config)
    
    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        log_with="wandb" if config['logging'].get('wandb_project') else None
    )
    
    # Setup logging
    logger = setup_logger(
        name="sgh_peft_finetune",
        log_file=os.path.join(args.output_dir, "sgh_peft_train.log")
    )
    
    # Setup W&B logging
    if accelerator.is_main_process and config['logging'].get('wandb_project'):
        setup_wandb(
            config=config,
            project=config['logging']['wandb_project'],
            run_name=f"sgh_peft_{args.task}"
        )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Step 1: Load pre-trained SDM model
    logger.info("Step 1: Loading pre-trained SDM model...")
    sdm_model = load_sdm_model(args.sdm_model, config)
    
    # Step 2: Create SGH-PEFT configuration
    logger.info("Step 2: Creating SGH-PEFT configuration...")
    sgh_peft_config = SGHPEFTConfig(
        lora_high_rank=config['sgh_peft']['lora_high_rank'],
        lora_low_rank=config['sgh_peft']['lora_low_rank'],
        lora_alpha_factor=config['sgh_peft']['lora_alpha_factor'],
        lora_dropout=config['sgh_peft']['lora_dropout'],
        high_importance_mean_threshold=config['sgh_peft']['high_importance_mean_threshold'],
        high_importance_active_threshold=config['sgh_peft']['high_importance_active_threshold'],
        medium_importance_mean_threshold=config['sgh_peft']['medium_importance_mean_threshold'],
        medium_importance_active_threshold=config['sgh_peft']['medium_importance_active_threshold'],
        low_importance_mean_threshold=config['sgh_peft']['low_importance_mean_threshold'],
        apply_sparsity_mask=config['sgh_peft']['apply_sparsity_mask'],
        freeze_base_model=config['sgh_peft']['freeze_base_model']
    )
    
    # Step 3: Create SGH-PEFT model
    logger.info("Step 3: Creating SGH-PEFT model...")
    sgh_peft_model = create_sgh_peft_model(sdm_model, sgh_peft_config)
    
    # Step 4: Create task-specific model
    logger.info("Step 4: Creating task-specific model...")
    task_config = config['tasks'][args.task]
    num_labels = task_config['num_labels']
    
    model = SGHPEFTForSequenceClassification(sgh_peft_model, num_labels)
    
    # Log model information
    if accelerator.is_main_process:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Trainable ratio: {trainable_params/total_params:.2%}")
        
        # Log adaptation summary
        adaptation_summary = sgh_peft_model.get_adaptation_summary()
        logger.info("SGH-PEFT Adaptation Summary:")
        for adapter_type, count in adaptation_summary['adapter_distribution'].items():
            logger.info(f"  {adapter_type}: {count} layers")
    
    # Step 5: Create data loaders
    logger.info("Step 5: Creating data loaders...")
    train_dataloader = get_glue_dataloader(
        task_name=args.task,
        tokenizer=tokenizer,
        batch_size=config['training']['batch_size'],
        max_length=args.max_length,
        split="train"
    )
    
    val_dataloader = get_glue_dataloader(
        task_name=args.task,
        tokenizer=tokenizer,
        batch_size=config['training']['batch_size'],
        max_length=args.max_length,
        split="validation"
    )
    
    # Step 6: Initialize optimizer and scheduler
    logger.info("Step 6: Setting up training...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    num_training_steps = config['training']['max_steps']
    warmup_steps = int(config['training'].get('warmup_steps', 0))
    warmup_steps_ratio = config['training'].get('warmup_steps_ratio')
    if warmup_steps_ratio is not None:
        warmup_steps = int(num_training_steps * float(warmup_steps_ratio))

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Prepare for distributed training
    model, optimizer, train_dataloader, val_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, scheduler
    )
    
    # Step 7: Fine-tuning loop
    logger.info("Step 7: Starting SGH-PEFT fine-tuning...")
    
    model.train()
    global_step = 0
    running_loss = 0.0
    best_accuracy = 0.0

    # Early stopping configuration
    early_stop_patience = config['training'].get('early_stopping_patience')
    early_stop_threshold = config['training'].get('early_stopping_threshold', 0.0)
    monitor_metric = config['training'].get('monitor_metric', 'eval_accuracy')
    # Normalize metric key from config to match evaluate_model output
    monitor_key = monitor_metric.replace('eval_', '').replace('eval/', '')
    best_metric = float('-inf')
    wait_count = 0
    
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(1000):  # Large number, will break when max_steps reached
        for batch_idx, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                # Forward pass
                input_ids = batch["input_ids"]
                attention_mask = batch.get("attention_mask", None)
                labels = batch["labels"]
                
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                
                # Backward pass
                accelerator.backward(loss)
                
                # Gradient clipping
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config['training']['max_grad_norm'])
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            if accelerator.sync_gradients:
                global_step += 1
                running_loss += loss.item()
                
                # Logging
                if global_step % config['logging']['log_interval'] == 0:
                    avg_loss = running_loss / config['logging']['log_interval']
                    
                    logger.info(f"Step {global_step}")
                    logger.info(f"  Loss: {avg_loss:.4f}")
                    logger.info(f"  LR: {scheduler.get_last_lr()[0]:.2e}")
                    
                    if accelerator.is_main_process and config['logging'].get('wandb_project'):
                        wandb.log({
                            "train/loss": avg_loss,
                            "train/learning_rate": scheduler.get_last_lr()[0],
                            "train/step": global_step
                        })
                    
                    running_loss = 0.0
                
                # Evaluation
                if global_step % config['logging']['eval_interval'] == 0:
                    logger.info("Running evaluation...")
                    eval_metrics = evaluate_model(model, val_dataloader, accelerator.device)

                    logger.info(f"Evaluation Results:")
                    logger.info(f"  Accuracy: {eval_metrics['accuracy']:.4f}")
                    logger.info(f"  F1: {eval_metrics['f1']:.4f}")
                    logger.info(f"  Loss: {eval_metrics['loss']:.4f}")

                    if accelerator.is_main_process and config['logging'].get('wandb_project'):
                        wandb.log({
                            "eval/accuracy": eval_metrics['accuracy'],
                            "eval/f1": eval_metrics['f1'],
                            "eval/loss": eval_metrics['loss'],
                            "train/step": global_step
                        })

                    # Track metric for early stopping
                    current_metric = eval_metrics.get(monitor_key)
                    if current_metric is not None:
                        if current_metric > best_metric + early_stop_threshold:
                            best_metric = current_metric
                            wait_count = 0
                        else:
                            wait_count += 1
                    else:
                        wait_count += 1

                    # Save best model based on accuracy
                    if eval_metrics['accuracy'] > best_accuracy:
                        best_accuracy = eval_metrics['accuracy']
                        if accelerator.is_main_process:
                            save_sgh_peft_checkpoint(
                                accelerator.unwrap_model(model),
                                optimizer,
                                scheduler,
                                global_step,
                                eval_metrics,
                                args.output_dir,
                                config
                            )

                    model.train()

                    # Early stopping check
                    if early_stop_patience is not None and wait_count >= early_stop_patience:
                        logger.info(
                            f"Early stopping triggered after {early_stop_patience} evaluations "
                            f"without improvement in {monitor_metric}."
                        )
                        final_metrics = eval_metrics
                        logger.info("Final Evaluation Results:")
                        logger.info(f"  Best {monitor_key}: {best_metric:.4f}")
                        logger.info(f"  Final {monitor_key}: {current_metric:.4f}")
                        return
                
                # Check if training is complete
                if global_step >= config['training']['max_steps']:
                    logger.info("SGH-PEFT fine-tuning completed!")
                    
                    # Final evaluation
                    final_metrics = evaluate_model(model, val_dataloader, accelerator.device)
                    logger.info("Final Evaluation Results:")
                    logger.info(f"  Best Accuracy: {best_accuracy:.4f}")
                    logger.info(f"  Final Accuracy: {final_metrics['accuracy']:.4f}")
                    logger.info(f"  Final F1: {final_metrics['f1']:.4f}")
                    
                    return


if __name__ == "__main__":
    main() 