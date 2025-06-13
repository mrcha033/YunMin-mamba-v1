"""
SDM Pre-training Script - Pillar 2: Structured Differentiable Masking

This script implements the pre-training of M_SDM with integrated sparsity learning.
The model learns both task performance and channel importance simultaneously.
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

from models.sdm_ssm import SDM_SSM, SDM_MambaBlock
from data.wikitext103 import get_wiktext103_dataloader
from utils.logger import setup_logger, setup_wandb, log_model_info, log_training_info
from utils.profiling import count_parameters


def calculate_sparsity_loss(model: nn.Module) -> torch.Tensor:
    """
    Calculate the sparsity regularization loss L_sparsity = Σ m_c.
    
    [Pillar 2: SDM] This loss encourages the learned masks to be sparse,
    balancing task performance with model compression.
    
    Args:
        model: SDM_SSM model with learnable sparsity masks
        
    Returns:
        Scalar tensor representing average sparsity loss
    """
    total_mask_sum = 0.0
    num_layers = 0
    
    for module in model.modules():
        if isinstance(module, SDM_MambaBlock):
            # Access the stochastic mask generated during forward pass
            if module.stochastic_mask is not None:
                # Sum of mask values represents "amount of computation used"
                total_mask_sum += torch.sum(module.stochastic_mask)
                num_layers += 1
    
    # Return average mask usage across layers and channels
    if num_layers > 0:
        avg_mask_usage = total_mask_sum / num_layers
        return avg_mask_usage / model.layers[0].d_inner  # Normalize by channel count
    else:
        return torch.tensor(0.0, device=next(model.parameters()).device)


def log_sparsity_metrics(model: nn.Module, logger, step: int, wandb_log: bool = False):
    """
    Log detailed sparsity metrics for monitoring training progress.
    
    Args:
        model: SDM_SSM model
        logger: Logger instance
        step: Current training step
        wandb_log: Whether to log to wandb
    """
    if hasattr(model, 'get_sparsity_summary'):
        sparsity_summary = model.get_sparsity_summary()
        
        logger.info(f"Step {step} - Sparsity Summary:")
        logger.info(f"  Overall sparsity: {sparsity_summary['overall_sparsity']:.4f}")
        logger.info(f"  Compression ratio: {sparsity_summary['compression_ratio']:.2f}x")
        logger.info(f"  Channels kept: {sparsity_summary['total_kept']}/{sparsity_summary['total_channels']}")
        
        if wandb_log:
            wandb.log({
                "sparsity/overall_sparsity": sparsity_summary['overall_sparsity'],
                "sparsity/compression_ratio": sparsity_summary['compression_ratio'],
                "sparsity/channels_kept": sparsity_summary['total_kept'],
                "step": step
            })
        
        # Log per-layer statistics
        for layer_stat in sparsity_summary['layer_stats'][:3]:  # First 3 layers
            layer_idx = layer_stat['layer_idx']
            logger.info(f"  Layer {layer_idx}: sparsity={layer_stat['deterministic_sparsity']:.4f}, "
                       f"kept={layer_stat['num_channels_kept']}/{layer_stat['total_channels']}")


def adaptive_temperature_schedule(step: int, total_steps: int, start_temp: float = 5.0, end_temp: float = 0.1) -> float:
    """
    Implement temperature annealing for Gumbel-Sigmoid sampling.
    
    [Pillar 2: SDM] Starting with higher temperature allows more exploration of mask configurations.
    Gradually reducing temperature sharpens the masks toward binary decisions.
    
    Args:
        step: Current training step
        total_steps: Total number of training steps
        start_temp: Initial temperature (higher = more stochastic)
        end_temp: Final temperature (lower = more deterministic)
        
    Returns:
        Current temperature value
    """
    progress = step / total_steps
    return start_temp * (end_temp / start_temp) ** progress


def update_model_temperature(model: nn.Module, new_temp: float):
    """
    Update the Gumbel temperature for all SDM blocks in the model.
    
    Args:
        model: SDM_SSM model
        new_temp: New temperature value
    """
    for module in model.modules():
        if isinstance(module, SDM_MambaBlock):
            module.temperature = new_temp


def save_sdm_checkpoint(model, optimizer, scheduler, step, losses, sparsity_stats, output_dir, config):
    """
    Save SDM model checkpoint with additional sparsity information.
    
    Args:
        model: SDM_SSM model
        optimizer: Optimizer state
        scheduler: Scheduler state
        step: Current step
        losses: Loss history
        sparsity_stats: Current sparsity statistics
        output_dir: Output directory
        config: Model configuration
    """
    checkpoint_dir = os.path.join(output_dir, f"sdm-checkpoint-{step}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Extract importance scores for future SGH-PEFT use
    importance_scores = model.get_layer_importance_scores() if hasattr(model, 'get_layer_importance_scores') else {}
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'step': step,
        'losses': losses,
        'sparsity_stats': sparsity_stats,
        'importance_scores': importance_scores,
        'config': config,
        'sdm_applied': True
    }
    
    torch.save(checkpoint, os.path.join(checkpoint_dir, "pytorch_model.bin"))
    
    # Save config for easy loading
    with open(os.path.join(checkpoint_dir, "config.yaml"), 'w') as f:
        yaml.dump(config, f)
    
    print(f"SDM checkpoint saved at step {step}")


def parse_args():
    parser = argparse.ArgumentParser(description="Pre-train SDM-enhanced SSM model")
    parser.add_argument("--config", type=str, default="configs/pretrain_sdm.yaml",
                        help="Path to SDM configuration file")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/sdm",
                        help="Output directory for checkpoints")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--base_model", type=str, default=None,
                        help="Path to base model (M_base or M_CSP) to initialize from")
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
        name="sdm_pretrain",
        log_file=os.path.join(args.output_dir, "sdm_train.log")
    )
    
    # Setup W&B logging
    if accelerator.is_main_process and config['logging'].get('wandb_project'):
        setup_wandb(
            config=config,
            project=config['logging']['wandb_project'],
            run_name=config['logging'].get('run_name', 'sdm_pretrain')
        )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize SDM model
    model = SDM_SSM(
        d_model=config['model']['d_model'],
        n_layer=config['model']['n_layer'],
        vocab_size=config['model']['vocab_size'],
        d_state=config['model']['d_state'],
        d_conv=config['model']['d_conv'],
        gumbel_temp=config['sdm']['initial_temperature']
    )
    
    # Load base model if specified
    if args.base_model:
        try:
            logger.info(f"Loading base model from {args.base_model}")
            base_checkpoint = torch.load(args.base_model, map_location='cpu')
            
            # Handle different checkpoint formats
            if 'model_state_dict' in base_checkpoint:
                base_state_dict = base_checkpoint['model_state_dict']
            else:
                base_state_dict = base_checkpoint
            
            # Load compatible parameters (skip SDM-specific ones)
            model_state_dict = model.state_dict()
            compatible_params = {}
            
            for name, param in base_state_dict.items():
                if name in model_state_dict and 'z_logits' not in name:
                    if param.shape == model_state_dict[name].shape:
                        compatible_params[name] = param
                        
            model.load_state_dict(compatible_params, strict=False)
            logger.info(f"Loaded {len(compatible_params)} compatible parameters from base model")
            
        except Exception as e:
            logger.warning(f"Could not load base model: {e}")
            logger.info("Starting from random initialization")
    
    # Log model information
    if accelerator.is_main_process:
        log_model_info(logger, model, config['model'])
        log_training_info(logger, config['training'])
        
        # Log SDM-specific info
        logger.info("SDM Configuration:")
        for key, value in config['sdm'].items():
            logger.info(f"  {key}: {value}")
    
    # Create data loaders
    train_dataloader = get_wiktext103_dataloader(
        tokenizer=tokenizer,
        batch_size=config['training']['batch_size'],
        max_length=config['data']['max_length'],
        split="train",
        num_workers=config['data']['num_workers']
    )
    
    val_dataloader = get_wiktext103_dataloader(
        tokenizer=tokenizer,
        batch_size=config['training']['batch_size'],
        max_length=config['data']['max_length'],
        split="validation",
        num_workers=config['data']['num_workers']
    )
    
    # Initialize optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    num_training_steps = config['training']['max_steps']
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config['training']['warmup_steps'],
        num_training_steps=num_training_steps
    )
    
    # Prepare for distributed training
    model, optimizer, train_dataloader, val_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, scheduler
    )
    
    # Training loop
    model.train()
    global_step = 0
    running_loss = 0.0
    running_task_loss = 0.0
    running_sparsity_loss = 0.0
    
    # SDM hyperparameters
    lambda_sparsity = config['sdm']['lambda_sparsity']
    
    logger.info("Starting SDM pre-training...")
    logger.info(f"Sparsity regularization weight (λ): {lambda_sparsity}")
    
    for epoch in range(1000):  # Large number, will break when max_steps reached
        for batch_idx, batch in enumerate(train_dataloader):
            # Update temperature based on schedule
            current_temp = adaptive_temperature_schedule(
                global_step, 
                num_training_steps,
                config['sdm']['initial_temperature'],
                config['sdm']['final_temperature']
            )
            update_model_temperature(accelerator.unwrap_model(model), current_temp)
            
            with accelerator.accumulate(model):
                # Forward pass
                outputs = model(batch["input_ids"])
                
                # Task loss (standard language modeling)
                task_loss = nn.functional.cross_entropy(
                    outputs.view(-1, outputs.size(-1)),
                    batch["labels"].view(-1),
                    ignore_index=-100
                )
                
                # Sparsity regularization loss
                sparsity_loss = calculate_sparsity_loss(accelerator.unwrap_model(model))
                
                # Total loss: L_total = L_task + λ * L_sparsity
                total_loss = task_loss + lambda_sparsity * sparsity_loss
                
                # Backward pass
                accelerator.backward(total_loss)
                
                # Gradient clipping
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config['training']['max_grad_norm'])
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            if accelerator.sync_gradients:
                global_step += 1
                running_loss += total_loss.item()
                running_task_loss += task_loss.item()
                running_sparsity_loss += sparsity_loss.item()
                
                # Logging
                if global_step % config['logging']['log_interval'] == 0:
                    avg_loss = running_loss / config['logging']['log_interval']
                    avg_task_loss = running_task_loss / config['logging']['log_interval']
                    avg_sparsity_loss = running_sparsity_loss / config['logging']['log_interval']
                    
                    logger.info(f"Step {global_step}")
                    logger.info(f"  Total Loss: {avg_loss:.4f}")
                    logger.info(f"  Task Loss: {avg_task_loss:.4f}")
                    logger.info(f"  Sparsity Loss: {avg_sparsity_loss:.4f}")
                    logger.info(f"  Temperature: {current_temp:.4f}")
                    logger.info(f"  LR: {scheduler.get_last_lr()[0]:.2e}")
                    
                    if accelerator.is_main_process and config['logging'].get('wandb_project'):
                        wandb.log({
                            "train/total_loss": avg_loss,
                            "train/task_loss": avg_task_loss,
                            "train/sparsity_loss": avg_sparsity_loss,
                            "train/temperature": current_temp,
                            "train/learning_rate": scheduler.get_last_lr()[0],
                            "train/step": global_step
                        })
                    
                    running_loss = 0.0
                    running_task_loss = 0.0
                    running_sparsity_loss = 0.0
                
                # Sparsity logging
                if global_step % (config['logging']['log_interval'] * 5) == 0:
                    if accelerator.is_main_process:
                        log_sparsity_metrics(
                            accelerator.unwrap_model(model), 
                            logger, 
                            global_step,
                            config['logging'].get('wandb_project') is not None
                        )
                
                # Evaluation
                if global_step % config['logging']['eval_interval'] == 0:
                    model.eval()
                    val_loss = 0.0
                    val_steps = 0
                    
                    with torch.no_grad():
                        for val_batch in val_dataloader:
                            val_outputs = model(val_batch["input_ids"])
                            val_task_loss = nn.functional.cross_entropy(
                                val_outputs.view(-1, val_outputs.size(-1)),
                                val_batch["labels"].view(-1),
                                ignore_index=-100
                            )
                            val_loss += val_task_loss.item()
                            val_steps += 1
                            
                            if val_steps >= 100:  # Limit validation steps
                                break
                    
                    avg_val_loss = val_loss / val_steps
                    logger.info(f"Validation Loss: {avg_val_loss:.4f}")
                    
                    if accelerator.is_main_process and config['logging'].get('wandb_project'):
                        wandb.log({
                            "val/loss": avg_val_loss,
                            "train/step": global_step
                        })
                    
                    model.train()
                
                # Save checkpoint
                if global_step % config['logging']['save_interval'] == 0:
                    if accelerator.is_main_process:
                        sparsity_stats = accelerator.unwrap_model(model).get_sparsity_summary()
                        save_sdm_checkpoint(
                            accelerator.unwrap_model(model),
                            optimizer,
                            scheduler,
                            global_step,
                            {
                                'total_loss': avg_loss,
                                'task_loss': avg_task_loss,
                                'sparsity_loss': avg_sparsity_loss
                            },
                            sparsity_stats,
                            args.output_dir,
                            config
                        )
                
                # Check if training is complete
                if global_step >= config['training']['max_steps']:
                    logger.info("SDM pre-training completed!")
                    
                    # Final checkpoint with importance scores
                    if accelerator.is_main_process:
                        sparsity_stats = accelerator.unwrap_model(model).get_sparsity_summary()
                        save_sdm_checkpoint(
                            accelerator.unwrap_model(model),
                            optimizer,
                            scheduler,
                            global_step,
                            {
                                'total_loss': avg_loss,
                                'task_loss': avg_task_loss,
                                'sparsity_loss': avg_sparsity_loss
                            },
                            sparsity_stats,
                            args.output_dir,
                            config
                        )
                        
                        # Log final sparsity summary
                        logger.info("Final SDM Training Summary:")
                        logger.info("=" * 50)
                        log_sparsity_metrics(accelerator.unwrap_model(model), logger, global_step)
                        
                        # Save importance scores for SGH-PEFT
                        importance_scores = accelerator.unwrap_model(model).get_layer_importance_scores()
                        torch.save(importance_scores, os.path.join(args.output_dir, "importance_scores.pt"))
                        logger.info("Importance scores saved for SGH-PEFT fine-tuning")
                        
                    return


if __name__ == "__main__":
    main() 