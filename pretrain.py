"""
Main script for pre-training the baseline SSM model (Phase A).
This script implements the M_base training and will be extended for M_SDM with structured masking.
"""

import argparse
import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from accelerate import Accelerator

from models.baseline_ssm import BaselineSSM
from data.wiktext103 import get_wiktext103_dataloader
from utils.logger import setup_logger, setup_wandb, log_model_info, log_training_info
from utils.profiling import count_parameters, count_flops


def parse_args():
    parser = argparse.ArgumentParser(description="Pre-train baseline SSM model")
    parser.add_argument("--config", type=str, default="configs/pretrain_base.yaml", 
                        help="Path to configuration file")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", 
                        help="Output directory for checkpoints")
    parser.add_argument("--resume_from", type=str, default=None, 
                        help="Path to checkpoint to resume from")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_checkpoint(model, optimizer, scheduler, step, loss, output_dir):
    """Save model checkpoint."""
    checkpoint_dir = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'step': step,
        'loss': loss
    }
    
    torch.save(checkpoint, os.path.join(checkpoint_dir, "pytorch_model.bin"))
    print(f"Checkpoint saved at step {step}")


def main():
    args = parse_args()
    config = load_config(args.config)
    
    # Initialize accelerator for distributed training
    accelerator = Accelerator(
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        log_with="wandb" if config['logging'].get('wandb_project') else None
    )
    
    # Setup logging
    logger = setup_logger(
        name="pretrain",
        log_file=os.path.join(args.output_dir, "train.log")
    )
    
    # Setup W&B logging
    if accelerator.is_main_process and config['logging'].get('wandb_project'):
        setup_wandb(
            config=config,
            project=config['logging']['wandb_project'],
            run_name=config['logging']['run_name']
        )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize model
    model = BaselineSSM(
        d_model=config['model']['d_model'],
        n_layer=config['model']['n_layer'],
        vocab_size=config['model']['vocab_size'],
        d_state=config['model']['d_state'],
        d_conv=config['model']['d_conv']
    )
    
    # Log model information
    if accelerator.is_main_process:
        log_model_info(logger, model, config['model'])
        log_training_info(logger, config['training'])
        
        # Profile model
        param_info = count_parameters(model)
        logger.info(f"Parameter analysis: {param_info}")
        
        # Count FLOPs (on CPU to avoid memory issues)
        try:
            flop_info = count_flops(model, (1, config['data']['max_length']), device="cpu")
            logger.info(f"FLOPs analysis: {flop_info['total_flops']:,}")
        except Exception as e:
            logger.warning(f"FLOPs counting failed: {e}")
    
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
    
    logger.info("Starting training...")
    
    for epoch in range(1000):  # Large number, will break when max_steps reached
        for batch_idx, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                # Forward pass
                outputs = model(batch["input_ids"])
                
                # Compute loss (cross-entropy for language modeling)
                loss = nn.functional.cross_entropy(
                    outputs.view(-1, outputs.size(-1)),
                    batch["labels"].view(-1),
                    ignore_index=-100
                )
                
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
                    logger.info(f"Step {global_step}, Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.2e}")
                    
                    if accelerator.is_main_process and config['logging'].get('wandb_project'):
                        import wandb
                        wandb.log({
                            "train/loss": avg_loss,
                            "train/learning_rate": scheduler.get_last_lr()[0],
                            "train/step": global_step
                        })
                    
                    running_loss = 0.0
                
                # Evaluation
                if global_step % config['logging']['eval_interval'] == 0:
                    model.eval()
                    val_loss = 0.0
                    val_steps = 0
                    
                    with torch.no_grad():
                        for val_batch in val_dataloader:
                            val_outputs = model(val_batch["input_ids"])
                            val_loss += nn.functional.cross_entropy(
                                val_outputs.view(-1, val_outputs.size(-1)),
                                val_batch["labels"].view(-1),
                                ignore_index=-100
                            ).item()
                            val_steps += 1
                            
                            if val_steps >= 100:  # Limit validation steps
                                break
                    
                    avg_val_loss = val_loss / val_steps
                    logger.info(f"Validation Loss: {avg_val_loss:.4f}")
                    
                    if accelerator.is_main_process and config['logging'].get('wandb_project'):
                        import wandb
                        wandb.log({
                            "val/loss": avg_val_loss,
                            "train/step": global_step
                        })
                    
                    model.train()
                
                # Save checkpoint
                if global_step % config['logging']['save_interval'] == 0:
                    if accelerator.is_main_process:
                        save_checkpoint(
                            accelerator.unwrap_model(model),
                            optimizer,
                            scheduler,
                            global_step,
                            loss.item(),
                            args.output_dir
                        )
                
                # Check if training is complete
                if global_step >= config['training']['max_steps']:
                    logger.info("Training completed!")
                    
                    # Final checkpoint
                    if accelerator.is_main_process:
                        save_checkpoint(
                            accelerator.unwrap_model(model),
                            optimizer,
                            scheduler,
                            global_step,
                            loss.item(),
                            args.output_dir
                        )
                    
                    return


if __name__ == "__main__":
    main() 