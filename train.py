"""
Comprehensive Training Script for Adaptive Hybrid-PEFT Mamba
Orchestrates all three pillars: Variable-Aware Scan, Learned Masking, and Hybrid PEFT
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
import warnings

# Model imports
try:
    from .model import AdaptiveMambaModel
except ImportError:
    from model import AdaptiveMambaModel

# PEFT imports with fallback
try:
    from peft import LoraConfig, IA3Config, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    warnings.warn("PEFT library not available. PEFT functionality will be disabled.")
    PEFT_AVAILABLE = False

# IA3 utilities
try:
    from .layers.ia3_layers import insert_ia3_modules
except ImportError:
    from layers.ia3_layers import insert_ia3_modules

# Wandb for logging with fallback
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    warnings.warn("Wandb not available. Logging will be to console only.")
    WANDB_AVAILABLE = False

@dataclass
class TrainingConfig:
    """Configuration for training process."""
    
    # Model configuration
    vocab_size: int = 1000
    d_model: int = 256
    n_layers: int = 4
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    
    # Training hyperparameters
    batch_size: int = 16
    learning_rate: float = 1e-4
    num_epochs: int = 10
    max_seq_length: int = 128
    gradient_clip: float = 1.0
    warmup_steps: int = 100
    
    # Masking configuration (Pillar 2)
    enable_masking: bool = True
    masking_tau: float = 0.5
    masking_init_sparsity: float = 0.5
    masking_target_sparsity: float = 0.3
    sparsity_weight: float = 1e-5
    
    # PEFT configuration (Pillar 3)
    enable_peft: bool = True
    peft_r: int = 16
    peft_alpha: int = 32
    peft_dropout: float = 0.1
    enable_ia3: bool = False
    importance_threshold: float = 0.5  # Fraction of tuned layers to apply LoRA
    peft_application_ratio: float = 0.3  # Fraction of total layers to adapt
    
    # Scan optimization (Pillar 1)
    scan_update_frequency: int = 1000
    
    # Logging and saving
    log_interval: int = 50
    save_interval: int = 1000
    project_name: str = "adaptive-mamba"
    run_name: Optional[str] = None
    output_dir: str = "./outputs"
    save_checkpoints: bool = True
    
    # Evaluation
    eval_interval: int = 200
    eval_steps: int = 50

class PEFTManager:
    """
    🚀 SELF-OPTIMIZING PEFT MANAGER 🚀
    
    Implements the "Self-Optimizing" paradigm where the model learns its own inefficiencies
    and applies the most economical tuning methods accordingly. This is the heart of the
    Adaptive Hybrid-PEFT Mamba architecture.
    
    Core Philosophy: Dynamic allocation of LoRA to high-importance layers and IA³ to 
    mid-importance layers based on learned importance patterns from Pillar 2.
    """
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.peft_applied = False
        self.importance_cache = {}  # Cache importance scores for consistency

    def _get_comprehensive_importance_scores(self, model: nn.Module) -> Dict[str, float]:
        """
        Extract importance scores from ALL eligible layers across the entire model.
        This is the foundation of the self-optimizing paradigm.
        
        Returns:
            Dict mapping layer names to importance scores (higher = more important)
        """
        scores: Dict[str, float] = {}
        
        # Method 1: Collect from Pillar 2 (Learned Masking) - Primary method
        if hasattr(model, "blocks"):
            for i, block in enumerate(model.blocks):
                if hasattr(block, "get_importance_scores"):
                    block_scores = block.get_importance_scores(method="mask_probability")
                    for layer_name, score in block_scores.items():
                        full_name = f"blocks.{i}.{layer_name}"
                        scores[full_name] = score
                        
        # Method 2: Fallback for non-masked layers (weight magnitude based)
        fallback_modules = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)) and name not in scores:
                # Only include modules that can benefit from PEFT
                if any(target in name for target in ['mixer', 'in_proj', 'out_proj', 'conv1d']):
                    scores[name] = module.weight.abs().mean().item()
                    fallback_modules.append(name)
        
        if fallback_modules:
            logging.info(f"Using fallback importance calculation for {len(fallback_modules)} non-masked modules")
            
        # Cache scores for consistency across the training process
        self.importance_cache = scores.copy()
        
        logging.info(f"Collected importance scores from {len(scores)} eligible layers")
        return scores

    def _allocate_peft_methods(self, importance_scores: Dict[str, float]) -> Tuple[List[str], List[str], List[str]]:
        """
        🎯 CORE SELF-OPTIMIZATION LOGIC 🎯
        
        Intelligently allocate tuning methods based on learned importance patterns:
        - High importance → LoRA (maximum expressiveness)
        - Medium importance → IA³ (parameter efficiency)  
        - Low importance → Frozen (computational savings)
        
        This implements the mathematical framework from docs/math_spec.md
        
        Returns:
            Tuple of (lora_layers, ia3_layers, frozen_layers) - lists of layer names
        """
        if not importance_scores:
            logging.warning("No importance scores available for PEFT allocation")
            return [], [], []
            
        # Sort layers by importance (descending)
        sorted_layers = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Calculate allocation thresholds based on configuration
        total_layers = len(sorted_layers)
        num_layers_to_tune = max(1, int(total_layers * self.config.peft_application_ratio))
        num_lora_layers = max(1, int(num_layers_to_tune * self.config.importance_threshold))
        num_ia3_layers = num_layers_to_tune - num_lora_layers
        
        # Allocate methods
        lora_layers = [name for name, _ in sorted_layers[:num_lora_layers]]
        ia3_layers = [name for name, _ in sorted_layers[num_lora_layers:num_layers_to_tune]]
        frozen_layers = [name for name, _ in sorted_layers[num_layers_to_tune:]]
        
        # Log the self-optimization decision process
        logging.info("🧠 SELF-OPTIMIZATION ALLOCATION SUMMARY:")
        logging.info(f"   Total layers analyzed: {total_layers}")
        logging.info(f"   Layers selected for tuning: {num_layers_to_tune} ({self.config.peft_application_ratio:.1%})")
        logging.info(f"   LoRA allocation: {len(lora_layers)} high-importance layers")
        logging.info(f"   IA³ allocation: {len(ia3_layers)} mid-importance layers") 
        logging.info(f"   Frozen: {len(frozen_layers)} low-importance layers")
        
        return lora_layers, ia3_layers, frozen_layers

    def _log_detailed_allocation(self, sorted_layers: List[Tuple[str, float]], 
                                lora_layers: List[str], ia3_layers: List[str]):
        """Log detailed allocation decisions for transparency and debugging."""
        logging.info("📊 DETAILED IMPORTANCE-DRIVEN ALLOCATION:")
        
        # Show top layers with their allocation decisions
        top_layers_to_show = min(15, len(sorted_layers))
        for name, score in sorted_layers[:top_layers_to_show]:
            if name in lora_layers:
                method = "LoRA (High-Importance)"
            elif name in ia3_layers:
                method = "IA³ (Mid-Importance)"
            else:
                method = "Frozen (Low-Importance)"
            
            # Truncate long names for readability
            display_name = name if len(name) <= 35 else name[:32] + "..."
            logging.info(f"   {display_name:<35} | {score:.4f} | {method}")
        
        if len(sorted_layers) > top_layers_to_show:
            logging.info(f"   ... and {len(sorted_layers) - top_layers_to_show} more layers")

    def apply_peft_to_model(self, model: nn.Module) -> Tuple[nn.Module, List[nn.Parameter]]:
        """
        🚀 APPLY SELF-OPTIMIZING PEFT ALLOCATION 🚀
        
        This is where the magic happens - the model learns its own inefficiencies
        and applies the most economical tuning strategy automatically.
        
        Returns:
            Tuple of (modified_model, new_peft_parameters)
        """
        if not PEFT_AVAILABLE or not self.config.enable_peft or self.peft_applied:
            logging.info("PEFT application skipped (already applied or disabled)")
            return model, []

        logging.info("🧠 INITIATING SELF-OPTIMIZING PEFT ALLOCATION...")
        
        # Step 1: Extract comprehensive importance scores (Self-Analysis)
        importance_scores = self._get_comprehensive_importance_scores(model)
        if not importance_scores:
            logging.error("❌ Failed to extract importance scores. Cannot proceed with PEFT allocation.")
            return model, []

        # Step 2: Intelligent allocation based on learned patterns (Self-Optimization)
        lora_layers, ia3_layers, frozen_layers = self._allocate_peft_methods(importance_scores)
        
        # Step 3: Detailed logging for transparency
        sorted_layers = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
        self._log_detailed_allocation(sorted_layers, lora_layers, ia3_layers)
        
        # Step 4: Apply IA³ to mid-importance layers (Parameter Efficiency)
        params_before = set(model.parameters())
        if ia3_layers:
            logging.info(f"🔧 Applying IA³ to {len(ia3_layers)} mid-importance layers...")
            insert_ia3_modules(model, target_module_names=ia3_layers)
            logging.info("✅ IA³ application completed")
        
        # Step 5: Apply LoRA to high-importance layers (Maximum Expressiveness)
        if lora_layers:
            logging.info(f"🔧 Applying LoRA to {len(lora_layers)} high-importance layers...")
            
            # Convert full module names to target module types for peft library
            target_module_types = set()
            for name in lora_layers:
                if 'in_proj' in name:
                    target_module_types.add('in_proj')
                if 'out_proj' in name:
                    target_module_types.add('out_proj')
                if 'conv1d' in name:
                    target_module_types.add('conv1d')
                # Add other module types as needed
            
            if target_module_types:
                lora_config = LoraConfig(
                    r=self.config.peft_r,
                    lora_alpha=self.config.peft_alpha,
                    lora_dropout=self.config.peft_dropout,
                    target_modules=list(target_module_types),
                    bias="none",
                    task_type=TaskType.CAUSAL_LM,
                )
                model = get_peft_model(model, lora_config)
                logging.info("✅ LoRA application completed")
            else:
                logging.warning("⚠️ No compatible modules found for LoRA application")
        
        # Step 6: Calculate and report efficiency gains
        params_after = set(model.parameters())
        new_params = [p for p in params_after if p not in params_before and p.requires_grad]
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        efficiency_ratio = trainable_params / total_params
        
        logging.info("🎯 SELF-OPTIMIZATION RESULTS:")
        logging.info(f"   New PEFT parameters: {len(new_params):,}")
        logging.info(f"   Total parameters: {total_params:,}")
        logging.info(f"   Trainable parameters: {trainable_params:,}")
        logging.info(f"   Efficiency ratio: {efficiency_ratio:.1%} (lower is better)")
        
        if hasattr(model, "print_trainable_parameters"):
            model.print_trainable_parameters()

        self.peft_applied = True
        logging.info("✅ SELF-OPTIMIZING PEFT ALLOCATION COMPLETED SUCCESSFULLY")
        
        return model, new_params
    
    def get_allocation_summary(self) -> Dict[str, Any]:
        """Get summary of PEFT allocation for reporting and analysis."""
        if not self.peft_applied:
            return {"status": "not_applied"}
            
        return {
            "status": "applied",
            "total_layers_analyzed": len(self.importance_cache),
            "config": {
                "importance_threshold": self.config.importance_threshold,
                "peft_application_ratio": self.config.peft_application_ratio,
                "lora_rank": self.config.peft_r,
                "lora_alpha": self.config.peft_alpha
            },
            "importance_scores": self.importance_cache.copy()
        }

class SimpleDataset(Dataset):
    """Simple synthetic dataset for demonstration."""
    
    def __init__(self, vocab_size: int, seq_length: int, num_samples: int = 1000):
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.num_samples = num_samples
        
        # Generate random sequences
        self.data = torch.randint(0, vocab_size, (num_samples, seq_length))
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        sequence = self.data[idx]
        # For language modeling: input is sequence[:-1], target is sequence[1:]
        return {
            'input_ids': sequence[:-1],
            'labels': sequence[1:]
        }

class AdaptiveMambaTrainer:
    """
    Complete trainer for Adaptive Mamba model.
    Orchestrates all three pillars and manages the training process.
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup logging
        self._setup_logging()
        
        # Initialize model
        self.model = self._create_model()
        self.model.to(self.device)
        
        # Initialize PEFT manager
        self.peft_manager = PEFTManager(config)
        
        # Initialize optimizer and scheduler
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config.learning_rate)
        self.scheduler = self._create_scheduler()
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        
        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        
        logging.info(f"Trainer initialized. Device: {self.device}")
        logging.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        # Create output directory first
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.config.output_dir, 'training.log')),
                logging.StreamHandler()
            ]
        )
        
        # Initialize Wandb if available
        if WANDB_AVAILABLE:
            wandb.init(
                project=self.config.project_name,
                name=self.config.run_name,
                config=asdict(self.config)
            )
    
    def _create_model(self) -> AdaptiveMambaModel:
        """Create the Adaptive Mamba model with proper configuration."""
        block_config = {
            'd_state': self.config.d_state,
            'd_conv': self.config.d_conv,
            'expand': self.config.expand,
            'enable_masking': self.config.enable_masking,
            'masking_config': {
                'tau': self.config.masking_tau,
                'init_sparsity': self.config.masking_init_sparsity,
                'target_sparsity': self.config.masking_target_sparsity,
                'sparsity_weight': self.config.sparsity_weight
            }
        }
        
        return AdaptiveMambaModel(
            vocab_size=self.config.vocab_size,
            d_model=self.config.d_model,
            n_layers=self.config.n_layers,
            block_config=block_config
        )
    
    def _create_scheduler(self):
        """Create learning rate scheduler with warmup."""
        def lr_lambda(step):
            if step < self.config.warmup_steps:
                return step / self.config.warmup_steps
            return 1.0
        
        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def _create_dataloader(self, dataset: Dataset, shuffle: bool = True) -> DataLoader:
        """Create dataloader with proper configuration."""
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=0,  # Avoid multiprocessing issues
            pin_memory=True if self.device.type == 'cuda' else False
        )
    
    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor, 
                    regularization_loss: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total loss with regularization.
        
        Returns:
            Tuple of (total_loss, loss_components)
        """
        # Cross-entropy loss
        ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        # Total loss with regularization
        total_loss = ce_loss + regularization_loss
        
        loss_components = {
            'cross_entropy': ce_loss.item(),
            'regularization': regularization_loss.item(),
            'total': total_loss.item()
        }
        
        return total_loss, loss_components
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Execute a single training step."""
        self.model.train()
        
        # Move batch to device
        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        # Forward pass with scan update logic
        update_scan = (self.global_step % self.config.scan_update_frequency) == 0
        logits = self.model(input_ids, update_scan=update_scan)
        
        # Get regularization loss (Pillar 2: Learned Masking)
        reg_loss = self.model.get_total_regularization_loss()
        
        # Compute total loss
        total_loss, loss_components = self.compute_loss(logits, labels, reg_loss)
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
        
        self.optimizer.step()
        self.scheduler.step()
        
        # Collect training metrics
        metrics = {
            'loss': loss_components,
            'learning_rate': self.scheduler.get_last_lr()[0],
            'step': self.global_step
        }
        
        # Get masking statistics (Pillar 2)
        if self.config.enable_masking:
            masking_stats = {}
            for i, block in enumerate(self.model.blocks):
                block_stats = block.get_masking_statistics()
                if block_stats:
                    masking_stats[f'block_{i}'] = block_stats
            metrics['masking'] = masking_stats
        
        # Apply PEFT after warmup (Pillar 3)
        if self.global_step == self.config.warmup_steps and not self.peft_manager.peft_applied:
            self.model, new_peft_params = self.peft_manager.apply_peft_to_model(self.model)
            if new_peft_params:
                # Add only the new PEFT parameters to the existing optimizer
                self.optimizer.add_param_group({"params": new_peft_params})
                logging.info("Added PEFT parameter group to existing optimizer.")
        
        self.global_step += 1
        return metrics
    
    def evaluate(self, eval_dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate the model."""
        self.model.eval()
        total_loss = 0.0
        total_steps = 0
        
        with torch.no_grad():
            for step, batch in enumerate(eval_dataloader):
                if step >= self.config.eval_steps:
                    break
                
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                logits = self.model(input_ids)
                reg_loss = self.model.get_total_regularization_loss()
                
                total_loss_step, _ = self.compute_loss(logits, labels, reg_loss)
                total_loss += total_loss_step.item()
                total_steps += 1
        
        return {'eval_loss': total_loss / total_steps if total_steps > 0 else 0.0}
    
    def log_metrics(self, metrics: Dict[str, Any], prefix: str = "train"):
        """Log metrics to console and wandb."""
        # Console logging
        if self.global_step % self.config.log_interval == 0:
            loss_info = metrics.get('loss', {})
            lr = metrics.get('learning_rate', 0.0)
            
            log_str = f"Step {self.global_step} | "
            log_str += f"Loss: {loss_info.get('total', 0.0):.4f} "
            log_str += f"(CE: {loss_info.get('cross_entropy', 0.0):.4f}, "
            log_str += f"Reg: {loss_info.get('regularization', 0.0):.6f}) | "
            log_str += f"LR: {lr:.6f}"
            
            logging.info(log_str)
        
        # Wandb logging
        if WANDB_AVAILABLE:
            wandb_metrics = {}
            
            # Flatten loss metrics
            if 'loss' in metrics:
                for k, v in metrics['loss'].items():
                    wandb_metrics[f"{prefix}/loss_{k}"] = v
            
            # Add other metrics
            if 'learning_rate' in metrics:
                wandb_metrics[f"{prefix}/learning_rate"] = metrics['learning_rate']
            
            # Note: Importance scores no longer tracked dynamically in simplified PEFT approach
            
            # Add masking statistics
            if 'masking' in metrics:
                for block_name, block_stats in metrics['masking'].items():
                    for layer_name, stats in block_stats.items():
                        for stat_name, stat_value in stats.items():
                            wandb_metrics[f"{prefix}/masking_{block_name}_{layer_name}_{stat_name}"] = stat_value
            
            wandb.log(wandb_metrics, step=self.global_step)
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'current_epoch': self.current_epoch,
            'config': asdict(self.config)
        }
        
        torch.save(checkpoint, path)
        logging.info(f"Checkpoint saved to {path}")
    
    def train(self, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None):
        """Main training loop."""
        train_dataloader = self._create_dataloader(train_dataset, shuffle=True)
        eval_dataloader = self._create_dataloader(eval_dataset, shuffle=False) if eval_dataset else None
        
        logging.info(f"Starting training for {self.config.num_epochs} epochs")
        logging.info(f"Training samples: {len(train_dataset)}")
        if eval_dataset:
            logging.info(f"Evaluation samples: {len(eval_dataset)}")
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            logging.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            
            for batch in train_dataloader:
                # Training step
                metrics = self.train_step(batch)
                
                # Logging
                self.log_metrics(metrics, prefix="train")
                
                # Evaluation
                if (eval_dataloader and 
                    self.global_step % self.config.eval_interval == 0 and
                    self.global_step > 0):
                    eval_metrics = self.evaluate(eval_dataloader)
                    self.log_metrics(eval_metrics, prefix="eval")
                
                # Checkpointing
                if (self.config.save_checkpoints and 
                    self.global_step % self.config.save_interval == 0 and
                    self.global_step > 0):
                    checkpoint_path = os.path.join(
                        self.config.output_dir, 
                        f"checkpoint_step_{self.global_step}.pt"
                    )
                    self.save_checkpoint(checkpoint_path)
        
        # Final checkpoint
        if self.config.save_checkpoints:
            final_checkpoint_path = os.path.join(self.config.output_dir, "final_checkpoint.pt")
            self.save_checkpoint(final_checkpoint_path)
        
        logging.info("Training completed!")

def parse_args():
    """Parse command line arguments."""
    import argparse
    parser = argparse.ArgumentParser(description="Training utility for YunMin Mamba")
    parser.add_argument("--epochs", type=int, default=2, help="number of training epochs")
    parser.add_argument("--ia3", action="store_true", help="enable IA³ scaling modules")
    parser.add_argument("--output-dir", default="./training_outputs", help="directory to save outputs")
    parser.add_argument("--batch-size", type=int, default=8, help="batch size for training")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--vocab-size", type=int, default=1000, help="vocabulary size")
    parser.add_argument("--seq-length", type=int, default=64, help="sequence length")
    parser.add_argument("--disable-masking", action="store_true", help="disable learned masking (Pillar 2)")
    parser.add_argument("--disable-peft", action="store_true", help="disable PEFT (Pillar 3)")
    return parser.parse_args()

def main():
    """Main training entry point."""
    args = parse_args()
    
    # Configuration
    config = TrainingConfig(
        vocab_size=args.vocab_size,
        d_model=256,
        n_layers=4,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        max_seq_length=args.seq_length,
        learning_rate=args.learning_rate,
        enable_masking=not args.disable_masking,
        enable_peft=PEFT_AVAILABLE and not args.disable_peft,
        enable_ia3=args.ia3,
        log_interval=10,
        eval_interval=50,
        save_interval=100,
        output_dir=args.output_dir
    )
    
    # Create datasets
    train_dataset = SimpleDataset(
        vocab_size=config.vocab_size,
        seq_length=config.max_seq_length,
        num_samples=1000
    )
    
    eval_dataset = SimpleDataset(
        vocab_size=config.vocab_size,
        seq_length=config.max_seq_length,
        num_samples=200
    )
    
    # Initialize trainer
    trainer = AdaptiveMambaTrainer(config)
    
    # Start training
    trainer.train(train_dataset, eval_dataset)

if __name__ == "__main__":
    main()
