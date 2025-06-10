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
    from .model import AdaptiveMambaModel, AdaptiveMambaConfig
    from .layers.masked_linear import MaskedLinear, MaskedConv1d
except ImportError:
    from model import AdaptiveMambaModel, AdaptiveMambaConfig
    from layers.masked_linear import MaskedLinear, MaskedConv1d

# PEFT imports with fallback
try:
    from peft import LoraConfig, IA3Config, get_peft_model, TaskType, prepare_model_for_kbit_training
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
    enable_ia3: bool = True
    importance_threshold: float = 0.5  # Top X% of *tuned* layers get LoRA, rest get IA3.
    peft_application_ratio: float = 0.3  # Fraction of total eligible layers to adapt
    
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
    
    Core Philosophy: One-time, data-driven allocation of LoRA to high-importance layers and 
    IA³ to mid-importance layers based on learned importance patterns from Pillar 2.
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
        eligible_modules = (MaskedLinear, MaskedConv1d)

        for name, module in model.named_modules():
            if isinstance(module, eligible_modules):
                # ### REFACTOR ### Use the most stable importance metric from the math spec
                # 'logit_magnitude' is a good choice as it's continuous and reflects learned importance
                scores[name] = module.get_importance_score(method="logit_magnitude")
        
        # ### REFACTOR ### Fallback logic is simplified. If masking is disabled, PEFT allocation
        # would need a different strategy, but for this architecture, masking is the primary source.
        if not scores and self.config.enable_masking:
            logging.warning("No masked layers found to derive importance scores from. PEFT will not be applied.")
        elif not self.config.enable_masking:
             logging.warning("Masking is disabled. Importance-driven PEFT allocation is not possible.")

        self.importance_cache = scores.copy()
        logging.info(f"Collected importance scores from {len(scores)} eligible layers.")
        return scores

    def _allocate_peft_methods(self, importance_scores: Dict[str, float]) -> Tuple[List[str], List[str]]:
        """
        🎯 CORE SELF-OPTIMIZATION LOGIC 🎯
        
        Intelligently allocate tuning methods based on learned importance patterns:
        - High importance → LoRA (maximum expressiveness)
        - Medium importance → IA³ (parameter efficiency)  
        - Low importance → Frozen (computational savings)
        
        Returns:
            Tuple of (lora_target_modules, ia3_target_modules)
        """
        if not importance_scores:
            logging.warning("No importance scores available for PEFT allocation.")
            return [], []
            
        sorted_layers = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
        
        total_eligible_layers = len(sorted_layers)
        num_layers_to_tune = max(1, int(total_eligible_layers * self.config.peft_application_ratio))
        num_lora_layers = max(1, int(num_layers_to_tune * self.config.importance_threshold))
        num_ia3_layers = num_layers_to_tune - num_lora_layers if self.config.enable_ia3 else 0
        
        lora_targets = [name for name, _ in sorted_layers[:num_lora_layers]]
        ia3_targets = [name for name, _ in sorted_layers[num_lora_layers:num_layers_to_tune]]
        
        # ### REFACTOR ### Log the self-optimization decision process clearly
        logging.info("🧠 SELF-OPTIMIZING PEFT ALLOCATION SUMMARY:")
        logging.info(f"   Total eligible layers analyzed: {total_eligible_layers}")
        logging.info(f"   Layers to be tuned: {num_layers_to_tune} ({self.config.peft_application_ratio:.1%})")
        logging.info(f"   LoRA allocation (High-Importance): {len(lora_targets)} layers")
        logging.info(f"   IA³ allocation (Mid-Importance): {len(ia3_targets) if self.config.enable_ia3 else 0} layers")
        logging.info(f"   Frozen (Low-Importance): {total_eligible_layers - num_layers_to_tune} layers")
        
        self._log_detailed_allocation(sorted_layers, lora_targets, ia3_targets)
        
        return lora_targets, ia3_targets

    def _log_detailed_allocation(self, sorted_layers: List[Tuple[str, float]], 
                                lora_layers: List[str], ia3_layers: List[str]):
        """Log detailed allocation decisions for transparency and debugging."""
        logging.info("📊 DETAILED IMPORTANCE-DRIVEN ALLOCATION:")
        top_layers_to_show = min(15, len(sorted_layers))
        for name, score in sorted_layers[:top_layers_to_show]:
            method = "Frozen (Low-Importance)"
            if name in lora_layers:
                method = "LoRA (High-Importance)"
            elif name in ia3_layers:
                method = "IA³ (Mid-Importance)"
            
            display_name = name if len(name) <= 45 else name[:42] + "..."
            logging.info(f"   {display_name:<45} | Score: {score:.4f} | Method: {method}")
        if len(sorted_layers) > top_layers_to_show:
            logging.info(f"   ... and {len(sorted_layers) - top_layers_to_show} more layers.")

    def apply_peft_to_model(self, model: nn.Module) -> nn.Module:
        """
        🚀 APPLY SELF-OPTIMIZING PEFT ALLOCATION 🚀
        
        This is a one-time setup step that configures the model for efficient tuning.
        """
        if not PEFT_AVAILABLE or not self.config.enable_peft or self.peft_applied:
            logging.info("PEFT application skipped (disabled, unavailable, or already applied).")
            return model

        logging.info("🧠 INITIATING SELF-OPTIMIZING PEFT ALLOCATION...")
        
        # Freeze all parameters initially
        for param in model.parameters():
            param.requires_grad = False
        
        # Step 1: Extract comprehensive importance scores
        importance_scores = self._get_comprehensive_importance_scores(model)
        if not importance_scores:
            logging.error("❌ Failed to get importance scores. Cannot apply PEFT.")
            return model

        # Step 2: Allocate tuning methods based on scores
        lora_targets, ia3_targets = self._allocate_peft_methods(importance_scores)
        
        # Step 3: Apply IA³ to mid-importance layers
        if ia3_targets and self.config.enable_ia3:
            logging.info(f"[PEFT] Applying IA³ to {len(ia3_targets)} mid-importance layers...")
            try:
                insert_ia3_modules(model, target_module_names=ia3_targets)
                logging.info("[PEFT] IA³ application completed.")
            except Exception as e:
                logging.error(f"Failed to apply IA³: {e}", exc_info=True)
        
        # Step 4: Apply LoRA to high-importance layers
        if lora_targets:
            logging.info(f"[PEFT] Applying LoRA to {len(lora_targets)} high-importance layers...")
            try:
                # ### REFACTOR ### Use precise targeting with `target_modules=lora_targets`
                lora_config = LoraConfig(
                    r=self.config.peft_r,
                    lora_alpha=self.config.peft_alpha,
                    lora_dropout=self.config.peft_dropout,
                    target_modules=lora_targets,
                    bias="none",
                    task_type=TaskType.CAUSAL_LM,
                )
                model = get_peft_model(model, lora_config)
                logging.info("[PEFT] LoRA application completed.")
            except Exception as e:
                logging.error(f"Failed to apply LoRA: {e}", exc_info=True)
        
        # ### REFACTOR ### Log efficiency gains *after* all modifications
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logging.info("🎯 SELF-OPTIMIZATION RESULTS:")
        logging.info(f"   Total parameters:     {total_params:,}")
        logging.info(f"   Trainable parameters: {trainable_params:,}")
        if total_params > 0:
            efficiency_ratio = trainable_params / total_params
            logging.info(f"   Efficiency ratio:     {efficiency_ratio:.2%} (Trainable/Total)")
        
        if hasattr(model, "print_trainable_parameters"):
            model.print_trainable_parameters()

        self.peft_applied = True
        logging.info("✅ SELF-OPTIMIZING PEFT ALLOCATION COMPLETED SUCCESSFULLY.")
        
        return model
    
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
        self.data = torch.randint(0, vocab_size, (num_samples, seq_length))
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        sequence = self.data[idx]
        return {'input_ids': sequence[:-1], 'labels': sequence[1:]}

class AdaptiveMambaTrainer:
    """
    Complete trainer for Adaptive Mamba model.
    Orchestrates all three pillars and manages the training process.
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        self._setup_logging()
        
        # Initialize model
        self.model = self._create_model().to(self.device)
        
        # ### REFACTOR ### Apply PEFT as a one-time setup step BEFORE creating optimizer
        self.peft_manager = PEFTManager(config)
        self.model = self.peft_manager.apply_peft_to_model(self.model)
        
        # ### FIX ### Ensure model (including new PEFT parameters) is on correct device
        self.model = self.model.to(self.device)
        
        # Log PEFT allocation details to wandb AFTER PEFT application (only if we initialized wandb)
        if WANDB_AVAILABLE and not getattr(self, 'external_wandb', False):
            wandb.config.update({"peft_allocation": self.peft_manager.get_allocation_summary()})
        
        # Initialize optimizer and scheduler AFTER model is potentially modified by PEFT
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        self.global_step = 0
        self.current_epoch = 0
        
        logging.info(f"Trainer initialized. Device: {self.device}")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.config.output_dir, 'training.log')),
                logging.StreamHandler()
            ]
        )
        
        # ### FIX ### Only initialize wandb if there is no active run
        if WANDB_AVAILABLE:
            if wandb.run is None:
                # No active run, so this script is responsible for init
                self.external_wandb = False
                wandb.init(
                    project=self.config.project_name,
                    name=self.config.run_name,
                    config=asdict(self.config)
                )
                logging.info("Wandb initialized by trainer (standalone execution)")
            else:
                # An active run already exists (likely from research_ablation_study.py)
                self.external_wandb = True
                logging.info("Wandb is already initialized by an external script. Skipping init.")
    
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
            },
            'scan_update_frequency': self.config.scan_update_frequency,
        }
        
        return AdaptiveMambaModel(
            vocab_size=self.config.vocab_size,
            d_model=self.config.d_model,
            n_layers=self.config.n_layers,
            block_config=block_config
        )

    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer for only the trainable parameters."""
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        logging.info(f"Optimizer will manage {len(trainable_params)} trainable parameter tensors.")
        return optim.AdamW(trainable_params, lr=self.config.learning_rate)

    def _create_scheduler(self):
        """Create learning rate scheduler with warmup."""
        def lr_lambda(step):
            if step < self.config.warmup_steps:
                return float(step) / float(max(1, self.config.warmup_steps))
            return 1.0
        
        # ### BUG FIX ### Scheduler is now created with the correct, final optimizer
        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def _create_dataloader(self, dataset: Dataset, shuffle: bool = True) -> DataLoader:
        return DataLoader(
            dataset, batch_size=self.config.batch_size, shuffle=shuffle,
            num_workers=0, pin_memory=True if self.device.type == 'cuda' else False
        )
    
    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor, 
                    regularization_loss: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
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
        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        # ### REFACTOR ### PEFT logic is removed from the training step.
        update_scan = (self.global_step % self.config.scan_update_frequency) == 0
        logits = self.model(input_ids, update_scan=update_scan)
        
        reg_loss = self.model.get_total_regularization_loss()
        total_loss, loss_components = self.compute_loss(logits, labels, reg_loss)
        
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_((p for p in self.model.parameters() if p.requires_grad), self.config.gradient_clip)
        self.optimizer.step()
        self.scheduler.step()
        
        metrics = {
            'loss': loss_components,
            'learning_rate': self.scheduler.get_last_lr()[0],
            'step': self.global_step
        }
        
        if self.config.enable_masking:
            masking_stats = {}
            # Simplified stat collection
            all_sparsities = []
            for i, block in enumerate(self.model.blocks):
                block_stats = block.get_masking_statistics()
                for layer_name, stats in block_stats.items():
                     all_sparsities.append(stats.get('current_sparsity', 0))
            if all_sparsities:
                metrics['masking_avg_sparsity'] = np.mean(all_sparsities)

        self.global_step += 1
        return metrics
    
    def log_metrics(self, metrics: Dict[str, Any], prefix: str = "train"):
        """Log metrics to console and wandb."""
        log_str_parts = []
        if self.global_step % self.config.log_interval == 0 and prefix == "train":
            loss_info = metrics.get('loss', {})
            lr = metrics.get('learning_rate', 0.0)
            log_str_parts.append(f"Step {self.global_step}")
            log_str_parts.append(f"Loss: {loss_info.get('total', 0.0):.4f}")
            log_str_parts.append(f"(CE: {loss_info.get('cross_entropy', 0.0):.4f}, Reg: {loss_info.get('regularization', 0.0):.6f})")
            log_str_parts.append(f"LR: {lr:.6f}")
            if 'masking_avg_sparsity' in metrics:
                 log_str_parts.append(f"Sparsity: {metrics['masking_avg_sparsity']:.2%}")
            logging.info(" | ".join(log_str_parts))

        if WANDB_AVAILABLE:
            wandb_metrics = {}
            if 'loss' in metrics:
                for k, v in metrics['loss'].items():
                    wandb_metrics[f"{prefix}/loss_{k}"] = v
            
            # Add other top-level metrics
            for k, v in metrics.items():
                if k != 'loss':
                    wandb_metrics[f"{prefix}/{k}"] = v
            
            wandb.log(wandb_metrics, step=self.global_step)
    
    def evaluate(self, eval_dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate the model."""
        self.model.eval()
        total_loss = 0.0
        total_steps = 0
        with torch.no_grad():
            for step, batch in enumerate(eval_dataloader):
                if step >= self.config.eval_steps: break
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                logits = self.model(input_ids)
                reg_loss = self.model.get_total_regularization_loss()
                total_loss_step, _ = self.compute_loss(logits, labels, reg_loss)
                total_loss += total_loss_step.item()
                total_steps += 1
        return {'eval_loss': total_loss / total_steps if total_steps > 0 else 0.0}

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        # Use peft's save_pretrained for adapters
        if self.peft_manager.peft_applied:
            self.model.save_pretrained(os.path.dirname(path))
        else:
             torch.save(self.model.state_dict(), path)
        logging.info(f"Checkpoint saved to {os.path.dirname(path)}")
    
    def train(self, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None):
        """Main training loop."""
        train_dataloader = self._create_dataloader(train_dataset, shuffle=True)
        eval_dataloader = self._create_dataloader(eval_dataset, shuffle=False) if eval_dataset else None
        
        logging.info(f"Starting training for {self.config.num_epochs} epochs")
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            logging.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            
            for batch in train_dataloader:
                metrics = self.train_step(batch)
                self.log_metrics(metrics, prefix="train")
                
                if (eval_dataloader and self.global_step % self.config.eval_interval == 0 and self.global_step > 0):
                    eval_metrics = self.evaluate(eval_dataloader)
                    self.log_metrics(eval_metrics, prefix="eval")
                
                if (self.config.save_checkpoints and self.global_step % self.config.save_interval == 0 and self.global_step > 0):
                    checkpoint_path = os.path.join(self.config.output_dir, f"checkpoint_step_{self.global_step}")
                    self.save_checkpoint(checkpoint_path)
        
        final_checkpoint_path = os.path.join(self.config.output_dir, "final_model")
        self.save_checkpoint(final_checkpoint_path)
        logging.info("Training completed!")

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Training utility for Adaptive Hybrid-PEFT Mamba")
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
    args = parse_args()
    config = TrainingConfig(
        vocab_size=args.vocab_size, d_model=256, n_layers=4, batch_size=args.batch_size,
        num_epochs=args.epochs, max_seq_length=args.seq_length, learning_rate=args.learning_rate,
        enable_masking=not args.disable_masking,
        enable_peft=PEFT_AVAILABLE and not args.disable_peft,
        enable_ia3=args.ia3,
        log_interval=10, eval_interval=50, save_interval=200, output_dir=args.output_dir
    )
    train_dataset = SimpleDataset(config.vocab_size, config.max_seq_length, 1000)
    eval_dataset = SimpleDataset(config.vocab_size, config.max_seq_length, 200)
    trainer = AdaptiveMambaTrainer(config)
    trainer.train(train_dataset, eval_dataset)

if __name__ == "__main__":
    main()
