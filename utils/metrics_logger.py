"""
Comprehensive Metrics Logger for Hardware-Data-Parameter Co-Design Framework

Tracks and logs all essential training metrics:
- Training Loss
- Validation Loss / Perplexity  
- Learning Rate
- Elapsed Time
- GPU Memory Usage
"""

import time
import torch
import wandb
import json
import csv
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass, asdict
import logging


@dataclass
class MetricsSnapshot:
    """Single snapshot of training metrics."""
    # Timing
    step: int
    epoch: int
    elapsed_time: float
    step_time: float
    
    # Training metrics
    train_loss: float
    learning_rate: float
    
    # Validation metrics (optional)
    val_loss: Optional[float] = None
    val_perplexity: Optional[float] = None
    
    # System metrics
    gpu_memory_used: Optional[float] = None
    gpu_memory_total: Optional[float] = None
    gpu_utilization: Optional[float] = None
    
    # Additional metrics
    gradient_norm: Optional[float] = None
    sparsity_loss: Optional[float] = None
    
    # Metadata
    timestamp: Optional[float] = None


class ComprehensiveMetricsLogger:
    """
    Comprehensive metrics logger that ensures all essential metrics
    are recorded at every step and epoch.
    """
    
    def __init__(
        self,
        output_dir: Path,
        experiment_name: str,
        use_wandb: bool = True,
        log_interval: int = 100,
        val_interval: int = 1000
    ):
        """Initialize the metrics logger."""
        self.output_dir = Path(output_dir)
        self.experiment_name = experiment_name
        self.use_wandb = use_wandb
        self.log_interval = log_interval
        self.val_interval = val_interval
        
        # Create output directories
        self.metrics_dir = self.output_dir / "metrics"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize storage
        self.metrics_history: List[MetricsSnapshot] = []
        self.epoch_metrics: Dict[int, Dict[str, Any]] = {}
        
        # Timing
        self.start_time = time.time()
        self.last_step_time = time.time()
        
        # CSV logging
        self.csv_path = self.metrics_dir / "metrics.csv"
        self.csv_headers_written = False
        
        # Setup logger
        self.logger = logging.getLogger(f"metrics_{experiment_name}")
        
        # GPU availability check
        self.gpu_available = torch.cuda.is_available()
        
        self.logger.info(f"📊 Metrics logger initialized")
        self.logger.info(f"📁 Metrics dir: {self.metrics_dir}")
        self.logger.info(f"💾 GPU available: {self.gpu_available}")
    
    def log_step_metrics(
        self,
        step: int,
        epoch: int,
        train_loss: float,
        learning_rate: float,
        optimizer: torch.optim.Optimizer = None,
        model: torch.nn.Module = None,
        **kwargs
    ) -> MetricsSnapshot:
        """
        Log metrics for a single training step.
        
        Args:
            step: Current training step
            epoch: Current epoch  
            train_loss: Training loss value
            learning_rate: Current learning rate
            optimizer: Optimizer (for gradient norm)
            model: Model (for additional metrics)
            **kwargs: Additional metrics
        """
        current_time = time.time()
        
        # Calculate timing
        elapsed_time = current_time - self.start_time
        step_time = max(current_time - self.last_step_time, 1e-6)  # Minimum 1 microsecond
        self.last_step_time = current_time
        
        # Get GPU metrics
        gpu_memory_used, gpu_memory_total, gpu_utilization = self._get_gpu_metrics()
        
        # Get gradient norm
        gradient_norm = self._get_gradient_norm(optimizer) if optimizer else None
        
        # Create metrics snapshot
        snapshot = MetricsSnapshot(
            step=step,
            epoch=epoch,
            elapsed_time=elapsed_time,
            step_time=step_time,
            train_loss=train_loss,
            learning_rate=learning_rate,
            gpu_memory_used=gpu_memory_used,
            gpu_memory_total=gpu_memory_total,
            gpu_utilization=gpu_utilization,
            gradient_norm=gradient_norm,
            timestamp=current_time,
            **kwargs
        )
        
        # Store snapshot
        self.metrics_history.append(snapshot)
        
        # Log to console (every log_interval steps)
        if step % self.log_interval == 0:
            self._log_to_console(snapshot)
        
        # Log to W&B
        if self.use_wandb:
            self._log_to_wandb(snapshot)
        
        # Log to CSV
        self._log_to_csv(snapshot)
        
        # Log to JSON (detailed)
        if step % self.log_interval == 0:
            self._save_detailed_metrics()
        
        return snapshot
    
    def log_validation_metrics(
        self,
        step: int,
        epoch: int,
        val_loss: float,
        val_perplexity: float = None,
        **kwargs
    ):
        """
        Log validation metrics.
        
        Args:
            step: Current training step
            epoch: Current epoch
            val_loss: Validation loss
            val_perplexity: Validation perplexity
            **kwargs: Additional validation metrics
        """
        # Update the latest snapshot with validation metrics
        if self.metrics_history:
            latest = self.metrics_history[-1]
            latest.val_loss = val_loss
            latest.val_perplexity = val_perplexity or self._calculate_perplexity(val_loss)
        
        # Log validation to console
        self.logger.info(f"Step {step:6d} | Val Loss: {val_loss:.4f} | Val PPL: {latest.val_perplexity:.2f}")
        
        # Log validation to W&B
        if self.use_wandb:
            wandb.log({
                "val_loss": val_loss,
                "val_perplexity": latest.val_perplexity,
                "step": step
            })
    
    def log_epoch_summary(self, epoch: int):
        """
        Log epoch summary metrics.
        
        Args:
            epoch: Completed epoch number
        """
        # Get all metrics for this epoch
        epoch_metrics = [m for m in self.metrics_history if m.epoch == epoch]
        
        if not epoch_metrics:
            return
        
        # Calculate epoch averages
        avg_train_loss = sum(m.train_loss for m in epoch_metrics) / len(epoch_metrics)
        avg_step_time = sum(m.step_time for m in epoch_metrics) / len(epoch_metrics)
        total_epoch_time = sum(m.step_time for m in epoch_metrics)
        
        # Get final learning rate for this epoch
        final_lr = epoch_metrics[-1].learning_rate
        
        # GPU metrics (average and peak)
        gpu_metrics_valid = [m for m in epoch_metrics if m.gpu_memory_used is not None]
        avg_gpu_memory = sum(m.gpu_memory_used for m in gpu_metrics_valid) / len(gpu_metrics_valid) if gpu_metrics_valid else 0
        peak_gpu_memory = max(m.gpu_memory_used for m in gpu_metrics_valid) if gpu_metrics_valid else 0
        
        # Validation metrics (latest)
        latest_val_loss = None
        latest_val_ppl = None
        for m in reversed(epoch_metrics):
            if m.val_loss is not None:
                latest_val_loss = m.val_loss
                latest_val_ppl = m.val_perplexity
                break
        
        # Store epoch summary
        epoch_summary = {
            'epoch': epoch,
            'avg_train_loss': avg_train_loss,
            'final_learning_rate': final_lr,
            'avg_step_time': avg_step_time,
            'total_epoch_time': total_epoch_time,
            'steps_in_epoch': len(epoch_metrics),
            'avg_gpu_memory_gb': avg_gpu_memory,
            'peak_gpu_memory_gb': peak_gpu_memory,
            'val_loss': latest_val_loss,
            'val_perplexity': latest_val_ppl,
            'timestamp': time.time()
        }
        
        self.epoch_metrics[epoch] = epoch_summary
        
        # Log epoch summary to console
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"EPOCH {epoch} SUMMARY")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Avg Train Loss: {avg_train_loss:.4f}")
        self.logger.info(f"Learning Rate: {final_lr:.2e}")
        self.logger.info(f"Avg Step Time: {avg_step_time:.3f}s")
        self.logger.info(f"Total Time: {total_epoch_time:.1f}s")
        self.logger.info(f"Steps: {len(epoch_metrics)}")
        if self.gpu_available:
            self.logger.info(f"Avg GPU Memory: {avg_gpu_memory:.1f}GB")
            self.logger.info(f"Peak GPU Memory: {peak_gpu_memory:.1f}GB")
        if latest_val_loss:
            self.logger.info(f"Val Loss: {latest_val_loss:.4f}")
            self.logger.info(f"Val Perplexity: {latest_val_ppl:.2f}")
        self.logger.info(f"{'='*60}\n")
        
        # Log epoch summary to W&B
        if self.use_wandb:
            wandb.log({
                "epoch_summary/avg_train_loss": avg_train_loss,
                "epoch_summary/learning_rate": final_lr,
                "epoch_summary/avg_step_time": avg_step_time,
                "epoch_summary/total_time": total_epoch_time,
                "epoch_summary/avg_gpu_memory": avg_gpu_memory,
                "epoch_summary/peak_gpu_memory": peak_gpu_memory,
                "epoch": epoch
            })
        
        # Save epoch summary to file
        epoch_file = self.metrics_dir / f"epoch_{epoch}_summary.json"
        with open(epoch_file, 'w') as f:
            json.dump(epoch_summary, f, indent=2)
    
    def _get_gpu_metrics(self) -> tuple:
        """Get current GPU metrics."""
        if not self.gpu_available:
            return None, None, None
        
        try:
            # Memory usage
            memory_used = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            
            # GPU utilization (simplified)
            gpu_utilization = torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else None
            
            return memory_used, memory_total, gpu_utilization
        except Exception as e:
            self.logger.warning(f"Failed to get GPU metrics: {e}")
            return None, None, None
    
    def _get_gradient_norm(self, optimizer: torch.optim.Optimizer) -> Optional[float]:
        """Calculate gradient norm."""
        if not optimizer:
            return None
        
        try:
            total_norm = 0.0
            for group in optimizer.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            return total_norm
        except Exception as e:
            self.logger.warning(f"Failed to calculate gradient norm: {e}")
            return None
    
    def _calculate_perplexity(self, loss: float) -> float:
        """Calculate perplexity from loss."""
        return torch.exp(torch.tensor(loss)).item()
    
    def _log_to_console(self, snapshot: MetricsSnapshot):
        """Log snapshot to console."""
        gpu_info = f" | GPU: {snapshot.gpu_memory_used:.1f}GB" if snapshot.gpu_memory_used else ""
        grad_info = f" | Grad: {snapshot.gradient_norm:.3f}" if snapshot.gradient_norm else ""
        val_info = f" | Val: {snapshot.val_loss:.4f}" if snapshot.val_loss else ""
        
        self.logger.info(
            f"Step {snapshot.step:6d} | Epoch {snapshot.epoch:2d} | "
            f"Loss: {snapshot.train_loss:.4f} | LR: {snapshot.learning_rate:.2e} | "
            f"Time: {snapshot.step_time:.3f}s{gpu_info}{grad_info}{val_info}"
        )
    
    def _log_to_wandb(self, snapshot: MetricsSnapshot):
        """Log snapshot to Weights & Biases."""
        if not self.use_wandb:
            return
        
        log_dict = {
            "train_loss": snapshot.train_loss,
            "learning_rate": snapshot.learning_rate,
            "step_time": snapshot.step_time,
            "elapsed_time": snapshot.elapsed_time,
            "step": snapshot.step,
            "epoch": snapshot.epoch
        }
        
        # Add optional metrics
        if snapshot.gpu_memory_used is not None:
            log_dict["gpu_memory_used"] = snapshot.gpu_memory_used
            log_dict["gpu_memory_total"] = snapshot.gpu_memory_total
        
        if snapshot.gradient_norm is not None:
            log_dict["gradient_norm"] = snapshot.gradient_norm
        
        if snapshot.val_loss is not None:
            log_dict["val_loss"] = snapshot.val_loss
            log_dict["val_perplexity"] = snapshot.val_perplexity
        
        if snapshot.sparsity_loss is not None:
            log_dict["sparsity_loss"] = snapshot.sparsity_loss
        
        wandb.log(log_dict)
    
    def _log_to_csv(self, snapshot: MetricsSnapshot):
        """Log snapshot to CSV file."""
        # Write headers if first time
        if not self.csv_headers_written:
            headers = list(asdict(snapshot).keys())
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
            self.csv_headers_written = True
        
        # Write data
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(list(asdict(snapshot).values()))
    
    def _save_detailed_metrics(self):
        """Save detailed metrics to JSON."""
        metrics_file = self.metrics_dir / "detailed_metrics.json"
        
        # Convert snapshots to dict format
        detailed_data = {
            'experiment_name': self.experiment_name,
            'start_time': self.start_time,
            'metrics_history': [asdict(m) for m in self.metrics_history],
            'epoch_summaries': self.epoch_metrics,
            'total_steps': len(self.metrics_history),
            'gpu_available': self.gpu_available
        }
        
        with open(metrics_file, 'w') as f:
            json.dump(detailed_data, f, indent=2)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary."""
        if not self.metrics_history:
            return {}
        
        latest = self.metrics_history[-1]
        
        # Calculate averages
        avg_loss = sum(m.train_loss for m in self.metrics_history) / len(self.metrics_history)
        avg_step_time = sum(m.step_time for m in self.metrics_history) / len(self.metrics_history)
        
        # GPU metrics
        gpu_metrics = [m for m in self.metrics_history if m.gpu_memory_used is not None]
        peak_gpu = max(m.gpu_memory_used for m in gpu_metrics) if gpu_metrics else 0
        
        return {
            'total_steps': len(self.metrics_history),
            'total_epochs': latest.epoch + 1,
            'total_time': latest.elapsed_time,
            'avg_loss': avg_loss,
            'final_loss': latest.train_loss,
            'avg_step_time': avg_step_time,
            'final_learning_rate': latest.learning_rate,
            'peak_gpu_memory': peak_gpu,
            'final_val_loss': latest.val_loss,
            'final_val_perplexity': latest.val_perplexity
        }
    
    def finalize(self):
        """Finalize logging and save final summary."""
        self.logger.info("📊 Finalizing metrics logging...")
        
        # Save final detailed metrics
        self._save_detailed_metrics()
        
        # Save final summary
        summary = self.get_metrics_summary()
        summary_file = self.metrics_dir / "final_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"📊 Metrics logging completed")
        self.logger.info(f"📁 Files saved in: {self.metrics_dir}")
        self.logger.info(f"📋 Total steps logged: {len(self.metrics_history)}")


def create_metrics_logger(
    output_dir: Path,
    experiment_name: str,
    config: Dict[str, Any]
) -> ComprehensiveMetricsLogger:
    """
    Factory function to create a properly configured metrics logger.
    
    Args:
        output_dir: Output directory for metrics
        experiment_name: Name of the experiment
        config: Configuration dictionary
        
    Returns:
        Configured ComprehensiveMetricsLogger instance
    """
    use_wandb = config.get('logging', {}).get('use_wandb', True)
    log_interval = config.get('training', {}).get('pretrain', {}).get('log_interval', 100)
    val_interval = config.get('training', {}).get('pretrain', {}).get('eval_interval', 1000)
    
    return ComprehensiveMetricsLogger(
        output_dir=output_dir,
        experiment_name=experiment_name,
        use_wandb=use_wandb,
        log_interval=log_interval,
        val_interval=val_interval
    ) 