"""
Logging utilities for the hardware-data-parameter co-design project.
"""

import logging
import os
import sys
from datetime import datetime
from typing import Optional
import wandb


def setup_logger(
    name: str = "codesign",
    log_file: Optional[str] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Set up a logger with both console and file handlers.
    
    Args:
        name: Logger name
        log_file: Path to log file (optional)
        level: Logging level
    
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def setup_wandb(config: dict, project: str, run_name: str) -> None:
    """
    Initialize Weights & Biases logging.
    
    Args:
        config: Configuration dictionary to log
        project: W&B project name
        run_name: W&B run name
    """
    wandb.init(
        project=project,
        name=run_name,
        config=config,
        save_code=True
    )


def log_model_info(logger: logging.Logger, model, config: dict) -> None:
    """
    Log model architecture and parameter information.
    
    Args:
        logger: Logger instance
        model: PyTorch model
        config: Model configuration
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info("=" * 50)
    logger.info("MODEL INFORMATION")
    logger.info("=" * 50)
    logger.info(f"Model architecture: {model.__class__.__name__}")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Non-trainable parameters: {total_params - trainable_params:,}")
    
    # Log model configuration
    logger.info("Model configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    logger.info("=" * 50)


def log_training_info(logger: logging.Logger, config: dict) -> None:
    """
    Log training configuration and environment information.
    
    Args:
        logger: Logger instance
        config: Training configuration
    """
    logger.info("=" * 50)
    logger.info("TRAINING INFORMATION")
    logger.info("=" * 50)
    
    # Log training configuration
    logger.info("Training configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    
    # Log environment information
    logger.info(f"Python version: {sys.version}")
    try:
        import torch
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA device count: {torch.cuda.device_count()}")
            logger.info(f"CUDA device name: {torch.cuda.get_device_name()}")
    except ImportError:
        logger.warning("PyTorch not available")
    
    logger.info("=" * 50) 