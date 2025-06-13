"""
GLUE benchmark dataloader for fine-tuning experiments.
"""

import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
from typing import Dict, List, Optional, Tuple


class GLUEDataset(Dataset):
    """GLUE dataset for fine-tuning."""
    
    def __init__(self, tokenizer, task_name: str, split: str = "train", max_length: int = 512):
        self.tokenizer = tokenizer
        self.task_name = task_name
        self.max_length = max_length
        
        # Load GLUE dataset
        self.dataset = load_dataset("glue", task_name, split=split)
        
        # Task-specific configurations
        self.task_configs = {
            "cola": {"text_cols": ["sentence"], "label_col": "label", "num_labels": 2},
            "sst2": {"text_cols": ["sentence"], "label_col": "label", "num_labels": 2},
            "mrpc": {"text_cols": ["sentence1", "sentence2"], "label_col": "label", "num_labels": 2},
            "qqp": {"text_cols": ["question1", "question2"], "label_col": "label", "num_labels": 2},
            "mnli": {"text_cols": ["premise", "hypothesis"], "label_col": "label", "num_labels": 3},
            "qnli": {"text_cols": ["question", "sentence"], "label_col": "label", "num_labels": 2},
            "rte": {"text_cols": ["sentence1", "sentence2"], "label_col": "label", "num_labels": 2},
            "wnli": {"text_cols": ["sentence1", "sentence2"], "label_col": "label", "num_labels": 2},
        }
        
        self.config = self.task_configs[task_name]
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        example = self.dataset[idx]
        
        # Extract text(s)
        text_cols = self.config["text_cols"]
        if len(text_cols) == 1:
            text = example[text_cols[0]]
            encoded = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt"
            )
        else:
            text1, text2 = example[text_cols[0]], example[text_cols[1]]
            encoded = self.tokenizer(
                text1,
                text2,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt"
            )
        
        # Extract label
        label = example[self.config["label_col"]]
        if label == -1:  # Handle invalid labels in some GLUE tasks
            label = 0
        
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }


def get_glue_dataloader(
    tokenizer,
    task_name: str,
    batch_size: int = 16,
    max_length: int = 512,
    split: str = "train",
    num_workers: int = 4
) -> DataLoader:
    """
    Create a DataLoader for GLUE benchmark tasks.
    
    Args:
        tokenizer: Tokenizer to use for encoding text
        task_name: GLUE task name (e.g., "cola", "sst2", etc.)
        batch_size: Batch size for the DataLoader
        max_length: Maximum sequence length
        split: Dataset split ("train", "validation", "test")
        num_workers: Number of worker processes for data loading
    
    Returns:
        DataLoader for the specified GLUE task
    """
    dataset = GLUEDataset(tokenizer, task_name, split, max_length)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True
    )


def get_glue_metrics(task_name: str) -> Dict[str, str]:
    """
    Get the appropriate metrics for each GLUE task.
    
    Args:
        task_name: GLUE task name
    
    Returns:
        Dictionary mapping metric names to their types
    """
    metrics = {
        "cola": {"matthews_correlation": "matthews_correlation"},
        "sst2": {"accuracy": "accuracy"},
        "mrpc": {"accuracy": "accuracy", "f1": "f1"},
        "qqp": {"accuracy": "accuracy", "f1": "f1"},
        "mnli": {"accuracy": "accuracy"},
        "qnli": {"accuracy": "accuracy"},
        "rte": {"accuracy": "accuracy"},
        "wnli": {"accuracy": "accuracy"},
    }
    
    return metrics.get(task_name, {"accuracy": "accuracy"}) 