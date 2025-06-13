"""
WikiText-103 dataloader for pre-training the baseline SSM model.
"""

import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
from typing import Dict, List, Optional


class WikiText103Dataset(Dataset):
    """WikiText-103 dataset for language modeling."""
    
    def __init__(self, tokenizer, max_length: int = 1024, split: str = "train"):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load WikiText-103 dataset
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
        
        # Tokenize all texts
        self.examples = []
        for example in dataset:
            if example["text"].strip():  # Skip empty lines
                tokens = self.tokenizer(
                    example["text"],
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt"
                )
                if tokens["input_ids"].size(1) > 10:  # Skip very short sequences
                    self.examples.append(tokens["input_ids"].squeeze(0))
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        input_ids = self.examples[idx]
        # For language modeling, labels are the same as input_ids
        return {
            "input_ids": input_ids,
            "labels": input_ids.clone()
        }


def get_wiktext103_dataloader(
    tokenizer,
    batch_size: int = 32,
    max_length: int = 1024,
    split: str = "train",
    num_workers: int = 4
) -> DataLoader:
    """
    Create a DataLoader for WikiText-103 dataset.
    
    Args:
        tokenizer: Tokenizer to use for encoding text
        batch_size: Batch size for the DataLoader
        max_length: Maximum sequence length
        split: Dataset split ("train", "validation", "test")
        num_workers: Number of worker processes for data loading
    
    Returns:
        DataLoader for WikiText-103 dataset
    """
    dataset = WikiText103Dataset(tokenizer, max_length, split)
    
    def collate_fn(batch):
        # Pad sequences to the same length within the batch
        input_ids = [item["input_ids"] for item in batch]
        labels = [item["labels"] for item in batch]
        
        # Pad to max length in batch
        max_len = max(len(seq) for seq in input_ids)
        
        padded_input_ids = []
        padded_labels = []
        
        for seq, label in zip(input_ids, labels):
            pad_length = max_len - len(seq)
            padded_seq = torch.nn.functional.pad(seq, (0, pad_length), value=tokenizer.pad_token_id)
            padded_label = torch.nn.functional.pad(label, (0, pad_length), value=-100)
            
            padded_input_ids.append(padded_seq)
            padded_labels.append(padded_label)
        
        return {
            "input_ids": torch.stack(padded_input_ids),
            "labels": torch.stack(padded_labels)
        }
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    ) 