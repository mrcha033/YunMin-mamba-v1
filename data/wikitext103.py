"""
WikiText-103 Dataset Loading and Processing

This module provides comprehensive data loading for WikiText-103 dataset
with efficient tokenization, batching, and preprocessing for large-scale training.

Features:
- Full WikiText-103 dataset download and processing
- Efficient tokenization with GPT-2 tokenizer
- Memory-efficient data loading with streaming
- Configurable sequence lengths and batching
- Support for distributed training
"""

import os
import sys
from typing import Dict, Optional, Union, Iterator, List
from pathlib import Path
import logging

import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from datasets import load_dataset, Dataset as HFDataset
from transformers import AutoTokenizer
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WikiText103Dataset(Dataset):
    """
    WikiText-103 dataset with efficient tokenization and sequence packing.
    
    Features:
    - Streaming data loading for memory efficiency
    - Configurable sequence lengths
    - Automatic tokenization with GPT-2 tokenizer
    - Support for causal language modeling
    """
    
    def __init__(
        self,
        split: str = "train",
        max_length: int = 1024,
        tokenizer_name: str = "gpt2",
        cache_dir: Optional[str] = None,
        streaming: bool = False,
        num_proc: int = 4,
        trust_remote_code: bool = False
    ):
        """
        Initialize WikiText-103 dataset.
        
        Args:
            split: Dataset split ('train', 'validation', 'test')
            max_length: Maximum sequence length
            tokenizer_name: Tokenizer to use
            cache_dir: Directory to cache processed data
            streaming: Whether to use streaming (memory efficient)
            num_proc: Number of processes for data processing
            trust_remote_code: Whether to trust remote code
        """
        self.split = split
        self.max_length = max_length
        self.streaming = streaming
        self.num_proc = num_proc
        
        # Initialize tokenizer
        logger.info(f"Loading tokenizer: {tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=trust_remote_code
        )
        
        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load dataset
        logger.info(f"Loading WikiText-103 {split} split...")
        self.dataset = load_dataset(
            "wikitext",
            "wikitext-103-v1",
            split=split,
            cache_dir=cache_dir,
            streaming=streaming,
            trust_remote_code=trust_remote_code
        )
        
        # Process dataset
        if not streaming:
            logger.info("Tokenizing dataset...")
            self.dataset = self.dataset.map(
                self._tokenize_function,
                batched=True,
                num_proc=num_proc,
                remove_columns=self.dataset.column_names,
                desc="Tokenizing"
            )
            
            # Pack sequences for efficiency
            logger.info("Packing sequences...")
            self.dataset = self._pack_sequences(self.dataset)
        else:
            # For streaming, apply tokenization on-the-fly
            self.dataset = self.dataset.map(
                self._tokenize_function,
                batched=True,
                remove_columns=self.dataset.column_names
            )
    
    def _tokenize_function(self, examples: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
        """Tokenize text examples."""
        # Filter out empty texts
        texts = [text for text in examples["text"] if text.strip()]
        
        if not texts:
            return {"input_ids": []}
        
        # Tokenize
        tokenized = self.tokenizer(
            texts,
            truncation=False,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False
        )
        
        return tokenized
    
    def _pack_sequences(self, dataset: HFDataset) -> HFDataset:
        """
        Pack tokenized sequences into fixed-length chunks for efficient training.
        
        This concatenates all sequences and splits them into chunks of max_length,
        which is more efficient than padding individual sequences.
        """
        logger.info("Concatenating all sequences...")
        
        # Concatenate all input_ids
        all_input_ids = []
        for example in dataset:
            if example["input_ids"]:
                all_input_ids.extend(example["input_ids"])
                all_input_ids.append(self.tokenizer.eos_token_id)  # Add separator
        
        logger.info(f"Total tokens: {len(all_input_ids):,}")
        
        # Split into chunks
        chunks = []
        for i in range(0, len(all_input_ids) - self.max_length, self.max_length):
            chunk = all_input_ids[i:i + self.max_length]
            if len(chunk) == self.max_length:
                chunks.append({"input_ids": chunk})
        
        logger.info(f"Created {len(chunks):,} sequences of length {self.max_length}")
        
        # Convert to HuggingFace dataset
        return HFDataset.from_list(chunks)
    
    def __len__(self) -> int:
        """Return dataset length."""
        if self.streaming:
            # For streaming datasets, we can't know the exact length
            # Return a large number for training loop
            return 1_000_000
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single example."""
        if self.streaming:
            # For streaming, we need to handle differently
            try:
                example = next(iter(self.dataset.skip(idx).take(1)))
            except StopIteration:
                # If we've exhausted the stream, restart
                example = next(iter(self.dataset.take(1)))
        else:
            example = self.dataset[idx]
        
        # Convert to tensors
        input_ids = torch.tensor(example["input_ids"], dtype=torch.long)
        
        # For causal language modeling, labels are the same as input_ids
        # but shifted by one position
        return {
            "input_ids": input_ids,
            "labels": input_ids.clone(),
            "attention_mask": torch.ones_like(input_ids)
        }


def get_wikitext103_dataloader(
    split: str = "train",
    batch_size: int = 8,
    max_length: int = 1024,
    tokenizer_name: str = "gpt2",
    cache_dir: Optional[str] = None,
    streaming: bool = False,
    num_workers: int = 4,
    pin_memory: bool = True,
    distributed: bool = False,
    seed: int = 42,
    **kwargs
) -> DataLoader:
    """
    Create a DataLoader for WikiText-103 dataset.
    
    Args:
        split: Dataset split ('train', 'validation', 'test')
        batch_size: Batch size
        max_length: Maximum sequence length
        tokenizer_name: Tokenizer to use
        cache_dir: Directory to cache processed data
        streaming: Whether to use streaming
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory
        distributed: Whether to use distributed sampling
        seed: Random seed for reproducibility
        **kwargs: Additional arguments for DataLoader
        
    Returns:
        DataLoader for WikiText-103
    """
    logger.info(f"Creating WikiText-103 DataLoader for {split} split")
    
    # Create dataset
    dataset = WikiText103Dataset(
        split=split,
        max_length=max_length,
        tokenizer_name=tokenizer_name,
        cache_dir=cache_dir,
        streaming=streaming
    )
    
    # Create sampler for distributed training
    sampler = None
    shuffle = split == "train" and not distributed
    
    if distributed and not streaming:
        sampler = DistributedSampler(
            dataset,
            shuffle=split == "train",
            seed=seed
        )
        shuffle = False
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=split == "train",  # Drop last batch for training
        collate_fn=collate_fn,
        **kwargs
    )
    
    logger.info(f"Created DataLoader with {len(dataset):,} examples")
    return dataloader


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for batching.
    
    Args:
        batch: List of examples
        
    Returns:
        Batched tensors
    """
    # Stack tensors
    input_ids = torch.stack([item["input_ids"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask
    }


def calculate_dataset_stats(split: str = "train", max_samples: int = 10000) -> Dict[str, float]:
    """
    Calculate statistics for WikiText-103 dataset.
    
    Args:
        split: Dataset split to analyze
        max_samples: Maximum number of samples to analyze
        
    Returns:
        Dictionary with dataset statistics
    """
    logger.info(f"Calculating statistics for WikiText-103 {split} split")
    
    # Load dataset
    dataset = load_dataset("wikitext", "wikitext-103-v1", split=split)
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Calculate statistics
    total_chars = 0
    total_tokens = 0
    num_samples = 0
    
    for i, example in enumerate(dataset):
        if i >= max_samples:
            break
            
        text = example["text"].strip()
        if text:
            total_chars += len(text)
            tokens = tokenizer.encode(text)
            total_tokens += len(tokens)
            num_samples += 1
    
    # Calculate averages
    avg_chars_per_sample = total_chars / num_samples if num_samples > 0 else 0
    avg_tokens_per_sample = total_tokens / num_samples if num_samples > 0 else 0
    
    stats = {
        "num_samples_analyzed": num_samples,
        "total_characters": total_chars,
        "total_tokens": total_tokens,
        "avg_chars_per_sample": avg_chars_per_sample,
        "avg_tokens_per_sample": avg_tokens_per_sample,
        "estimated_total_samples": len(dataset)
    }
    
    logger.info(f"Dataset statistics: {stats}")
    return stats


def verify_dataset_loading():
    """Verify that dataset loading works correctly."""
    logger.info("Verifying WikiText-103 dataset loading...")
    
    try:
        # Test small batch
        dataloader = get_wikitext103_dataloader(
            split="validation",
            batch_size=2,
            max_length=512,
            streaming=False
        )
        
        # Get first batch
        batch = next(iter(dataloader))
        
        logger.info(f"Batch keys: {batch.keys()}")
        logger.info(f"Input IDs shape: {batch['input_ids'].shape}")
        logger.info(f"Labels shape: {batch['labels'].shape}")
        logger.info(f"Attention mask shape: {batch['attention_mask'].shape}")
        
        # Verify shapes match
        assert batch['input_ids'].shape == batch['labels'].shape
        assert batch['input_ids'].shape == batch['attention_mask'].shape
        
        logger.info("✅ Dataset loading verification successful!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Dataset loading verification failed: {e}")
        return False


if __name__ == "__main__":
    # Verify dataset loading
    verify_dataset_loading()
    
    # Calculate and display dataset statistics
    stats = calculate_dataset_stats()
    print("\nWikiText-103 Dataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:,}" if isinstance(value, (int, float)) else f"  {key}: {value}") 