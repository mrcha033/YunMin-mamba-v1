"""
Real Dataset Integration for Adaptive Hybrid-PEFT Mamba Research
Supports WikiText-2, CNN/DM, HotpotQA, and HumanEval datasets.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import datasets
from datasets import load_dataset
import numpy as np
from typing import List, Dict, Any, Optional, Union
import logging
from pathlib import Path
import json

class WikiText2Dataset(Dataset):
    """WikiText-2 dataset for language modeling evaluation."""
    
    def __init__(self, split: str = "train", max_length: int = 512, 
                 tokenizer_name: str = "gpt2", cache_dir: Optional[str] = None):
        self.max_length = max_length
        self.split = split
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load dataset
        logging.info(f"Loading WikiText-2 {split} split...")
        try:
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", 
                                 cache_dir=cache_dir, split=split)
        except Exception as e:
            logging.warning(f"Failed to load WikiText-2: {e}")
            # Fallback to local simple dataset
            dataset = self._create_fallback_dataset()
        
        # Process and tokenize texts
        self.examples = self._process_dataset(dataset)
        logging.info(f"Processed {len(self.examples)} examples from WikiText-2 {split}")
    
    def _create_fallback_dataset(self):
        """Create fallback dataset if WikiText-2 is not available."""
        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subset of artificial intelligence.",
            "Natural language processing enables computers to understand human language.",
            "Deep learning uses neural networks with multiple layers.",
            "Transformers revolutionized the field of natural language processing.",
        ] * 100  # Repeat to create more samples
        
        return [{"text": text} for text in texts]
    
    def _process_dataset(self, dataset):
        """Process and tokenize the dataset."""
        examples = []
        
        for item in dataset:
            if isinstance(item, dict) and "text" in item:
                text = item["text"].strip()
            else:
                text = str(item).strip()
            
            if len(text) < 10:  # Skip very short texts
                continue
            
            # Tokenize
            encoded = self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
            
            examples.append({
                "input_ids": encoded["input_ids"].squeeze(0),
                "attention_mask": encoded["attention_mask"].squeeze(0),
                "labels": encoded["input_ids"].squeeze(0).clone()
            })
        
        return examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]

class CNNDMDataset(Dataset):
    """CNN/DailyMail dataset for summarization evaluation."""
    
    def __init__(self, split: str = "train", max_source_length: int = 512,
                 max_target_length: int = 128, tokenizer_name: str = "t5-small",
                 cache_dir: Optional[str] = None, num_samples: Optional[int] = None):
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.split = split
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load dataset
        logging.info(f"Loading CNN/DM {split} split...")
        try:
            dataset = load_dataset("cnn_dailymail", "3.0.0", 
                                 cache_dir=cache_dir, split=split)
            if num_samples:
                dataset = dataset.select(range(min(num_samples, len(dataset))))
        except Exception as e:
            logging.warning(f"Failed to load CNN/DM: {e}")
            dataset = self._create_fallback_dataset()
        
        # Process dataset
        self.examples = self._process_dataset(dataset)
        logging.info(f"Processed {len(self.examples)} examples from CNN/DM {split}")
    
    def _create_fallback_dataset(self):
        """Create fallback dataset for summarization."""
        examples = []
        for i in range(100):
            article = f"This is a sample article number {i}. " * 20
            summary = f"Summary of article {i}."
            examples.append({"article": article, "highlights": summary})
        return examples
    
    def _process_dataset(self, dataset):
        """Process and tokenize the dataset."""
        examples = []
        
        for item in dataset:
            article = item.get("article", "").strip()
            summary = item.get("highlights", "").strip()
            
            if len(article) < 50 or len(summary) < 10:
                continue
            
            # Tokenize source
            source_encoded = self.tokenizer(
                f"summarize: {article}",
                max_length=self.max_source_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
            
            # Tokenize target
            target_encoded = self.tokenizer(
                summary,
                max_length=self.max_target_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
            
            examples.append({
                "input_ids": source_encoded["input_ids"].squeeze(0),
                "attention_mask": source_encoded["attention_mask"].squeeze(0),
                "labels": target_encoded["input_ids"].squeeze(0),
                "target_text": summary  # Keep for ROUGE calculation
            })
        
        return examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]

class HotpotQADataset(Dataset):
    """HotpotQA dataset for question answering evaluation."""
    
    def __init__(self, split: str = "train", max_length: int = 512,
                 tokenizer_name: str = "bert-base-uncased", 
                 cache_dir: Optional[str] = None, num_samples: Optional[int] = None):
        self.max_length = max_length
        self.split = split
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load dataset
        logging.info(f"Loading HotpotQA {split} split...")
        try:
            dataset = load_dataset("hotpot_qa", "distractor", 
                                 cache_dir=cache_dir, split=split)
            if num_samples:
                dataset = dataset.select(range(min(num_samples, len(dataset))))
        except Exception as e:
            logging.warning(f"Failed to load HotpotQA: {e}")
            dataset = self._create_fallback_dataset()
        
        # Process dataset
        self.examples = self._process_dataset(dataset)
        logging.info(f"Processed {len(self.examples)} examples from HotpotQA {split}")
    
    def _create_fallback_dataset(self):
        """Create fallback QA dataset."""
        examples = []
        for i in range(100):
            question = f"What is the answer to question {i}?"
            context = f"This is context {i} that contains the answer {i}."
            answer = f"answer {i}"
            examples.append({
                "question": question,
                "context": context,
                "answer": answer
            })
        return examples
    
    def _process_dataset(self, dataset):
        """Process and tokenize the dataset."""
        examples = []
        
        for item in dataset:
            question = item.get("question", "").strip()
            context = " ".join(item.get("context", {}).get("sentences", []))
            answer = item.get("answer", "").strip()
            
            if len(question) < 5 or len(context) < 10:
                continue
            
            # Create input text
            input_text = f"Question: {question} Context: {context}"
            
            # Tokenize
            encoded = self.tokenizer(
                input_text,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
            
            examples.append({
                "input_ids": encoded["input_ids"].squeeze(0),
                "attention_mask": encoded["attention_mask"].squeeze(0),
                "question": question,
                "context": context,
                "answer": answer
            })
        
        return examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]

class HumanEvalDataset(Dataset):
    """HumanEval dataset for code generation evaluation."""
    
    def __init__(self, split: str = "test", max_length: int = 512,
                 tokenizer_name: str = "microsoft/CodeBERT-base",
                 cache_dir: Optional[str] = None):
        self.max_length = max_length
        self.split = split
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load dataset
        logging.info(f"Loading HumanEval dataset...")
        try:
            dataset = load_dataset("openai_humaneval", cache_dir=cache_dir, split="test")
        except Exception as e:
            logging.warning(f"Failed to load HumanEval: {e}")
            dataset = self._create_fallback_dataset()
        
        # Process dataset
        self.examples = self._process_dataset(dataset)
        logging.info(f"Processed {len(self.examples)} examples from HumanEval")
    
    def _create_fallback_dataset(self):
        """Create fallback code generation dataset."""
        examples = []
        for i in range(50):
            prompt = f'def function_{i}(x):\n    """Return x + {i}"""\n'
            canonical_solution = f'    return x + {i}\n'
            test = f'assert function_{i}(5) == {5 + i}'
            examples.append({
                "prompt": prompt,
                "canonical_solution": canonical_solution,
                "test": test,
                "task_id": f"HumanEval/{i}"
            })
        return examples
    
    def _process_dataset(self, dataset):
        """Process and tokenize the dataset."""
        examples = []
        
        for item in dataset:
            prompt = item.get("prompt", "").strip()
            solution = item.get("canonical_solution", "").strip()
            test = item.get("test", "").strip()
            task_id = item.get("task_id", "")
            
            if len(prompt) < 10:
                continue
            
            # Tokenize prompt
            encoded = self.tokenizer(
                prompt,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
            
            examples.append({
                "input_ids": encoded["input_ids"].squeeze(0),
                "attention_mask": encoded["attention_mask"].squeeze(0),
                "prompt": prompt,
                "canonical_solution": solution,
                "test": test,
                "task_id": task_id
            })
        
        return examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]

class DatasetFactory:
    """Factory for creating datasets based on task type."""
    
    DATASET_CONFIGS = {
        "language_modeling": {
            "class": WikiText2Dataset,
            "splits": ["train", "validation", "test"],
            "default_params": {
                "max_length": 512,
                "tokenizer_name": "gpt2"
            }
        },
        "summarization": {
            "class": CNNDMDataset,
            "splits": ["train", "validation", "test"],
            "default_params": {
                "max_source_length": 512,
                "max_target_length": 128,
                "tokenizer_name": "t5-small"
            }
        },
        "question_answering": {
            "class": HotpotQADataset,
            "splits": ["train", "validation"],
            "default_params": {
                "max_length": 512,
                "tokenizer_name": "bert-base-uncased"
            }
        },
        "code_generation": {
            "class": HumanEvalDataset,
            "splits": ["test"],
            "default_params": {
                "max_length": 512,
                "tokenizer_name": "microsoft/CodeBERT-base"
            }
        }
    }
    
    @classmethod
    def create_dataset(cls, task: str, split: str = "train", 
                      num_samples: Optional[int] = None, **kwargs):
        """Create dataset for specific task and split."""
        
        if task not in cls.DATASET_CONFIGS:
            raise ValueError(f"Unsupported task: {task}. "
                           f"Available: {list(cls.DATASET_CONFIGS.keys())}")
        
        config = cls.DATASET_CONFIGS[task]
        
        if split not in config["splits"]:
            logging.warning(f"Split '{split}' not available for {task}. "
                          f"Available: {config['splits']}")
            split = config["splits"][0]  # Use first available split
        
        # Merge default parameters with user parameters
        params = config["default_params"].copy()
        params.update(kwargs)
        params["split"] = split
        
        if num_samples is not None and "num_samples" not in params:
            params["num_samples"] = num_samples
        
        # Create dataset
        dataset_class = config["class"]
        return dataset_class(**params)
    
    @classmethod
    def create_dataloader(cls, task: str, split: str = "train", 
                         batch_size: int = 8, num_samples: Optional[int] = None,
                         shuffle: bool = None, **kwargs):
        """Create DataLoader for specific task."""
        
        if shuffle is None:
            shuffle = split == "train"
        
        dataset = cls.create_dataset(task, split, num_samples, **kwargs)
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,  # Set to 0 to avoid multiprocessing issues
            pin_memory=torch.cuda.is_available()
        )

def test_datasets():
    """Test all dataset implementations."""
    logging.basicConfig(level=logging.INFO)
    
    tasks = ["language_modeling", "summarization", "question_answering", "code_generation"]
    
    for task in tasks:
        print(f"\n🧪 Testing {task} dataset...")
        try:
            # Create small test dataset
            dataset = DatasetFactory.create_dataset(task, split="train", num_samples=5)
            print(f"✅ {task}: {len(dataset)} samples loaded")
            
            # Test first item
            sample = dataset[0]
            print(f"   Sample keys: {list(sample.keys())}")
            print(f"   Input shape: {sample['input_ids'].shape}")
            
            # Create dataloader
            dataloader = DatasetFactory.create_dataloader(task, batch_size=2, num_samples=4)
            batch = next(iter(dataloader))
            print(f"   Batch input shape: {batch['input_ids'].shape}")
            
        except Exception as e:
            print(f"❌ {task}: Error - {e}")

if __name__ == "__main__":
    test_datasets() 