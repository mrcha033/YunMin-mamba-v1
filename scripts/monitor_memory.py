#!/usr/bin/env python3
"""
Real-time GPU memory monitoring during experiments.
Automatically adjusts batch sizes if OOM detected.
"""

import torch
import psutil
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

class MemoryMonitor:
    def __init__(self, device: str = "cuda", log_file: Optional[str] = None):
        self.device = device
        self.log_file = log_file
        self.memory_history = []
        
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory statistics."""
        stats = {
            'timestamp': time.time(),
            'gpu_allocated_mb': 0,
            'gpu_reserved_mb': 0,
            'gpu_free_mb': 0,
            'cpu_percent': psutil.cpu_percent(),
            'ram_percent': psutil.virtual_memory().percent
        }
        
        if torch.cuda.is_available():
            stats.update({
                'gpu_allocated_mb': torch.cuda.memory_allocated(self.device) / 1024**2,
                'gpu_reserved_mb': torch.cuda.memory_reserved(self.device) / 1024**2,
                'gpu_free_mb': (torch.cuda.get_device_properties(self.device).total_memory - 
                               torch.cuda.memory_reserved(self.device)) / 1024**2
            })
        
        return stats
    
    def suggest_batch_size(self, current_batch_size: int, target_memory_percent: float = 80.0) -> int:
        """Suggest optimal batch size based on current memory usage."""
        if not torch.cuda.is_available():
            return current_batch_size
            
        total_memory = torch.cuda.get_device_properties(self.device).total_memory / 1024**2
        current_usage = torch.cuda.memory_allocated(self.device) / 1024**2
        current_percent = (current_usage / total_memory) * 100
        
        if current_percent > target_memory_percent:
            # Reduce batch size
            reduction_factor = target_memory_percent / current_percent
            new_batch_size = max(1, int(current_batch_size * reduction_factor))
            print(f"🔽 Memory usage {current_percent:.1f}% > {target_memory_percent}%")
            print(f"   Suggesting batch size reduction: {current_batch_size} → {new_batch_size}")
            return new_batch_size
        elif current_percent < target_memory_percent * 0.6:
            # Can increase batch size
            increase_factor = (target_memory_percent * 0.8) / current_percent
            new_batch_size = int(current_batch_size * increase_factor)
            print(f"🔼 Memory usage {current_percent:.1f}% < {target_memory_percent*0.6}%")
            print(f"   Suggesting batch size increase: {current_batch_size} → {new_batch_size}")
            return new_batch_size
        
        return current_batch_size
    
    def log_memory_usage(self):
        """Log current memory usage."""
        stats = self.get_memory_stats()
        self.memory_history.append(stats)
        
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(stats) + '\n')
        
        # Print summary
        if torch.cuda.is_available():
            print(f"GPU: {stats['gpu_allocated_mb']:.0f}MB allocated, "
                  f"{stats['gpu_free_mb']:.0f}MB free | "
                  f"CPU: {stats['cpu_percent']:.1f}% | "
                  f"RAM: {stats['ram_percent']:.1f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor GPU memory usage")
    parser.add_argument("--device", default="cuda", help="Device to monitor")
    parser.add_argument("--interval", type=int, default=5, help="Monitoring interval in seconds")
    parser.add_argument("--log_file", help="Log file for memory stats")
    args = parser.parse_args()
    
    monitor = MemoryMonitor(args.device, args.log_file)
    
    print(f"🔍 Monitoring memory usage every {args.interval} seconds...")
    print("Press Ctrl+C to stop")
    
    try:
        while True:
            monitor.log_memory_usage()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n✅ Memory monitoring stopped") 