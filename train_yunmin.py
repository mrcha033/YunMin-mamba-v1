import os
import torch
import wandb
import argparse
import numpy as np
import time
import psutil
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer, TrainingArguments,
    DataCollatorForLanguageModeling,
    default_data_collator,
    set_seed,
    TrainerCallback
)
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType
from scan_patch import apply_scan_patch, is_scan_patched
from ia3_layers import insert_ia3_modules

# -------------------------
# Memory & Performance Monitoring
# -------------------------
class PerformanceCallback(TrainerCallback):
    def __init__(self):
        self.epoch_start_time = None
        self.total_params = None
        self.trainable_params = None
        
    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            
    def on_epoch_end(self, args, state, control, **kwargs):
        epoch_time = time.time() - self.epoch_start_time
        
        # GPU Memory
        gpu_memory = 0
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
            
        # CPU Memory
        cpu_memory = psutil.Process().memory_info().rss / 1024**3  # GB
        
        wandb.log({
            "epoch_time": epoch_time,
            "gpu_peak_memory_gb": gpu_memory,
            "cpu_memory_gb": cpu_memory,
            "trainable_params": self.trainable_params,
            "total_params": self.total_params,
            "epoch": state.epoch
        })
        
        print(f"Epoch {state.epoch}: {epoch_time:.2f}s, GPU: {gpu_memory:.2f}GB, CPU: {cpu_memory:.2f}GB")

# -------------------------
# Argument Parsing
# -------------------------
parser = argparse.ArgumentParser(description="YunMin Correlation Scan Experiments")
parser.add_argument("--mode", type=str, required=True, 
                   choices=["baseline", "lora", "scan", "hybrid", "ia3", "ia3_lora"],
                   help="Training mode")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
parser.add_argument("--batch_size", type=int, default=4, help="Per-device batch size")
parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
parser.add_argument("--max_length", type=int, default=256, help="Max sequence length")
args = parser.parse_args()

set_seed(args.seed)
print(f"🚀 Starting YunMin experiment: {args.mode} mode with seed {args.seed}")

# -------------------------
# Config
# -------------------------
model_id = "state-spaces/mamba-130m"
run_name = f"yunmin_{args.mode}_seed{args.seed}"

print(f"📊 Experiment Configuration:")
print(f"   Model: {model_id}")
print(f"   Mode: {args.mode}")
print(f"   Epochs: {args.epochs}")
print(f"   Batch Size: {args.batch_size}")
print(f"   Learning Rate: {args.lr}")
print(f"   Max Length: {args.max_length}")

# -------------------------
# Load Dataset
# -------------------------
print("📚 Loading WikiText-2 dataset...")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
train_data = dataset["train"]
eval_data = dataset["validation"]

# Mamba uses GPTNeoX tokenizer
print("🔤 Setting up tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
tokenizer.pad_token = tokenizer.eos_token

def tokenize_and_filter(examples):
    """Tokenize and filter out empty sequences"""
    tokenized = tokenizer(
        examples["text"], 
        truncation=True, 
        max_length=args.max_length,
        padding=False,
        return_attention_mask=False
    )
    
    # Filter out sequences that are too short
    filtered_input_ids = []
    for input_ids in tokenized["input_ids"]:
        if len(input_ids) >= 10:  # Minimum sequence length
            filtered_input_ids.append(input_ids)
    
    return {"input_ids": filtered_input_ids}

print("🔄 Tokenizing dataset...")
train_data = train_data.map(tokenize_and_filter, batched=True, remove_columns=["text"])
eval_data = eval_data.map(tokenize_and_filter, batched=True, remove_columns=["text"])

# Remove empty sequences
train_data = train_data.filter(lambda x: len(x["input_ids"]) >= 10)
eval_data = eval_data.filter(lambda x: len(x["input_ids"]) >= 10)

print(f"✅ Dataset ready: {len(train_data)} train, {len(eval_data)} eval samples")

# -------------------------
# Load Base Model
# -------------------------
print(f"🤖 Loading model: {model_id}")
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.float32,
    ignore_mismatched_sizes=True
)

# Count original parameters
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

original_total, original_trainable = count_parameters(model)
print(f"📊 Original model: {original_total:,} total, {original_trainable:,} trainable")

# -------------------------
# Apply Scan Patch if needed
# -------------------------
scan_applied = False
if "scan" in args.mode:
    print("🔧 Applying YunMin Correlation Scan patch...")
    try:
        apply_scan_patch(model)
        scan_applied = is_scan_patched()
        print(f"✅ Scan patch applied: {scan_applied}")
    except Exception as e:
        print(f"❌ Scan patch failed: {e}")
        print("🔄 Continuing without scan patch...")

# -------------------------
# Apply LoRA if needed
# -------------------------
lora_applied = False
if "lora" in args.mode:
    print("🔧 Applying LoRA @ SSM-only...")
    target_modules = ["mixer.in_proj", "mixer.x_proj", "mixer.dt_proj", "mixer.out_proj"]
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    lora_applied = True
    print("✅ LoRA applied")

# -------------------------
# Apply IA3 if needed
# -------------------------
ia3_applied = False
if "ia3" in args.mode:
    print("🔧 Inserting IA3 modules...")
    insert_ia3_modules(model)
    ia3_applied = True
    print("✅ IA3 modules inserted")

# Count final parameters
final_total, final_trainable = count_parameters(model)
trainable_pct = (final_trainable / final_total) * 100

print(f"\n📊 Final model configuration:")
print(f"   Mode: {args.mode}")
print(f"   Scan Applied: {scan_applied}")
print(f"   IA3 Applied: {ia3_applied}")
print(f"   LoRA Applied: {lora_applied}")
print(f"   Total Parameters: {final_total:,}")
print(f"   Trainable Parameters: {final_trainable:,} ({trainable_pct:.2f}%)")

if hasattr(model, 'print_trainable_parameters'):
    model.print_trainable_parameters()

# -------------------------
# W&B Init
# -------------------------
print("📈 Initializing Weights & Biases...")
wandb.init(
    project="yunmin-mamba-wikitext2", 
    name=run_name, 
    config={
        "mode": args.mode,
        "model_id": model_id,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "max_length": args.max_length,
        "seed": args.seed,
        "scan_applied": scan_applied,
        "ia3_applied": ia3_applied,
        "lora_applied": lora_applied,
        "total_params": final_total,
        "trainable_params": final_trainable,
        "trainable_pct": trainable_pct
    }
)

# -------------------------
# Training Setup
# -------------------------
print("🏋️ Setting up training...")
training_args = TrainingArguments(
    output_dir=f"./runs/{run_name}",
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    evaluation_strategy="epoch",
    save_strategy="no",
    num_train_epochs=args.epochs,
    logging_steps=50,
    report_to="wandb",
    gradient_accumulation_steps=4,
    learning_rate=args.lr,
    fp16=False,
    logging_first_step=True,
    remove_unused_columns=False,
    dataloader_drop_last=True,
    metric_for_best_model="eval_loss",
    load_best_model_at_end=False,
)

# Create performance callback
perf_callback = PerformanceCallback()
perf_callback.total_params = final_total
perf_callback.trainable_params = final_trainable

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=eval_data,
    tokenizer=tokenizer,
    data_collator=default_data_collator,
    callbacks=[perf_callback],
)

# -------------------------
# Train
# -------------------------
print("🚀 Starting training...")
start_time = time.time()

try:
    trainer.train()
    training_success = True
except Exception as e:
    print(f"❌ Training failed: {e}")
    training_success = False

training_time = time.time() - start_time
print(f"⏱️ Total training time: {training_time:.2f} seconds")

# -------------------------
# Evaluate
# -------------------------
if training_success:
    print("📊 Running final evaluation...")
    eval_metrics = trainer.evaluate()
    ppl = torch.exp(torch.tensor(eval_metrics["eval_loss"])).item()
    
    # Log final metrics
    final_metrics = {
        "final_eval_ppl": ppl,
        "final_eval_loss": eval_metrics["eval_loss"],
        "total_training_time": training_time,
        "training_success": training_success,
        "scan_applied": scan_applied,
        "ia3_applied": ia3_applied,
        "lora_applied": lora_applied,
    }
    
    wandb.log(final_metrics)
    
    print(f"\n🎯 FINAL RESULTS:")
    print(f"   Mode: {args.mode}")
    print(f"   Perplexity: {ppl:.2f}")
    print(f"   Eval Loss: {eval_metrics['eval_loss']:.4f}")
    print(f"   Training Time: {training_time:.2f}s")
    print(f"   Trainable Params: {final_trainable:,} ({trainable_pct:.2f}%)")
    
    # Save results to file
    results_file = f"./results_{run_name}.txt"
    with open(results_file, 'w') as f:
        f.write(f"YunMin Correlation Scan Experiment Results\n")
        f.write(f"==========================================\n")
        f.write(f"Mode: {args.mode}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Perplexity: {ppl:.2f}\n")
        f.write(f"Eval Loss: {eval_metrics['eval_loss']:.4f}\n")
        f.write(f"Training Time: {training_time:.2f}s\n")
        f.write(f"Total Parameters: {final_total:,}\n")
        f.write(f"Trainable Parameters: {final_trainable:,} ({trainable_pct:.2f}%)\n")
        f.write(f"Scan Applied: {scan_applied}\n")
        f.write(f"IA3 Applied: {ia3_applied}\n")
        f.write(f"LoRA Applied: {lora_applied}\n")
    
    print(f"📄 Results saved to: {results_file}")

else:
    wandb.log({"training_success": False})
    print("❌ Training failed - no evaluation performed")

print("🏁 Experiment completed!")
wandb.finish()
