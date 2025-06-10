# Experiment Guide

This guide highlights how to run experiments with the IA³ adaptation modules.

## Running with IA³

Use `train_yunmin.py` with the `--ia3` flag to insert per-channel IA³ scaling layers before PEFT adapters.

```bash
python train_yunmin.py --ia3
```

The script uses the same defaults as `train.py` but exposes a command line interface for quick experimentation.
