# Notes

- `insert_ia3_modules` now wraps `Linear`, `Conv1d` and `LayerNorm` layers with per-channel scaling modules.
- `train_yunmin.py` provides a simple CLI with `--ia3` to enable these modules.
- `quick_test.py` and `batch_test.py` include an `ia3` mode for fast sanity checks.
