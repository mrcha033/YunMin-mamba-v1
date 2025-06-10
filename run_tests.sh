#!/bin/bash
set -e
pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu
pip install --quiet peft
pip install --quiet pytest
pytest -v
