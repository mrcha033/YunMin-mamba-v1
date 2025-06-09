#!/bin/bash
set -e
pip install --quiet pytest
pytest -v
