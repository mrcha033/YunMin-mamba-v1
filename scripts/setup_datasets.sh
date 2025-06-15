#!/bin/bash

# Automatic Dataset Setup Script
# Downloads and prepares WikiText-103 and GLUE datasets

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${PROJECT_ROOT}/data_cache"

echo "📥 SETTING UP DATASETS"
echo "====================="

# Create data directory
mkdir -p "${DATA_DIR}"

# Function to check if dataset is available
check_dataset() {
    local dataset_name="$1"
    echo "Checking ${dataset_name}..."
    
    python << EOF
import sys
sys.path.append('${PROJECT_ROOT}')

try:
    if '${dataset_name}' == 'wikitext103':
        from data.wikitext103 import get_wiktext103_dataloader
        # Test loading a small sample
        dataloader = get_wiktext103_dataloader(
            split='train',
            batch_size=2,
            max_length=128,
            cache_dir='${DATA_DIR}',
            streaming=False
        )
        # Try to get one batch
        batch = next(iter(dataloader))
        print(f"✅ WikiText-103: {len(dataloader.dataset):,} samples loaded")
        
    elif '${dataset_name}' == 'glue':
        from data.glue import get_glue_dataloader
        # Test loading GLUE tasks
        tasks = ['sst2', 'mrpc', 'qnli', 'mnli']
        for task in tasks:
            try:
                dataloader = get_glue_dataloader(
                    task_name=task,
                    split='train',
                    batch_size=2,
                    max_length=128,
                    cache_dir='${DATA_DIR}'
                )
                batch = next(iter(dataloader))
                print(f"✅ GLUE {task}: {len(dataloader.dataset):,} samples loaded")
            except Exception as e:
                print(f"❌ GLUE {task}: {e}")
                sys.exit(1)
    
    print(f"✅ {dataset_name} dataset ready")
    
except Exception as e:
    print(f"❌ {dataset_name} dataset failed: {e}")
    sys.exit(1)
EOF
}

# Setup WikiText-103
echo "Setting up WikiText-103..."
check_dataset "wikitext103"

# Setup GLUE
echo "Setting up GLUE benchmark..."
check_dataset "glue"

# Create dataset info file
cat > "${DATA_DIR}/dataset_info.json" << EOF
{
    "wikitext103": {
        "status": "ready",
        "cache_dir": "${DATA_DIR}",
        "splits": ["train", "validation", "test"]
    },
    "glue": {
        "status": "ready", 
        "cache_dir": "${DATA_DIR}",
        "tasks": ["sst2", "mrpc", "qnli", "mnli", "cola", "stsb", "qqp", "rte", "wnli"]
    },
    "setup_date": "$(date -Iseconds)"
}
EOF

echo "✅ All datasets ready!"
echo "📁 Data cached in: ${DATA_DIR}"
echo "💾 Total cache size: $(du -sh "${DATA_DIR}" | cut -f1)" 