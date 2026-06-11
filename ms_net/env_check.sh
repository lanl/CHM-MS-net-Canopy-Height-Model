#!/usr/bin/env bash

echo "===== SYSTEM / ENV INFO ====="

echo -e "\n--- NVIDIA / GPU INFO ---"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
else
    echo "nvidia-smi not found"
fi

echo -e "\n--- CUDA INFO ---"
if command -v nvcc &> /dev/null; then
    nvcc --version | grep "release"
else
    echo "nvcc not found"
fi

echo -e "\n--- PYTHON INFO ---"
python --version 2>&1

echo -e "\n--- PYTORCH INFO ---"
python - << 'EOF'
try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA (torch): {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
except ImportError:
    print("PyTorch not installed")
EOF

echo -e "\n===== DONE ====="
