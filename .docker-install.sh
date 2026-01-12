#!/bin/bash
# Helper script to install dependencies system-wide in Docker container
# PyTorch is already pre-installed, so we skip it

# Install dependencies from requirements.txt (version constraints prevent conflicts)
# This includes scipy which is needed by physical_ai_av
pip install -r requirements.txt

# Ensure scipy>=1.14.0 for RigidTransform support (required by physical_ai_av)
pip install "scipy>=1.14.0" --upgrade

# Install physical_ai_av separately with --no-deps to bypass numpy>=2.0.0 requirement
pip install --no-deps physical_ai_av>=0.1.0

# Install package in editable mode
pip install -e . --no-deps

echo "✓ Dependencies installed system-wide"
echo "Note: PyTorch, torchvision, numpy, pandas, pyarrow are pre-installed and won't be upgraded"
echo ""
echo "⚠️  IMPORTANT: To use the dataset, authenticate with HuggingFace:"
echo "   hf auth login"

