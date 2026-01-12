<div align="center">

# 🏔️ Alpamayo 1

### Bridging Reasoning and Action Prediction for Generalizable Autonomous Driving

[![HuggingFace](https://img.shields.io/badge/🤗%20Model-Alpamayo--R1--10B-blue)](https://huggingface.co/nvidia/Alpamayo-R1-10B)
[![arXiv](https://img.shields.io/badge/arXiv-2511.00088-b31b1b.svg)](https://arxiv.org/abs/2511.00088)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](./LICENSE)

</div>

_Note: Following the release of [NVIDIA Alpamayo](https://nvidianews.nvidia.com/news/alpamayo-autonomous-vehicle-development) at CES 2026, Alpamayo-R1 has been renamed to Alpamayo 1._

### 1. Install uv (if not already installed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

### 2. Set up the environment

```bash
uv venv ar1_venv
source ar1_venv/bin/activate
uv sync --active
```

### 3. Authenticate with HuggingFace

The model requires access to gated resources. Request access here:
- 🤗 [Physical AI AV Dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)
- 🤗 [Alpamayo Model Weights](https://huggingface.co/nvidia/Alpamayo-R1-10B)

Then authenticate:

```bash
hf auth login
```

Get your token at: https://huggingface.co/settings/tokens

## Docker Setup

### Prerequisites
- Docker with NVIDIA Container Toolkit installed
- NVIDIA GPU with CUDA support

### Using Docker Compose (Recommended)

```bash
# Build and start the container
docker-compose build
docker-compose up -d

# Access the container
docker-compose exec alpamayo bash

# Inside the container, install dependencies system-wide:
# PyTorch with CUDA is already pre-installed, so install remaining dependencies
# Install dependencies first, then the package (order matters for imports to work)

# Option 1: Using traditional pip (simpler, recommended)
# requirements.txt includes version constraints for numpy/pyarrow to prevent conflicts
pip install -r requirements.txt  # Install dependencies including scipy (version constraints prevent upgrades)
pip install "scipy>=1.14.0" --upgrade  # Ensure scipy>=1.14.0 for RigidTransform (required by physical_ai_av)
pip install --no-deps physical_ai_av>=0.1.0  # Install with --no-deps to bypass numpy>=2.0.0 requirement
pip install -e . --no-deps  # Install package in editable mode system-wide

# Option 2: Using uv (if you prefer uv)
# uv pip install --system -r requirements.txt
# uv pip install --system -e . --no-deps

# Authenticate with HuggingFace
hf auth login
```

### Using Docker directly

```bash
# Build the image
docker build -t alpamayo-r1:latest .

# Run the container with GPU support
docker run --gpus all --ipc=host -it -v $(pwd):/workspace alpamayo-r1:latest bash

# Inside the container, install dependencies system-wide:
# PyTorch is already pre-installed, install remaining dependencies
pip install --upgrade-strategy only-if-needed -r requirements.txt  # Won't upgrade existing packages
pip install -e . --no-deps  # Install package in editable mode system-wide
hf auth login
```

### Using VS Code Dev Containers

1. Open the repository in VS Code
2. Press `F1` and select "Dev Containers: Reopen in Container"
3. VS Code will build and start the container automatically
4. Once inside, install dependencies system-wide:
   ```bash
   # PyTorch with CUDA is already pre-installed
   # requirements.txt includes version constraints to prevent conflicts
   pip install -r requirements.txt
   pip install "scipy>=1.14.0" --upgrade  # Ensure scipy>=1.14.0 for RigidTransform
   pip install --no-deps physical_ai_av>=0.1.0  # Install with --no-deps to bypass numpy>=2.0.0 requirement
   pip install -e . --no-deps
   hf auth login  # Required to access the gated dataset
   ```

**Note:** The Docker container uses `nvcr.io/nvidia/pytorch:25.08-py3` which includes PyTorch 2.8.0 with CUDA support, Python 3.12, and CUDA 13.0 pre-installed. You still need to create the virtual environment and install dependencies using `uv sync` as described in the original setup instructions.

**NGC Access:** To pull the NVIDIA PyTorch image, you may need to authenticate with NGC:
```bash
docker login nvcr.io
# Use your NGC API key (get it from https://ngc.nvidia.com/setup/api-key)
```

## Running Inference

### Test script

NOTE: This script will download both some example data (relatively small) and the model weights (22 GB).
The latter can be particularly slow depending on network bandwidth.
For reference, it takes around 2.5 minutes on a 100 MB/s wired connection.

```bash
python src/alpamayo_r1/test_inference.py
```

In case you would like to obtain more trajectories and reasoning traces, please feel free to change
the `num_traj_samples=1` argument to a higher number (Line 60).

### Interactive notebook

We provide a notebook with similar inference code at `notebook/inference.ipynb`.

## Project Structure

```
alpamayo/
├── notebook/
│   └── inference.ipynb                  # Example notebook
├── src/
│   └── alpamayo_r1/
│       ├── action_space/
│       │   └── ...                      # Action space definitions
│       ├── diffusion/
│       │   └── ...                      # Diffusion model components
│       ├── geometry/
│       │   └── ...                      # Geometry utilities and modules
│       ├── models/
│       │   ├── ...                      # Model components and utils functions
│       ├── __init__.py                  # Package marker
│       ├── config.py                    # Model and experiment configuration
│       ├── helper.py                    # Utility functions
│       ├── load_physical_aiavdataset.py # Dataset loader
│       ├── test_inference.py            # Inference test script
├── pyproject.toml                       # Project dependencies
└── uv.lock                              # Locked dependency versions
```

## Troubleshooting

### Flash Attention issues

The model uses Flash Attention 2 by default. If you encounter compatibility issues:

```python
# Use PyTorch's scaled dot-product attention instead
config.attn_implementation = "sdpa"
```

## License

Apache License 2.0 - see [LICENSE](./LICENSE) for details.

## Disclaimer

Alpamayo 1 is a pre-trained reasoning model designed to accelerate research and development in the autonomous vehicle (AV) domain. It is intended to serve as a foundation for a range of AV-related use cases-from instantiating an end-to-end backbone for autonomous driving to enabling reasoning-based auto-labeling tools. In short, it should be viewed as a building block for developing customized AV applications.

Important notes:

- Alpamayo 1 is provided solely for research, experimentation, and evaluation purposes.
- Alpamayo 1 is not a fully fledged driving stack. Among other limitations, it lacks access to critical real-world sensor inputs, does not incorporate required diverse and redundant safety mechanisms, and has not undergone automotive-grade validation for deployment.

By using this model, you acknowledge that it is a research tool intended to support scientific inquiry, benchmarking, and exploration—not a substitute for a certified AV stack. The developers and contributors disclaim any responsibility or liability for the use of the model or its outputs.
