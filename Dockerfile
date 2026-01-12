# Use NVIDIA PyTorch container with PyTorch and CUDA pre-installed
FROM nvcr.io/nvidia/pytorch:25.08-py3

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PATH="/root/.local/bin:$PATH"

# Install uv package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Set working directory
WORKDIR /workspace

# Copy entire repository (will be overridden by volume mount, but needed for build)
COPY . .

# Default command
CMD ["/bin/bash"]

