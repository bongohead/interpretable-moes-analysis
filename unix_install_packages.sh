#!/bin/bash

# Jupyter
pip install --upgrade jupyterlab ipywidgets jupyterlab-widgets

# Core
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install transformers==4.56.1
pip install accelerate==1.10.1
pip install flash-attn==2.8.3 --no-build-isolation # https://github.com/Dao-AILab/flash-attention/releases Make sure there's a prebuilt wheel for this torch+cuda+python vers
pip install triton==3.4.0

# Model specific
pip install tiktoken==0.11.0 blobfile==3.1.0 # For Deepseek-based architectures 
pip install kernels==0.10.1 # As of Sep 2025, dep for GPT-OSS (supports flash attention 3 sinks)
pip install compressed-tensors==0.11.0 # Dependency for 8bit Glm4.5
# pip install -v mamba-ssm==2.2.5 causal-conv1d==1.5.2 # Dependency for Granite

# Analysis + viz
pip install plotly.express pandas kaleido nbformat
pip install python-dotenv pyyaml tqdm termcolor
pip install wandb
pip install datasets

# CUDA-based ML https://docs.rapids.ai/install/
pip install --extra-index-url=https://pypi.nvidia.com "cudf-cu12==25.8.*" "cuml-cu12==25.8.*"