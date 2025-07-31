pip install --upgrade jupyterlab ipywidgets jupyterlab-widgets
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu124
pip install bitsandbytes==0.46.1
pip install transformers==4.54.1
pip install accelerate==1.9.0
pip install plotly.express
pip install wandb
pip install pyyaml
pip install pyarrow
pip install termcolor
pip install pandas
pip install tqdm
pip install python-dotenv
pip install datasets
pip install scikit-learn
pip install kaleido

# Model specific
pip install tiktoken blobfile # For Deepseek-based architectures 
pip install mamba-ssm==2.2.5 # Dependency for Granite
pip install causal-conv1d==1.5.2 # Dependency for Mamba
pip install compressed-tensors # Dependency for 8bit Glm4.5

# CUDA-based ML https://docs.rapids.ai/install/
pip install \
    --extra-index-url=https://pypi.nvidia.com \
    "cudf-cu12==25.6.*" "cuml-cu12==25.6.*"

# Misc
pip install flash-attn==2.8.1 --no-build-isolation
pip install triton==3.3.1