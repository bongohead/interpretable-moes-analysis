pip install --upgrade jupyterlab ipywidgets jupyterlab-widgets
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu124
pip install bitsandbytes==0.46.1
pip install transformers==4.53.2
pip install tiktoken blobfile # Deepseek-based 
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
pip install flash-attn==2.8.1 --no-build-isolation
pip install mamba-ssm[causal-conv1d]

# CUDA-based ML https://docs.rapids.ai/install/
pip install \
    --extra-index-url=https://pypi.nvidia.com \
    "cudf-cu12==25.6.*" "cuml-cu12==25.6.*"
# Misc
pip install triton==3.3.1
