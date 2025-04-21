pip install --upgrade jupyterlab ipywidgets jupyterlab-widgets
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install bitsandbytes==0.45.5
pip install transformers==4.51.3
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
# CUDA-based ML https://docs.rapids.ai/install/
pip install \
    --extra-index-url=https://pypi.nvidia.com \
    "cudf-cu12==25.4.*" "cuml-cu12==25.4.*"
# Misc
pip install triton==3.2.0
pip install flash-attn==2.7.4.post1 --no-build-isolation