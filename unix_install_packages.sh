pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install bitsandbytes==0.45.4
pip install transformers==4.50.3
pip install triton==3.2.0
pip install flash-attn==2.7.4.post1 --no-build-isolation
pip install jupyter lab
pip install plotly.express
pip install wandb
pip install pyyaml
pip install pyarrow
pip install termcolor
pip install pandas
pip install tqdm
pip install sqlalchemy
pip install python-dotenv
pip install datasets
pip install scikit-learn
pip install umap-learn
# CUDA clustering https://docs.rapids.ai/install/
pip install \
    --extra-index-url=https://pypi.nvidia.com \
    "cudf-cu12==25.2.*" "cuml-cu12==25.2.*"