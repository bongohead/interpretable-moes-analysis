## About
This directory contains code for various analyses dependent on the pre-MLP hidden states. All code here requires that `./../export-data/export-activations.ipynb` has been run.

## Usage
- `router-analysis.ipynb` contains code to analyze the MoE router and understand how the hidden state is correlated to routing weights. 
- `svd.ipynb` contains code to do SVD decomposition. 

2. Run `hidden-state-clustering-moe.ipynb` This is reliant on data from (1). This will test several alternate cross-layer clustering algorithms, e.g. k-means, PCA/UMAP -> HDBSCAN/K-Means, to analyze the clusters formed. This also contains code to analyze the routing weights and how they interact with activations.
3. Run `hidden-state-export-dense.ipynb`. This will run forward passes and output pre-MLP activations using a pretrained dense model, similar to (1).
4. Run `hidden-state-clustering-dense.ipynb`. This is reliant on data from (3), and is the dense equivalent to (2). This will also test several layer-wise clustering algorithms.