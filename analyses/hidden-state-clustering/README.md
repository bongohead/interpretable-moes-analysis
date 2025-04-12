## About
This directory contains code to compare cross-layer clusters created by clustering hidden states versus by clustering routing paths.
This is similar to `./../path-analysis/store-pretrained-model-paths.ipynb`, but also exports hidden states. 
- Use for clustering analysis to compare clusters of hidden states versus expert IDs.
- Compares dense models to MoE clusters.


## Usage
1. Run `hidden-state-export-moe.ipynb`. For one of several pretrained MoEs, this will run forward passes through a multilingual dataset. Then, it will export the pre-MLP activations, as well as corresponding topks and sample tokens for later reconstruction.
2. Run `hidden-state-clustering-moe.ipynb` This is reliant on data from (1). This will test several alternate cross-layer clustering algorithms, e.g. k-means, PCA/UMAP -> HDBSCAN/K-Means, to analyze the clusters formed. This also contains code to analyze the routing weights and how they interact with activations.
3. Run `hidden-state-export-dense.ipynb`. This will run forward passes and output pre-MLP activations using a pretrained dense model, similar to (1).
4. Run `hidden-state-clustering-dense.ipynb`. This is reliant on data from (3), and is the dense equivalent to (2). This will also test several layer-wise clustering algorithms.