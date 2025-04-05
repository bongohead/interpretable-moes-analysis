## About
This directory contains code to compare cross-layer clusters created by clustering hidden states versus by clustering routing paths.
This is similar to `./../path-analysis/store-pretrained-model-paths.ipynb`, but also exports hidden states. 
- Use for clustering analysis to compare clusters of hidden states versus expert IDs.
- Compares dense models to MoE clusters.


## Usage
1. Run `hidden-state-export.ipynb`. For one of several pretrained MoEs, this will run forward passes through a multilingual dataset. Then, it will export the pre-MLP activations for layers 1 - 8 on a, as well as corresponding topks and sample tokens for later reconstruction.
2. Run `hidden-state-clustering.ipynb` This is reliant on data from (1). This will test several alternate cross-layer clustering algorithmns, e.g. k-means, PCA/UMAP -> HDBSCAN/K-Means.