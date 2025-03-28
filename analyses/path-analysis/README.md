## About
This directory contains code to test and visualize expert path clustering & *path monosemanticity* for various pretrained models.

## Usage
- The `store-pretained-model-paths.ipynb` file contains code to run forward passes on a test dataset (the C4 validation set, though you can modify this easily) for a set of existing models (Deepseek v2 Lite, Qwen 1.5/2 MoE, OlMoE, Moonlight, and Deepseek v3/R1). The forward passes for the model are reverse engineered and modified such that they always return the sorted topk expert IDs and their corresponding weights, which are then exported to a CSV.
- The file `clustering-analysis.r` contains R code to group/cluster paths for visualization and understanding.