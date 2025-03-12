# About
This directory contains code to visualize the routing flow across different layers

# Usage
1. Save the model checkpoints during training.
2. Copy `export-activations-experiment-52.ipynb` to a new file. This notebook loads the model classes (modify it to match the class of the model used during training) and the saved model checkpoints. It then runs forward passes on the contextual tokens stores in `datasets/contextual-tokens` and exports the routing for different (token, meaning) pairs. 
3. Run `visualize-routing.r` to create visualizations of the routing by context. 
 modify it that the model classes match the class of the model used during training. This notebook 