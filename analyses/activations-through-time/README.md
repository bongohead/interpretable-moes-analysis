## About
This directory contains code to visualize expert routing through time. Requires that during training, you ran the `save_expert_usage` [function](https://github.com/bongohead/interpretable-moes/blob/master/helpers/expert_tracking.py).

## Usage
1. During model training, run the [`save_expert_usage` function](https://github.com/bongohead/interpretable-moes/blob/master/helpers/expert_tracking.py) to save logs of expert activations at various points of training. This retains them as `.pt` files.
2. On the remote machine for model training, run the [`convert_folder_of_pts` function](https://github.com/bongohead/interpretable-moes/blob/master/helpers/expert_tracking.py) to train the model.
3. Run `windows_downloader.ps1` to SCP the files into your local machine for testing.
4. Run `visualize-activations.r` to perform the visualizations of routing state changes over time.