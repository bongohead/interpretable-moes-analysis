# About
This contains code for visualizations and intepretability analyses linked to the [interpretable-moes](https://github.com/bongohead/interpretable-moes) project.

The folder `analysis` contains all code for visualizations/interpretability.

# Setup
Before running analysis code, run through the following setup:

1. First, clone this repo: `git clone https://github.com/bongohead/interpretable-moes-analysis`.
2. Then, run `windows_install_packages.ps1` and `windows_setup_paths.ps1` to setup the Python environment.
3. The `datasets` directory contains subdirectories with data re-used across multiple analyses. Some require additional setup.
    - `contextual-tokens`: A directory of YAML files containing text samples.
        - Each file corresponds to a single polysemantic token with 3 meanings. The file will contain a set of text samples corresponding to each of those 3 meanings. 
        - These are useful for a variety of tasks to understand contextual versus token-ID-based routing.
        - **Setup**: No additional ssetup needed.
    - `tokenizer-vocab`: A directory of CSVs containing tokenizer mappings of token IDs to vocabulary here.
        - **Setup**: these are not committed to Git, so you will need to create them as needed. Run `helpers/export_vocab_as_csv` for any tokenizers you need in your analyses.
    - `saved-models`: A directory of pytorch saved models.
        - **Setup**: Place any saved pytorch models for various analyses here.


# Analysis Direcotires
1. `analysis/activations-through-time` contains code to analyze and understand expert routing distributions through time.
<p align="center"><img src="images/expert-distribution-over-time.png" width="400px"></p>

2. `analysis/cross-layer-routing` contains code to visualize the routing flow across different layers.
<p align="center"><img src="images/cross-layer-routing-1.png" width="400px"></p>