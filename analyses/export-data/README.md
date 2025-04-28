## About
This directory contains code to run forward passes, storing activations, top-ks, or other metadata for other analyses.


## Usage
- `export-activations.ipynb`: For several pretrained MoEs, this will run forward passes through a multilingual C4 dataset. Then, it will export the pre-MLP activations, top-k expert outputs, as well as corresponding top-k expert ids and sample tokens for later reconstruction.
- `export-activations-dense.ipynb`: For several pretrained dense models, this will run forward passes through the same multilingual C4 dataset. Then, it will export the pre-MLP activations and corresponding sample tokens for later reconstruction.
- `export-topk.ipynb`: This runs forward passes through a multilingual C4 dataset, but does NOT store any activations - only the top-k expert ids and sample tokens. This is obviously way more memory efficient and can generate significantly larger samples!

These support models which have been reversed engineered in `utils/pretrained_models.py`. Supported MoEs include:
- OlMoE architecture, includes OLMoE-1B-7B-0125-Instruct (1B/7B)
- Qwen2MoE architecture, inclues Qwen1.5-MoE-A2.7B-Chat (2.7B/14.3B), Qwen2-57B-A14B (14B/57B)
- Deepseek v2 architecture, includes Deepseek-v2-Lite (2.4B/15.7B), Deepseek-v2 (21B/236B)
- Deepseek v3 architecture, includes Deepseek-v3 (37B/671B), Deepseek-R1 (37B/671B), Moonlight-16B-A3B (3B/16B)

Supported dense models include:
- Qwen2 architecture, including all Qwen2.5B versions.