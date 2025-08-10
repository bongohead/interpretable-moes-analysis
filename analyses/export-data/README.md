## About
This directory contains code to run forward passes, storing activations, top-ks, or other metadata for other analyses.


## Usage
- `export-activations.ipynb`: For several pretrained MoEs, this will run forward passes through a multilingual C4 dataset. Then, it will export the pre-MLP activations, top-k expert outputs, as well as corresponding top-k expert ids and sample tokens for later reconstruction.


- `export-activations-dense.ipynb`: For several pretrained dense models, this will run forward passes through the same multilingual C4 dataset. Then, it will export the pre-MLP activations and corresponding sample tokens for later reconstruction.
- `export-topk.ipynb`: This runs forward passes through a multilingual C4 dataset, but does NOT store any activations - only the top-k expert ids and sample tokens. This is obviously way more memory efficient and can generate significantly larger samples!

These support models which have been reversed engineered in `utils/pretrained_models.py`. Supported MoEs include:
- OlMoE architecture: OLMoE-1B-7B-* (1B/7B)
- Qwen2MoE architecture: Qwen1.5-MoE-A2.7B-* (2.7B/14.3B), Qwen2-57B-A14B-* (14B/57B)
- Deepseek v2 architecture: Deepseek-v2-Lite (2.4B/15.7B), Deepseek-v2 (21B/236B) -> use trust_remote_code = False
- Deepseek v3 architecture: Deepseek-v3 (37B/671B), Deepseek-R1 (37B/671B), Moonlight-16B-A3B (3B/16B) -> use trust_remote_code = False
- Qwen3MoE architecture: Qwen3-30B-A3B (3B/30B), Qwen3-235B-A22B (22B/235B), Qwen3-Coder (35B/480B)
- KimiVL architecture: Kimi-VL-A3B-* (3B/16B)
- Granite architecture: Granite-4.0-Tiny-* (1B/7B)
- GLM4MoE architecture: GLM-4.5 (32B/355B), GLM-4.5-Air (12B/106B) * Supports multi-GPU as well as official ZAI FP8 versions
- GTP-OSS architecture: GPT-OSS-120B (5B/117B), GPT-OSS-20B (4B/21B)

All models load in bf16 except GPT-OSS, which load in partial MXFP4.