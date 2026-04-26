# Supported models
Currently tested for transformers v5.6.2:
- Qwen 3.5/3.6 MoEs: `Qwen3.5-35B-A3B`, `Qwen-3.6-35B-A3B`
- Qwen 3 MoEs: `Qwen3-30B-A3B-Instruct-2507`, `Qwen3-30B-A3B-Thinking-2507`, `Qwen3-Coder-30B-A3B-Instruct`
- GPT-OSS: `gpt-oss-20b`, `gpt-oss-120b`
- OlMoE: `OLMoE-1B-7B-0125`
- LFM2 MoEs: `LFM2-8B-A1B`, `LFM2-24B-A2B`
- Gemma 4 MoEs: `gemma-4-26B-A4B` (see notes at top of `gemma4.py` - the returned values should be interpreted differently due to how Gemma handles models)
- Granite 4.0 Hybrid MoEs: `granite-4.0-h-tiny`
- Ring Mini 2.0: `Ring-Mini-2.0` (with patches)

## Patches for remote-loaded models on Transformers v5

Some remote-loaded `transformers` models (for example, models loaded with `trust_remote_code=True` from model-specific repos rather than from architectures natively supported by the library) were written against Transformers v4 APIs. Under Transformers v5, a small number of model-specific patches may be required.

Procedure:

1. Download/load the model files as normal so that the remote code is cached locally.
2. Locate the cached remote module under the Hugging Face modules cache, typically `~/.cache/huggingface/modules/transformers_modules/<org>/<model>/<hash>/`.
3. Edit the relevant `modeling_*.py` file inside that cached module directory.
4. Apply the model-specific changes listed below.
5. Reload the model in a fresh Python process.

Notes:
- These patches modify the local cached copy of the remote model code only.
- They may be overwritten if the cached module hash changes or the model is re-downloaded.
- Common v4 to v5 incompatibilities include:
  - old RoPE initialization assumptions (for example, assuming `ROPE_INIT_FUNCTIONS["default"]` exists),
  - deprecated attention-mask utilities from `transformers.modeling_attn_mask_utils`,
  - deprecated `config.use_return_dict` access instead of `return_dict`.

### Ring-mini-2.0

1. In `modeling_bailing_moe_v2`, replace the `__init__` and add the static method:

    ```python
    class BailingMoeV2RotaryEmbedding(nn.Module):
        def __init__(self, config: BailingMoeV2Config, device=None):
            super().__init__()
            if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
                self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
            else:
                self.rope_type = "default"

            self.max_seq_len_cached = config.max_position_embeddings
            self.original_max_seq_len = config.max_position_embeddings
            self.config = config

            rope_init_fn = self.compute_default_rope_parameters
            if self.rope_type != "default":
                rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

            inv_freq, self.attention_scaling = rope_init_fn(self.config, device)
            self.register_buffer("inv_freq", inv_freq, persistent=False)
            self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)

        @staticmethod
        def compute_default_rope_parameters(config=None, device=None, seq_len=None):
            base = getattr(config, "rope_theta", 10000.0)
            partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
            head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
            dim = int(head_dim * partial_rotary_factor)

            attention_factor = 1.0
            inv_freq = 1.0 / (
                base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
            )
            return inv_freq, attention_factor
    ```

2. Then in `BailingMoeV2Model.forward`, add `from transformers.masking_utils import create_causal_mask` at the top, and also replace the mask-building block:
    ```python
            if self._use_flash_attention_2:
                # 2d mask is passed through the layers
                attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
            elif self._use_sdpa and not output_attentions:
                attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                    attention_mask,
                    (batch_size, seq_length),
                    inputs_embeds,
                    past_seen_tokens,
                )
            else:
                attention_mask = _prepare_4d_causal_attention_mask(
                    attention_mask, (batch_size, seq_length), inputs_embeds, past_seen_tokens
                )
    ```

    with

    ```python
            if self._use_flash_attention_2:
                # 2d mask is passed through the layers
                attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
            else:
                # 4d mask is passed through the layers
                attention_mask = create_causal_mask(
                    config=self.config,
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    position_ids=position_ids,
                )
    ```

3. Then in both `BailingMoeV2Model.forward` and `BailingMoeV2ForCausalLM.forward`, replace:
    ```python
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    ```

    with

    ```python
            return_dict = return_dict if return_dict is not None else self.config.return_dict
    ```

### Kimi-VL-A3B

1. In `modeling_kimi_vl.py`, add this help near the top:

    ```python
    def _normalize_rope_scaling_compat(rope_scaling):
        if rope_scaling is None:
            return None

        if isinstance(rope_scaling, dict):
            rope_scaling = dict(rope_scaling)
            scaling_type = rope_scaling.get("type", rope_scaling.get("rope_type", "default"))

            # Old Kimi code expects plain RoPE to be represented as None
            if scaling_type == "default":
                return None

            rope_scaling["type"] = scaling_type
            return rope_scaling

        return rope_scaling
    ```

2. Then in `DeepseekV3Attention.__init__`, right BEFORE `self._init_rope()`, insert:

    ```python
    self.config.rope_scaling = _normalize_rope_scaling_compat(
        getattr(self.config, "rope_scaling", None)
    )
    ```
