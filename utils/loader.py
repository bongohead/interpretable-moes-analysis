"""
Model loaders
"""
import torch
from transformers.loss.loss_utils import ForCausalLMLoss
from transformers import AutoTokenizer, AutoModelForCausalLM
from packaging import version
import transformers
import importlib
# Below are for Mamba replicability - can remove if remove all SSMs
# os.environ['MAMBA_DISABLE_CUDA_KERNELS'] = '1'
# os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
# torch.use_deterministic_algorithms(True, warn_only = False)
TRANSFORMERS_VERSION = version.parse(transformers.__version__).major

def get_supported_model_metadata(model_prefix):
    """
    Get a list of supported models
    
    Params:
        @model_prefix: The model prefix string - must be one of below supported models.

    Returns:
        A tuple with:
        - HF model id
        - model arch (used for loading custom forward pass)
        - attn implementation
        - whether to use the HF default implementation
        - # hidden layers
        - # dense layers
    """
    models = {
        'olmoe': ('allenai/OLMoE-1B-7B-0125-Instruct', 'olmoe', None, True, 16, 0),
        'qwen3-30b-a3b': ('Qwen/Qwen3-30B-A3B-Instruct-2507', 'qwen3moe', None, True, 48, 0),
        'dsv2-lite': ('deepseek-ai/DeepSeek-V2-Lite', 'dsv2', None, True, 26, 1),
        'gpt-oss-20b': ('openai/gpt-oss-20b', 'gptoss', 'kernels-community/vllm-flash-attn3', True, 24, 0),
        'gpt-oss-120b': ('openai/gpt-oss-120b', 'gptoss', 'kernels-community/vllm-flash-attn3', True, 36, 0),
        'moonlight-a3b': ('moonshotai/Moonlight-16B-A3B', 'moonlight', None, True, 26, 1),
        'kimivl-a3b': ('moonshotai/Kimi-VL-A3B-Instruct', 'kimivl', None, False, 26, 1),
        'granite-4.0-tiny': ('ibm-granite/granite-4.0-tiny-preview', 'granite', None, True, 40, 0),
        'ring-mini-2.0': ('inclusionAI/Ring-mini-2.0', 'ringmini2', None, False, 19, 1),
        'glm-4.5-air': ('zai-org/GLM-4.5-Air-FP8', 'glm4moe', None, True, 45, 1),
        'glm-4.7-flash': ('zai-org/GLM-4.7-Flash', 'glm4moelite', None, True, 46, 1)
    }

    if model_prefix not in models:
        raise ValueError(f"Model index {model_prefix} not recognized. Available models: {list(models.keys())}")
    
    if TRANSFORMERS_VERSION != 5:
        raise ValueError(f"All supported models require transformers v5+. Current version: {transformers.__version__}")

    return models[model_prefix]

def load_model_and_tokenizer(model_prefix, device):
    """
    Load the model and tokenizer from HF, or from file if already downloaded.

    Params:
        @model_prefix: The model prefix string - must be one of supported models.
        @device: The device to load the model onto.
    
    Returns:
        A tuple with:
        - The tokenizer object
        - The model object
        - model architecture
        - # hidden layers
        - # dense layers
    """
    model_id, model_architecture, model_attn, model_use_hf, model_n_moe_layers, model_n_dense_layers = get_supported_model_metadata(model_prefix)

    cache_dir = '/workspace/hf'
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir = cache_dir, add_eos_token = False, add_bos_token = False, padding_side = 'left', trust_remote_code = True)
    load_params = {'cache_dir': cache_dir, 'dtype': 'auto', 'trust_remote_code': not model_use_hf, 'device_map': None, 'attn_implementation': model_attn}    
    model = AutoModelForCausalLM.from_pretrained(model_id, **load_params).to(device).eval()

    # Check loaded in correct format (MXFP4 / FA3)
    if model_architecture == 'gptoss':
        print(f"Expert precision: {model.model.layers[0].mlp.experts.down_proj.dtype}")
        print(f"Attention implementation: {model.model.config._attn_implementation}")

    # In transformers v5, avoid non-deterministic MoE implementations
    if TRANSFORMERS_VERSION == 5 and hasattr(model, 'set_experts_implementation'):
        model.set_experts_implementation('eager')

    return tokenizer, model, model_architecture, model_n_moe_layers, model_n_dense_layers


def load_custom_forward_pass(model_architecture, model = None, tokenizer = None):
    """
    Load the custom forward pass function for a given model architecture 
    
    Description:
        This loads a custom forward pass from utils.pretrained_models and verifies that it replicates the real model's forward pass exactly.
        The custom forward pass is used to extract hidden states per layer both post-attention and post-MLP.
        Also extracts a variety of MoE-related routing metadata.
        (For more basic purposes, this can be replaced by simpler hooks if desired).

    Params:
        @model_architecture: One of supported architectures returned by get_supported_model_metadata.
        @model: (optional) The standard HF model object; used for validation against the custom forward pass if passed with `model`.
        @tokenizer: (optional) The tokenizer object; used for validation against the custom forward pass if passed with `tokenizer`.
    """
    model_module = importlib.import_module(f"utils.pretrained_models.{model_architecture}")
    run_forward_with_hs = getattr(model_module, f"run_{model_architecture}_return_topk")

    @torch.no_grad()
    def _verify_custom_forward_pass(model, pad_token_id = tokenizer.pad_token_id):

        # Load and compare results
        inputs = tokenizer(
            ['Hi! I am a dog and I like to bark', 'Vegetables are good for'],
            return_tensors = 'pt', padding = 'max_length', truncation = True, max_length = 512
        ).to(model.device)
        original_results = model(**inputs, use_cache = False)
        custom_results = run_forward_with_hs(model, inputs['input_ids'], inputs['attention_mask'], return_hidden_states = True)
        assert torch.equal(original_results.logits, custom_results['logits']), 'Error in custom forward'
        assert len(custom_results['all_topk_experts']) == len(custom_results['all_topk_weights']), 'Length of topk IDs and weights not equal'

        # Misc checks
        print(f"Length of topk: {len(custom_results['all_topk_experts'])}")
        print(f"Topk size: {custom_results['all_topk_experts'][0].shape}")
        print(f"First token topk IDs: {custom_results['all_topk_experts'][0][1,]}")
        print(f"First token topk weights: {custom_results['all_topk_weights'][0][1,]}")
        loss = ForCausalLMLoss(
            custom_results['logits'], torch.where(inputs['input_ids'] == pad_token_id, torch.tensor(-100), inputs['input_ids']), custom_results['logits'].size(-1)
        ).detach().cpu().item()
        print(f"LM loss: {loss}")
        print(f"Hidden states layers (pre-mlp | post-layer): {len(custom_results['all_pre_mlp_hidden_states'])} | {len(custom_results['all_hidden_states'])}")
        print(f"Hidden state size (pre-mlp | post-layer): {(custom_results['all_pre_mlp_hidden_states'][0].shape)} | {(custom_results['all_hidden_states'][0].shape)}")
        if model_architecture != 'gptoss': # Check run_gptoss_return_topk
            print(f"Expert outputs : {(custom_results['all_expert_outputs'][0].shape)}")
        print(f"Router logits : {(custom_results['all_router_logits'][0].shape)}")
        print('Verified custom forward pass successfully matches original model output!')

    if model is not None and tokenizer is not None:
        _verify_custom_forward_pass(model, tokenizer.pad_token_id)

    return run_forward_with_hs
