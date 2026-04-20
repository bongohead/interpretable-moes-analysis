"""
Render single messages
""" 

def render_single_gpt_oss(role: str, content: str) -> str:
    """
    Wrap arbitrary text as a Harmony message for GPT-OSS.
    Notes:
        - Allows for an empty tool name.
    """
    if role == 'user':
        header = f"{role}<|message|>"
    elif role == 'cot':
        header = f"assistant<|channel|>analysis<|message|>"
    elif role == 'assistant':
        header = f"assistant<|channel|>final<|message|>"
    else:
        raise ValueError("Invalid role!")
    return f"<|start|>{header}{content}<|end|>"
    
def render_single_glm4(role: str, content: str) -> str:
    """
    Wrap arbitrary text as a single GLM4 message
    """
    if role == 'user':
        return f"<|user|>\n{content}"
    elif role == 'cot':
        return f"<|assistant|>\n<think>{content}</think>\n"
    elif role == 'assistant':
        return f"<|assistant|>\n<think></think>\n{content}"
    else:
        raise ValueError("Invalid role!")

def render_single_qwen3(role: str, content: str) -> str:
    """
    Wrap arbitrary text as a single Qwen3 message
    """
    if role == 'user':
        return f"<|im_start|>user\n{content}<|im_end|>\n"
    elif role == 'cot':
        return f"<|im_start|>assistant\n<think>\n{content}\n</think>\n\n<|im_end|>\n"
    elif role == 'assistant':
        return f"<|im_start|>assistant\n{content}<|im_end|>\n"
    else:
        raise ValueError("Invalid role!")
        
def render_single_message(model_prefix, role, content, tool_name = None) -> str:
    """
    Params:
        @model_architecture: The model prefix; see code for supported models
        @role: One of several suppored roles, includes: user, cot, assistant
        @content: The content of the message.
    Example:
        render_single_message('gptoss', 'user', None)
    """
    if model_prefix in ['gpt-oss-20b', 'gpt-oss-120b']:
        res = render_single_gpt_oss(role, content)
    elif model_prefix in ['qwen3-30b-a3b']:
        res = render_single_qwen3(role, content)
    else:
        raise ValueError("Invalid model!")

    return res