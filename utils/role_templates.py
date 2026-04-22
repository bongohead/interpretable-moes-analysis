"""
Render single messages
""" 

def render_single_gpt_oss(role: str, content: str) -> str:
    """
    Wrap arbitrary text as a Harmony message for GPT-OSS.
    """
    if role == 'user':
        header = f"{role}<|message|>"
    elif role == 'assistant':
        header = f"assistant<|channel|>final<|message|>"
    else:
        raise ValueError("Invalid role!")
    return f"<|start|>{header}{content}<|end|>"
    
def render_single_qwen3(role: str, content: str) -> str:
    """
    Wrap arbitrary text as a single Qwen3 message
    """
    if role == 'user':
        return f"<|im_start|>user\n{content}<|im_end|>\n"
    elif role == 'assistant':
        return f"<|im_start|>assistant\n{content}<|im_end|>\n"
    else:
        raise ValueError("Invalid role!")

def render_single_olmoe(role: str, content: str) -> str:
    """
    Wrap arbitrary text as a single OlMoE message
    """
    if role == 'user':
        return f"<|user|>\n{content}\n"
    elif role == 'assistant':
        return f"<|assistant|>\n{content}</s>\n"
    else:
        raise ValueError("Invalid role!")

def render_single_dsv2lite(role: str, content: str) -> str:
    if role == "user":
        return f"User: {content}\n\n"
    elif role == "assistant":
        return f"Assistant: {content}</s>"
    else:
        raise ValueError(f"Invalid role: {role}")

def render_single_kimi(role: str, content: str) -> str:
    """
    Supports both Moonlight and KimiVL
    """
    if role == "user":
        return f"<|im_user|>user<|im_middle|>{content}<|im_end|>"
    elif role == "assistant":
        return f"<|im_assistant|>assistant<|im_middle|>{content}<|im_end|>"
    else:
        raise ValueError(f"Invalid role: {role}")

def render_single_granite(role: str, content: str) -> str:
    if role == "user":
        return f"<|start_of_role|>user<|end_of_role|>{content}<|end_of_text|>\n"
    elif role == "assistant":
        return f"<|start_of_role|>assistant<|end_of_role|>{content}<|end_of_text|>\n"
    else:
        raise ValueError(f"Invalid role: {role}")
    
def render_single_ringmini2(role: str, content: str) -> str:
    if role == "user":
        return f"<role>HUMAN</role>{content}"
    elif role == "assistant":
        return f"<role>ASSISTANT</role>{content}"
    else:
        raise ValueError(f"Invalid role: {role}")

def render_single_glm45(role: str, content: str) -> str:
    if role == "user":
        return f"<|user|>\n{content}"
    elif role == "assistant":
        return f"<|assistant|>\n<think></think>\n{content}"
    else:
        raise ValueError(f"Invalid role: {role}")

def render_single_glm47(role: str, content: str) -> str:
    if role == "user":
        return f"<|user|>{content}"
    elif role == "assistant":
        return f"<|assistant|></think>{content}"
    else:
        raise ValueError(f"Invalid role: {role}")

def render_single_gemma4(role: str, content: str) -> str:
    if role == "user":
        return f"<|turn>user\n{content}<turn|>\n"
    elif role == "assistant":
        return f"<|turn>model\n{content}<turn|>\n"
    else:
        raise ValueError(f"Invalid role: {role}")

def render_single_lfm2(role: str, content: str) -> str:
    if role == "user":
        return f"<|im_start|>user\n{content}<|im_end|>\n"
    elif role == "assistant":
        return f"<|im_start|>assistant\n{content}<|im_end|>\n"
    else:
        raise ValueError("Invalid role!")
    
def render_single_message(model_prefix, role, content) -> str:
    """
    Params:
        @model_prefix: The model prefix; see code for supported models
        @role: One of several supported roles, includes: user, assistant
        @content: The content of the message.
    """
    if model_prefix in ['gpt-oss-20b', 'gpt-oss-120b']:
        res = render_single_gpt_oss(role, content)
    elif 'qwen3' in model_prefix:
        res = render_single_qwen3(role, content)
    elif 'moonlight' in model_prefix or 'kimi' in model_prefix:
        res = render_single_kimi(role, content)
    elif 'olmoe' in model_prefix:
        res = render_single_olmoe(role, content)
    elif 'dsv2' in model_prefix:
        res = render_single_dsv2lite(role, content)
    elif 'granite' in model_prefix:
        res = render_single_granite(role, content)
    elif 'ring-mini-2.0' in model_prefix:
        res = render_single_ringmini2(role, content)
    elif 'glm-4.5' in model_prefix:
        res = render_single_glm45(role, content)
    elif 'glm-4.7' in model_prefix:
        res = render_single_glm47(role, content)
    elif 'lfm2' in model_prefix:
        res = render_single_lfm2(role, content)
    elif 'gemma-4' in model_prefix:
        res = render_single_gemma4(role, content)
    else:
        raise ValueError("Invalid model!")

    return res