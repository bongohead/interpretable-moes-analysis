"""
Helper functions to assign tokens to their roles - these are largely LLM generated; they're validated carefully by checking token counts.
"""
import pandas as pd
import numpy as np
import re

def label_gpt_oss_content_roles(sample_df: pd.DataFrame) -> pd.DataFrame:
    """
    Label gpt-oss (Harmony) token streams with content-only roles and role segments.

    Returns:
        - role: one of {system, developer, user, assistant, cot, tool_call, tool} or None
        - is_content: bool
        - seg_ix: int or None
        - token_in_seg_ix: int or None
    """
    required = {"prompt_ix", "token_ix", "token"}
    missing = required - set(sample_df.columns)
    if missing:
        raise ValueError(f"sample_df missing required columns: {sorted(missing)}")

    d = (sample_df.sort_values(["prompt_ix", "token_ix"]).reset_index(drop = True).copy())

    # ---- Message segmentation + content-span detection (literal token matching) ----
    d["msg_seg_id"] = d.groupby("prompt_ix")["token"].transform(lambda s: (s == "<|start|>").cumsum())
    d["is_message"] = d["token"].eq("<|message|>")
    d["is_close"] = d["token"].isin(["<|end|>", "<|return|>", "<|call|>"])

    d["after_msg"] = d.groupby(["prompt_ix", "msg_seg_id"])["is_message"].cumsum().gt(0)
    d["before_end"] = d.groupby(["prompt_ix", "msg_seg_id"])["is_close"].cumsum().eq(0)

    # This is exactly the old in_content_span semantics.
    d["is_content"] = d["after_msg"] & d["before_end"] & ~d["is_message"]

    # ---- Header extraction and segment-role classification ----
    # Header tokens: tokens in [<|start|>, <|message|>) excluding <|start|>.
    d["is_header_tok"] = (d["msg_seg_id"] > 0) & (~d["after_msg"]) & (d["token"] != "<|start|>")

    def classify_header(tok_series: pd.Series):
        toks = tok_series.tolist()
        if not toks:
            return None

        toks_lc = [t.lower() for t in toks]
        header = "".join(toks_lc)
        header_ns = header.replace(" ", "")  # defensive, in case any tokens contain spaces

        # Tool output: functions.<tool_name> ...
        if header_ns.startswith("functions."):
            return "tool"

        # Tool call: assistant ... to=functions.<tool> ...
        if header_ns.startswith("assistant") and "to=functions" in header_ns:
            return "tool_call"

        # Assistant channels (no fallback)
        if header_ns.startswith("assistant") and "<|channel|>analysis" in header_ns:
            return "cot"
        if header_ns.startswith("assistant") and (
            "<|channel|>final" in header_ns or "<|channel|>commentary" in header_ns
        ):
            return "assistant"

        # System / user / developer
        if header_ns.startswith("user"):
            return "user"
        if header_ns.startswith("system"):
            return "system"
        if header_ns.startswith("developer"):
            return "developer"

        return None

    seg_roles = (
        d.loc[d["is_header_tok"]]
         .groupby(["prompt_ix", "msg_seg_id"])["token"]
         .agg(classify_header)
         .rename("segment_role")
    )
    d = d.merge(seg_roles, on=["prompt_ix", "msg_seg_id"], how="left")

    # Role is only assigned inside content spans.
    d["role"] = np.where(d["is_content"], d["segment_role"], None)

    # ---- seg_ix + token_in_seg_ix (only for labeled content tokens) ----
    is_labeled = d["is_content"] & d["role"].notna()

    prev_is_labeled = is_labeled.groupby(d["prompt_ix"]).shift(1, fill_value=False)
    prev_role = d.groupby("prompt_ix")["role"].shift(1)

    is_new_seg = is_labeled & (~prev_is_labeled | (d["role"] != prev_role))
    seg_counter = is_new_seg.groupby(d["prompt_ix"]).cumsum()  # 1,2,3,... at starts

    d["seg_ix"] = np.where(is_labeled, seg_counter - 1, None)

    d["token_in_seg_ix"] = None
    d.loc[is_labeled, "token_in_seg_ix"] = (
        d.loc[is_labeled].groupby(["prompt_ix", "seg_ix"]).cumcount()
    )

    # ---- Final cleanup / ordering ----
    out = d.drop(
        columns=["msg_seg_id", "is_message", "is_close", "after_msg", "before_end", "is_header_tok", "segment_role",],
        errors="ignore",
    )

    out = out.sort_values(["prompt_ix", "token_ix"]).reset_index(drop=True)
    return out


def label_content_roles(model_prefix, sample_df):
    """
    Takes a token-level df, labels each token with its role only within the content span. Makes no assumption about the number of messages in the sequence.
    The input is a token-level dataframe spanning multiple prompts (conversations). Each prompt is identified by prompt_ix and contains a flat serialized token stream (including template tags, role sentinels, wrappers, and semantic content).

    Params: 
        @sample_df: A df with columns:
        - prompt_ix: unique id for each serialized prompt/sequence, equivalent to an index on (batch_ix, sequence_ix)
        - token_ix: position within the prompt
        - token: decoded token string

    Description:
        - We assign `role` ONLY to semantic content tokens.
        - Template/sentinel/control tokens (role tags, begin/end markers, wrappers like <think>, tool tags, BOS/EOS, and template-attached 
          structural whitespace) are labeled with role=None.
        - We never default to a role. Tokens are labeled only when they fall inside a recognized content span for the given template.

    Returns:
        The original df with new columns:
        - role: str in {system, user, developer, cot, assistant, commentary, tool}, or None
        - is_content: bool
        - seg_ix: int or None
        - token_in_seg_ix: int or None
    """
    required_cols = {"prompt_ix", "token_ix", "token"}
    missing = required_cols - set(sample_df.columns)
    if missing:
        raise ValueError(f"sample_df missing required columns: {sorted(missing)}")

    dispatch = {
        'gpt-oss-20b': label_gpt_oss_content_roles,
        'gpt-oss-120b': label_gpt_oss_content_roles
    }

    try:
        fn = dispatch[model_prefix]
    except KeyError:
        raise ValueError(f"Model architecture {model_prefix} not supported")

    return fn(sample_df)