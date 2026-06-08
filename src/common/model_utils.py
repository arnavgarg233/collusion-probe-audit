"""Model loading, generation, and residual-stream extraction (the primary readout).

Kept tiny and dependency-light so generate_conversations.py and extract_activations.py
share exactly one definition of "the layer at ~0.7 depth, mean-pooled over the
response span".
"""
from __future__ import annotations
# --- path bootstrap: make sibling src/ subpackages importable as flat modules ---
import os as _os, sys as _sys
_SRCROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _sub in ('', 'common', 'synthetic', 'narcbench', 'audit', 'exploratory'):
    _pp = _os.path.join(_SRCROOT, _sub)
    if _os.path.isdir(_pp) and _pp not in _sys.path:
        _sys.path.insert(0, _pp)
# --- end bootstrap ---
import numpy as np
import config

_MODEL = None
_TOK = None


def load(model_name: str = config.MODEL_NAME):
    """Load (and cache) the tokenizer + model in fp16 on the resolved device."""
    global _MODEL, _TOK
    if _MODEL is not None:
        return _MODEL, _TOK
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = config.resolve_device()
    dtype = torch.float16 if device != "cpu" else torch.float32
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    except TypeError:  # older transformers
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    model.to(device).eval()
    _MODEL, _TOK = model, tok
    print(f"[model] {model_name} on {device} ({dtype}), "
          f"{model.config.num_hidden_layers} layers")
    return model, tok


def _prompt_ids(tok, messages):
    out = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    # transformers may return a tensor or a BatchEncoding/dict depending on version
    if hasattr(out, "input_ids"):
        return out.input_ids
    if isinstance(out, dict):
        return out["input_ids"]
    return out


def generate_response(messages, max_new_tokens: int = config.MAX_NEW_TOKENS) -> str:
    import torch
    model, tok = load()
    device = config.resolve_device()
    ids = _prompt_ids(tok, messages).to(device)
    attn = torch.ones_like(ids)
    torch.manual_seed(config.SEED)              # reproducible sampling (do_sample=True draws an RNG)
    with torch.no_grad():
        out = model.generate(
            ids, attention_mask=attn, max_new_tokens=max_new_tokens, do_sample=True,
            temperature=config.GEN_TEMPERATURE, top_p=config.GEN_TOP_P,
            pad_token_id=tok.pad_token_id,
        )
    text = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
    return text


def response_hidden(messages, response_text: str,
                    layer_frac: float = config.LAYER_FRAC, pool: str = config.POOL):
    """Forward-pass prompt+response, return mean(/last/max)-pooled residual at the
    layer nearest `layer_frac` of depth, over the RESPONSE token span only.
    Returns (vector[np.float32], layer_index, n_layers).
    """
    import torch
    model, tok = load()
    device = config.resolve_device()
    p_ids = _prompt_ids(tok, messages)
    r_ids = tok(response_text, return_tensors="pt", add_special_tokens=False).input_ids
    full = torch.cat([p_ids, r_ids], dim=1).to(device)
    with torch.no_grad():
        out = model(full, output_hidden_states=True)
    hs = out.hidden_states                      # tuple len (L+1), each [1, seq, H]
    L = len(hs) - 1
    layer = max(1, min(L, int(round(layer_frac * L))))
    h = hs[layer][0]                            # [seq, H]
    start = p_ids.shape[1]
    span = h[start:] if h.shape[0] > start else h[-1:]
    if span.shape[0] == 0:
        span = h[-1:]
    if pool == "mean":
        v = span.mean(0)
    elif pool == "last":
        v = span[-1]
    elif pool == "max":
        v = span.max(0).values
    else:
        raise ValueError(f"unknown pool {pool}")
    return v.float().cpu().numpy().astype(np.float32), layer, L
