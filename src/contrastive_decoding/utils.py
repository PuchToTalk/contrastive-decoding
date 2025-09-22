from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import transformers as tr

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant that writes concise docstrings"


def pick_device_dtype() -> Tuple[torch.device, torch.dtype]:
    """Select the fastest available device and an appropriate dtype"""
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability(0)
        dtype = torch.bfloat16 if major >= 8 else torch.float16
        return torch.device("cuda"), dtype
    if torch.backends.mps.is_available():
        return torch.device("mps"), torch.float32
    return torch.device("cpu"), torch.float32


def load_model(path: str, device: torch.device, dtype: torch.dtype) -> tr.PreTrainedModel:
    """Load a causal LM on the requested device, respecting dtype constraints"""
    torch_dtype = dtype if device.type == "cuda" else torch.float32
    model = tr.AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch_dtype,
        device_map="auto" if device.type == "cuda" else None,
    )
    if device.type in ("cpu", "mps"):
        model = model.to(device)
    return model.eval()


def format_prompt(
    tokenizer: tr.PreTrainedTokenizerBase,
    user_prompt: str,
    use_chat_template: bool = True,
    system_prompt: Optional[str] = DEFAULT_SYSTEM_PROMPT,
    chat_messages: Optional[Sequence[dict]] = None,
) -> str:
    """Format the raw prompt, optionally using the tokenizer's chat template"""
    if not use_chat_template or not hasattr(tokenizer, "apply_chat_template"):
        return user_prompt

    if chat_messages is None:
        messages: List[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
    else:
        messages = list(chat_messages)

    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )


def ensure_padding_token(tokenizer: tr.PreTrainedTokenizerBase) -> None:
    """Set a padding token for models that require one during batching"""
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
