"""Contrastive decoding package providing a CLI and reusable primitives"""

from .config import CDConfig
from .decoding import contrastive_decode
from .utils import DEFAULT_SYSTEM_PROMPT, ensure_padding_token, format_prompt, load_model, pick_device_dtype

__all__ = [
    "CDConfig",
    "contrastive_decode",
    "DEFAULT_SYSTEM_PROMPT",
    "ensure_padding_token",
    "format_prompt",
    "load_model",
    "pick_device_dtype",
]
