#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from typing import Iterable, Optional

import transformers as tr

from contrastive_decoding.config import CDConfig
from contrastive_decoding.decoding import contrastive_decode
from contrastive_decoding.utils import (
    DEFAULT_SYSTEM_PROMPT,
    ensure_padding_token,
    format_prompt,
    load_model,
    pick_device_dtype,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Contrastive Decoding with expert/amateur LMs")
    parser.add_argument("--expert", default="Qwen/Qwen2.5-3B-Instruct", help="Expert model identifier")
    parser.add_argument("--amateur", default="Qwen/Qwen2.5-Coder-0.5B-Instruct", help="Amateur model identifier")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Maximum number of new tokens to sample")
    parser.add_argument("--alpha", type=float, default=0.1, help="Expert plausibility threshold multiplier")
    parser.add_argument("--top-k", type=int, default=64, help="Top-k pruning size (0 disables)")
    parser.add_argument("--tau-amateur", type=float, default=1.0, help="Temperature applied to amateur logits")
    parser.add_argument("--no-stream", action="store_true", help="Disable token streaming to stdout")
    parser.add_argument("--no-chat-template", action="store_true", help="Skip chat template formatting")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt text; reads stdin if omitted")
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt used when chat templates are enabled",
    )
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    device, dtype = pick_device_dtype()

    tokenizer = tr.AutoTokenizer.from_pretrained(args.expert)
    ensure_padding_token(tokenizer)

    expert = load_model(args.expert, device, dtype)
    amateur = load_model(args.amateur, device, dtype)

    if args.prompt is None:
        print("[INPUT] Enter prompt, then Ctrl-D:")
        user_prompt = sys.stdin.read().strip()
    else:
        user_prompt = args.prompt

    formatted_prompt = format_prompt(
        tokenizer,
        user_prompt,
        use_chat_template=not args.no_chat_template,
        system_prompt=args.system_prompt,
    )

    cfg = CDConfig(
        max_new_tokens=args.max_new_tokens,
        alpha=args.alpha,
        top_k=args.top_k,
        tau_amateur=args.tau_amateur,
        stream=not args.no_stream,
        use_chat_template=not args.no_chat_template,
    )

    print("\n--- Contrastive Decoding Output ---")
    result = contrastive_decode(
        amateur=amateur,
        expert=expert,
        tokenizer=tokenizer,
        prompt=formatted_prompt,
        cfg=cfg,
        device=device,
    )

    if not cfg.stream:
        print(result)


if __name__ == "__main__":
    main()
