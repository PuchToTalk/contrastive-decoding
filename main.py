#!/usr/bin/env python3
import argparse
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import transformers as tr


@dataclass
class CDConfig:
    max_new_tokens: int = 128
    alpha: float = 0.1
    top_k: int = 64
    tau_amateur: float = 1.0
    stream: bool = True
    use_chat_template: bool = True


def pick_device_dtype() -> Tuple[torch.device, torch.dtype]:
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability(0)
        return torch.device("cuda"), (torch.bfloat16 if major >= 8 else torch.float16)
    if torch.backends.mps.is_available():
        return torch.device("mps"), torch.float32
    return torch.device("cpu"), torch.float32


def load_model(path: str, device: torch.device, dtype: torch.dtype) -> tr.PreTrainedModel:
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
    use_chat: bool,
    system_prompt: Optional[str] = None,
) -> str:
    if use_chat and hasattr(tokenizer, "apply_chat_template"):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
    return user_prompt


@torch.no_grad()
def contrastive_decode(
    amateur: tr.PreTrainedModel,
    expert: tr.PreTrainedModel,
    tokenizer: tr.PreTrainedTokenizerBase,
    prompt: str,
    cfg: CDConfig,
    device: torch.device,
    system_prompt: Optional[str] = None,
) -> str:
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = enc["input_ids"].to(device)

    out_e = expert(input_ids, use_cache=True)
    out_a = amateur(input_ids, use_cache=True)
    past_e, past_a = out_e.past_key_values, out_a.past_key_values
    logits_e, logits_a = out_e.logits[:, -1, :], out_a.logits[:, -1, :]

    generated: list[int] = []
    eos_id = tokenizer.eos_token_id

    for _ in range(cfg.max_new_tokens):
        logpe = F.log_softmax(logits_e, dim=-1)
        probs_e = logpe.exp()

        threshold = cfg.alpha * probs_e.max()
        mask = probs_e >= threshold

        if cfg.top_k > 0:
            top_k = min(cfg.top_k, probs_e.size(-1))
            _, topk_idx = torch.topk(probs_e, k=top_k)
            valid_mask = mask[0, topk_idx[0]]
            cand_idx = topk_idx[0][valid_mask]
        else:
            cand_idx = torch.nonzero(mask[0], as_tuple=False).flatten()

        if cand_idx.numel() == 0:
            next_id = int(torch.argmax(probs_e))
        else:
            logits_a_adj = logits_a / cfg.tau_amateur
            logpa = F.log_softmax(logits_a_adj, dim=-1)
            scores = logpe[0, cand_idx] - logpa[0, cand_idx]
            next_id = int(cand_idx[torch.argmax(scores)])

        if eos_id is not None and next_id == eos_id:
            break

        generated.append(next_id)

        if cfg.stream:
            sys.stdout.write(tokenizer.decode([next_id], skip_special_tokens=True))
            sys.stdout.flush()

        next_token = torch.tensor([[next_id]], dtype=torch.long, device=device)

        out_e = expert(next_token, past_key_values=past_e, use_cache=True)
        out_a = amateur(next_token, past_key_values=past_a, use_cache=True)
        logits_e, logits_a = out_e.logits[:, -1, :], out_a.logits[:, -1, :]
        past_e, past_a = out_e.past_key_values, out_a.past_key_values

    if cfg.stream and generated:
        print()

    return tokenizer.decode(generated, skip_special_tokens=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Contrastive Decoding with Qwen models")
    parser.add_argument("--expert", default="Qwen/Qwen2.5-3B-Instruct", help="Expert model identifier")
    parser.add_argument("--amateur", default="Qwen/Qwen2.5-Coder-0.5B-Instruct", help="Amateur model identifier")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Maximum new tokens to generate")
    parser.add_argument("--alpha", type=float, default=0.1, help="Expert plausibility threshold multiplier")
    parser.add_argument("--top-k", type=int, default=64, help="Top-k pruning size (0 disables)")
    parser.add_argument("--tau-amateur", type=float, default=1.0, help="Temperature applied to amateur logits")
    parser.add_argument("--no-stream", action="store_true", help="Disable token streaming")
    parser.add_argument("--no-chat-template", action="store_true", help="Skip chat template formatting")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt text; reads stdin if omitted")
    parser.add_argument(
        "--system-prompt",
        type=str,
        default="You are a helpful assistant that writes concise docstrings.",
        help="System prompt used when chat template is enabled",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    device, dtype = pick_device_dtype()

    tokenizer = tr.AutoTokenizer.from_pretrained(args.expert)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

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
        use_chat=not args.no_chat_template,
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
        system_prompt=args.system_prompt,
    )

    if not cfg.stream:
        print(result)


if __name__ == "__main__":
    # Default behavior: run with a demo prompt if no CLI args are given
    import sys
    if len(sys.argv) == 1:
        # No CLI args → use a built-in example
        demo_prompt = """Give a very very brief docstring for the following function:
        ```
        function updateEloScores(scores, results, kFactor = 4) {
            // updates scores...
        }
        ```"""
        sys.argv.extend([
            "--prompt", demo_prompt,
            "--expert", "Qwen/Qwen2.5-1.5B-Instruct",   # lighter for local run
            "--amateur", "Qwen/Qwen2.5-Coder-0.5B-Instruct",
            "--max-new-tokens", "40",
            "--alpha", "0.1",
            "--top-k", "50",
        ])
    main()