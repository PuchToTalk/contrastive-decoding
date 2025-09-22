from __future__ import annotations

import sys
from typing import Optional, TextIO

import torch
import torch.nn.functional as F
import transformers as tr

from contrastive_decoding.config import CDConfig


@torch.no_grad()
def contrastive_decode(
    amateur: tr.PreTrainedModel,
    expert: tr.PreTrainedModel,
    tokenizer: tr.PreTrainedTokenizerBase,
    prompt: str,
    cfg: CDConfig,
    device: torch.device,
    stream_writer: Optional[TextIO] = None,
) -> str:
    """Run contrastive decoding using an expert/amateur model pair."""
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = enc["input_ids"].to(device)

    out_e = expert(input_ids, use_cache=True)
    out_a = amateur(input_ids, use_cache=True)
    past_e, past_a = out_e.past_key_values, out_a.past_key_values
    logits_e, logits_a = out_e.logits[:, -1, :], out_a.logits[:, -1, :]

    generated: list[int] = []
    eos_id = tokenizer.eos_token_id
    writer = stream_writer or sys.stdout

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
            next_id = int(torch.argmax(probs_e).item())
        else:
            logits_a_adj = logits_a / cfg.tau_amateur
            logpa = F.log_softmax(logits_a_adj, dim=-1)
            scores = logpe[0, cand_idx] - logpa[0, cand_idx]
            best = torch.argmax(scores)
            next_id = int(cand_idx[best].item())

        if eos_id is not None and next_id == eos_id:
            break

        generated.append(next_id)

        if cfg.stream:
            token_text = tokenizer.decode([next_id], skip_special_tokens=True)
            writer.write(token_text)
            writer.flush()

        next_token = torch.tensor([[next_id]], dtype=torch.long, device=device)

        out_e = expert(next_token, past_key_values=past_e, use_cache=True)
        out_a = amateur(next_token, past_key_values=past_a, use_cache=True)
        logits_e, logits_a = out_e.logits[:, -1, :], out_a.logits[:, -1, :]
        past_e, past_a = out_e.past_key_values, out_a.past_key_values

    if cfg.stream:
        writer.write("\n")
        writer.flush()

    return tokenizer.decode(generated, skip_special_tokens=True)
