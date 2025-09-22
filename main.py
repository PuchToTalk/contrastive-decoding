import torch
import torch.nn.functional as F
import transformers as tr

amateur_path = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
expert_path = "Qwen/Qwen2.5-1.5B-Instruct"

# Advanced pass: contrastive decoding with KV cache, plausibility mask, and top-k pruning.
tokenizer = tr.AutoTokenizer.from_pretrained(amateur_path)
amateur = tr.AutoModelForCausalLM.from_pretrained(amateur_path).eval()
expert = tr.AutoModelForCausalLM.from_pretrained(expert_path).eval()


def contrastive_decode(
    prompt: str,
    max_new_tokens: int = 40,
    alpha: float = 0.1,
    top_k: int = 50,
) -> str:
    """Contrastive decoding that reuses KV cache and prunes the candidate set."""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    device = input_ids.device

    with torch.no_grad():
        out_e = expert(input_ids, use_cache=True)
        out_a = amateur(input_ids, use_cache=True)

    past_e, past_a = out_e.past_key_values, out_a.past_key_values
    logits_e = out_e.logits[:, -1, :]
    logits_a = out_a.logits[:, -1, :]

    generated: list[int] = []

    for _ in range(max_new_tokens):
        logpe = F.log_softmax(logits_e, dim=-1)
        probs_e = logpe.exp()

        threshold = alpha * probs_e.max(dim=-1, keepdim=True).values
        mask = probs_e >= threshold

        if top_k > 0:
            top_k = min(top_k, probs_e.size(-1))
            _, topk_idx = torch.topk(probs_e, k=top_k)
            valid_mask = mask[0, topk_idx[0]]
            cand_idx = topk_idx[0][valid_mask]
        else:
            cand_idx = torch.nonzero(mask[0], as_tuple=False).flatten()

        if cand_idx.numel() == 0:
            next_token_id = int(torch.argmax(probs_e, dim=-1).item())
        else:
            logpa = F.log_softmax(logits_a, dim=-1)
            scores = logpe[0, cand_idx] - logpa[0, cand_idx]
            next_token_id = int(cand_idx[torch.argmax(scores)].item())

        if next_token_id == tokenizer.eos_token_id:
            break

        generated.append(next_token_id)

        next_token = torch.tensor([[next_token_id]], dtype=torch.long, device=device)

        with torch.no_grad():
            out_e = expert(next_token, past_key_values=past_e, use_cache=True)
            logits_e = out_e.logits[:, -1, :]
            past_e = out_e.past_key_values

            out_a = amateur(next_token, past_key_values=past_a, use_cache=True)
            logits_a = out_a.logits[:, -1, :]
            past_a = out_a.past_key_values

    return tokenizer.decode(generated, skip_special_tokens=True)


if __name__ == "__main__":
    user_message = """Give a very very brief docstring for the following function:\n```\nfunction updateEloScores(\n\tscores,\n\tresults,\n\tkFactor = 4,\n) {\n\tfor (const result of results) {\n\t\tconst { first, second, outcome } = result;\n\t\tconst firstScore = scores[first] ?? 1000;\n\t\tconst secondScore = scores[second] ?? 1000;\n\n\t\tconst expectedScoreFirst = 1 / (1 + Math.pow(10, (secondScore - firstScore) / 400));\n\t\tconst expectedScoreSecond = 1 / (1 + Math.pow(10, (firstScore - secondScore) / 400));\n\t\tlet sa = 0.5;\n\t\tif (outcome === 1) {\n\t\t\tsa = 1;\n\t\t} else if (outcome === -1) {\n\t\t\tsa = 0;\n\t\t}\n\t\tscores[first] = firstScore + kFactor * (sa - expectedScoreFirst);\n\t\tscores[second] = secondScore + kFactor * (1 - sa - expectedScoreSecond);\n\t}\n\treturn scores;\n}\n```"""

    prompt = tokenizer.apply_chat_template(
        [
            {
                "role": "system",
                "content": "You are a helpful assistant that writes concise docstrings.",
            },
            {"role": "user", "content": user_message},
        ],
        add_generation_prompt=True,
        tokenize=False,
    )

    print("\n--- Contrastive Decoding Output ---")
    print(contrastive_decode(prompt, max_new_tokens=40, alpha=0.1, top_k=50))
